# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import torch
from fairseq.data import BaseWrapperDataset, FairseqDataset, LanguagePairDataset, iterators
from fairseq.data.language_pair_dataset import collate

from .data.utils import PERSONA_SEP_CHAR_ID


class MultiItr(object):
    def __init__(self, itr):
        self.itr = itr
        self._counts = [0 for x in itr]

    def __len__(self):
        return sum(len(itr) for itr in self.itr)

    def __iter__(self):
        return self

    def __next__(self):
        ratios = [count / len(itr) for count, itr in zip(self._counts, self.itr)]
        idx = ratios.index(min(ratios))
        self._counts[idx] += 1
        return next(self.itr[idx])


class MultidatasetEpochBatchIterator(iterators.EpochBatchIterating):
    """A wrapper around multiple epoch batch iterators."""

    def __init__(
        self,
        dataset,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):

        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)

        self.iterators = []

        self.epoch = epoch
        for key, dt in dataset.items():
            epoch_iter = iterators.EpochBatchIterator(
                dataset=dt,
                collate_fn=dt.collater,
                batch_sampler=batch_sampler[key],
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=0,
                epoch=epoch,
            )
            self.iterators.append(epoch_iter)

    def __len__(self):
        return sum(len(itr) for itr in self.iterators)

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        # `self.epoch += 1` should be handled by underlying `EpochBatchIterator`s.
        return MultiItr(
            [
                itr.next_epoch_itr(shuffle=shuffle, fix_batches_to_gpus=fix_batches_to_gpus)
                for itr in self.iterators
            ]
        )

    def end_of_epoch(self):
        return all(itr.end_of_epoch() for itr in self.iterators)

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""

        epochs = [itr.next_epoch_idx for itr in self.iterators]
        self.epoch = epochs[0]
        assert all(epoch == self.epoch for epoch in epochs)

        return self.epoch

    @property
    def iterations_in_epoch(self):
        return sum(itr.iterations_in_epoch for itr in self.iterators)

    def state_dict(self):
        return {
            "iterators": [it.state_dict() for it in self.iterators],
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        for it, d in zip(self.iterators, state_dict["iterators"]):
            it.load_state_dict(d)


class MultitaskDatasetWrapper(BaseWrapperDataset):
    """A wrapper for a multitask dataset."""

    def __init__(self, dataset, target_language_id, sample=1.0, name=""):
        super().__init__(dataset)
        self.target_language_id = target_language_id
        self.sample = sample
        self.name = name

    def collater(self, *args, **kwargs):
        ans = self.dataset.collater(*args, **kwargs)
        if "net_input" in ans:
            ans["net_input"]["target_language_id"] = self.target_language_id
            ans["net_input"]["dataset_name"] = self.name
        return ans

    def num_tokens(self, *args, **kwargs):
        return self.dataset.num_tokens(*args, **kwargs)

    def ordered_indices(self, *args, **kwargs):
        indices = self.dataset.ordered_indices(*args, **kwargs)
        # Hacky solution for sampling
        size = int(self.sample * indices.shape[0])

        return indices.take(np.sort(np.random.permutation(indices.shape[0])[:size]))

    def size(self, index: int):
        return self.dataset.size(index)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)


class DialogDataset(LanguagePairDataset):
    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        # Separate out the persona list
        # The original src_item has following format:
        #     dialog <SEP> persona <SEP> persona <SEP>...
        sep_ind = (src_item == PERSONA_SEP_CHAR_ID).nonzero(as_tuple=True)[0]
        start = sep_ind + 1
        end = torch.cat([sep_ind[1:], sep_ind.new_tensor([src_item.numel()])])
        personas = [src_item[s:e] for s, e in zip(start, end)]
        src_item = src_item[: sep_ind[0]]

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "personas": personas,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        personas_list = [sample.pop("personas") for sample in samples]
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )

        res["net_input"]["personas_list"] = personas_list
        return res
