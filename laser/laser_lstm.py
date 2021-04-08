# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from .data.utils import LASER

logger = logging.getLogger(__name__)


@register_model("laser_lstm")
class LSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens=None,
        tgt_tokens=None,
        tgt_lengths=None,
        target_language_id=None,
        dataset_name="",
    ):
        assert target_language_id is not None

        src_encoder_out = self.encoder(src_tokens, src_lengths, dataset_name, target_language_id)
        return self.decoder(prev_output_tokens, src_encoder_out)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-embed-path",
            default=None,
            type=str,
            metavar="STR",
            help="path to pre-trained encoder embedding",
        )
        parser.add_argument(
            "--encoder-hidden-size", type=int, metavar="N", help="encoder hidden size"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="number of encoder layers"
        )
        parser.add_argument(
            "--encoder-bidirectional",
            action="store_true",
            help="make all layers of encoder bidirectional",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-embed-path",
            default=None,
            type=str,
            metavar="STR",
            help="path to pre-trained decoder embedding",
        )
        parser.add_argument(
            "--decoder-hidden-size", type=int, metavar="N", help="decoder hidden size"
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="number of decoder layers"
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument(
            "--decoder-zero-init",
            type=str,
            metavar="BOOL",
            help="initialize the decoder hidden/cell state to zero",
        )
        parser.add_argument(
            "--decoder-lang-embed-dim",
            type=int,
            metavar="N",
            help="decoder language embedding dimension",
        )
        parser.add_argument(
            "--fixed-embeddings",
            action="store_true",
            help="keep embeddings fixed (ENCODER ONLY)",
        )  # TODO Also apply to decoder embeddings?

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            "--encoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for encoder input embedding",
        )
        parser.add_argument(
            "--encoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for encoder output",
        )
        parser.add_argument(
            "--decoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for decoder input embedding",
        )
        parser.add_argument(
            "--decoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for decoder output",
        )

        parser.add_argument(
            "--encoder-path",
            type=str,
            default=None,
            help="relative path to pretrained encoder model relative to LASER",
        )
        parser.add_argument(
            "--laser-path",
            type=str,
            default=None,
            help="relative path to pretrained laser model relative to LASER",
        )
        parser.add_argument(
            "--fixed-encoder", action="store_true", help="keep encoder parameters not updated"
        )
        parser.add_argument(
            "--fixed-decoder", action="store_true", help="keep decoder parameters not updated"
        )
        parser.add_argument(
            "--num-controller-layers", type=int, default=0, help="number of controller layers"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        pretrained_encoder_embed = None
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim
            )

        num_langs = task.num_tasks if hasattr(task, "num_tasks") else 0

        assert (
            args.encoder_path is None or args.laser_path is None
        ), "can't set encoder_path and laser_path together"

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            fixed_embeddings=args.fixed_embeddings,
            num_controller_layers=args.num_controller_layers,
        )

        if args.encoder_path is not None:
            encoder = cls.load_pretrained(encoder, args.encoder_path, strict=True)

        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            zero_init=options.eval_bool(args.decoder_zero_init),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            num_langs=num_langs,
            lang_embed_dim=args.decoder_lang_embed_dim,
        )

        def fix_model(model: torch.nn.Module, exclusive=None):
            logger.info(f"Fixing {model.__class__.__name__} parameters")
            for n, p in model.named_parameters():
                if exclusive is not None and n.startswith(exclusive):
                    continue
                p.requires_grad = False

        if args.fixed_encoder:
            fix_model(encoder, "controller")
        if args.fixed_decoder:
            fix_model(decoder)

        laser = cls(encoder, decoder)

        if args.laser_path is not None:
            laser = cls.load_pretrained(laser, args.laser_path)

        return laser

    @staticmethod
    def load_pretrained(model: nn.Module, pretrained_path, strict=False) -> nn.Module:
        model_path = osp.join(LASER, pretrained_path)
        logger.info(f"Creating {model.__class__.__name__} from {model_path}")
        x = torch.load(model_path)

        model.load_state_dict(x["model"], strict=strict)
        return model


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
        pretrained_embed=None,
        padding_value=0.0,
        fixed_embeddings=False,
        num_controller_layers=0,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        if fixed_embeddings:
            self.embed_tokens.weight.requires_grad = False

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        if num_controller_layers > 0:
            self.controller = NNController(d_model=self.output_units, dropout=dropout_out)

    def forward(self, src_tokens, src_lengths, dataset_name, target_language_id):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        try:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())
        except BaseException:
            raise Exception(f"Packing failed in dataset {dataset_name}")

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        if self.controller is None:
            # controller has dropout in PositionalEncoding
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        if self.controller is not None:
            padding_mask = src_tokens.eq(self.padding_idx)
            x = self.controller(x, src_key_padding_mask=padding_mask)

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask if encoder_padding_mask.any() else None,
            "target_language_id": target_language_id,
        }

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        encoder_out_dict["sentemb"] = encoder_out_dict["sentemb"].index_select(0, new_order)
        encoder_out_dict["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out_dict["encoder_out"]
        )
        if encoder_out_dict["encoder_padding_mask"] is not None:
            encoder_out_dict["encoder_padding_mask"] = encoder_out_dict[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        zero_init=False,
        encoder_embed_dim=512,
        encoder_output_units=512,
        pretrained_embed=None,
        num_langs=1,
        lang_embed_dim=0,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=encoder_output_units + embed_dim + lang_embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        if zero_init:
            self.sentemb2init = None
        else:
            self.sentemb2init = Linear(encoder_output_units, 2 * num_layers * hidden_size)

        if lang_embed_dim == 0:
            self.embed_lang = None
        else:
            self.embed_lang = nn.Embedding(num_langs, lang_embed_dim)
            nn.init.uniform_(self.embed_lang.weight, -0.1, 0.1)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        sentemb = encoder_out["sentemb"]
        _encoder_out = encoder_out["encoder_out"]
        lang_id = encoder_out["target_language_id"]

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = _encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # embed language identifier
        if self.embed_lang is not None:
            lang_ids = prev_output_tokens.data.new_full((bsz,), lang_id)
            langemb = self.embed_lang(lang_ids)
            # TODO Should we dropout here???

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, "cached_state")
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            if self.sentemb2init is None:
                prev_hiddens = [
                    x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)
                ]
                prev_cells = [x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)]
            else:
                init = self.sentemb2init(sentemb)
                prev_hiddens = [
                    init[:, (2 * i) * self.hidden_size : (2 * i + 1) * self.hidden_size]
                    for i in range(num_layers)
                ]
                prev_cells = [
                    init[
                        :,
                        (2 * i + 1) * self.hidden_size : (2 * i + 2) * self.hidden_size,
                    ]
                    for i in range(num_layers)
                ]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            if self.embed_lang is None:
                input = torch.cat((x[j, :, :], sentemb), dim=1)
            else:
                input = torch.cat((x[j, :, :], sentemb, langemb), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            "cached_state",
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        if hasattr(self, "additional_fc"):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.fc_out(x)

        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, "cached_state")
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number


class NNController(nn.Module):
    def __init__(
        self,
        max_len: int = 5000,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(NNController, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.d_model = d_model
        self.nhead = nhead

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [sequence length, batch size, embed dim].
            mask:
            src_key_padding_mask: :math:`(S, N)`.

        Returns:

        """
        src = src + self.positional_encoding(src)
        x = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture("laser_lstm", "laser_lstm")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_hidden_size = getattr(args, "encoder_hidden_size", args.encoder_embed_dim)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", args.decoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.decoder_zero_init = getattr(args, "decoder_zero_init", "0")
    args.decoder_lang_embed_dim = getattr(args, "decoder_lang_embed_dim", 0)
    args.fixed_embeddings = getattr(args, "fixed_embeddings", False)

    args.encoder_path = getattr(args, "encoder_path", None)
    args.laser_path = getattr(args, "laser_path", None)
    args.fixed_encoder = getattr(args, "fixed_encoder", False)
    args.fixed_decoder = getattr(args, "fixed_decoder", False)
    args.num_controller_layers = getattr(args, "num_controller_layers", 0)
