from __future__ import absolute_import, division, print_function

import json
import os
import os.path as osp

from laser.data.utils import LASER


def load_nli_jsonl(inp_file: str) -> list:
    """
    Load entailment sentence pairs from XNLI/MNLI jsonl file.

    Examples:
    ::
        # XNLI example
        {"annotator_labels":
            ["contradiction", "contradiction", "contradiction", "contradiction", "contradiction"],
         "genre": "facetoface",
         "gold_label": "contradiction",
         "language": "en",
         "match": "True",
         "pairID": "2",
         "promptID": "1",
         "sentence1": "And he said, Mama, I'm home.",
         "sentence1_tokenized": "And he said , Mama , I 'm home .",
         "sentence2": "He didn't say a word.",
         "sentence2_tokenized": "He didn 't say a word ."}

         # MNLI example
         {"annotator_labels":
             ["neutral", "entailment", "neutral", "neutral", "neutral"],
          "genre": "slate",
          "gold_label": "neutral",
          "pairID": "63735n",
          "promptID": "63735",
          "sentence1": "The new rights are nice enough",
          "sentence1_binary_parse": "( ( The ( new rights ) ) ( are ( nice enough ) ) )",
          "sentence1_parse": "(ROOT (S (NP (DT The) (JJ new) (NNS rights)) (VP (VBP are)
                             (ADJP (JJ nice) (RB enough)))))",
          "sentence2": "Everyone really likes the newest benefits ",
          "sentence2_binary_parse": "( Everyone ( really ( likes ( the ( newest benefits ) ) ) ) )",
          "sentence2_parse": "(ROOT (S (NP (NN Everyone)) (VP (ADVP (RB really)) (VBZ likes)
                             (NP (DT the) (JJS newest) (NNS benefits)))))"}
    """
    res = []
    with open(inp_file, "r") as f:
        for line in f:
            example = json.loads(line)
            # XNLI contains 15 different languages and we only use English examples
            if "language" in example and example["language"] != "en":
                continue
            # use entailment paris to do training
            if example["gold_label"] != "entailment":
                continue
            res.append(example)

    return res


def dump_to_txt(context: list, oup_tmpl: str) -> None:
    oup_file1 = oup_tmpl.format("e1-e2", "e1")
    oup_file2 = oup_tmpl.format("e1-e2", "e2")
    with open(oup_file1, "w") as f1, open(oup_file2, "w") as f2:
        for example in context:
            f1.write(example["sentence1"])
            f1.write("\n")
            f2.write(example["sentence2"])
            f2.write("\n")


def main():
    inp_file = osp.join(LASER, "data", "XNLI-1.0", "xnli.dev.jsonl")

    res = load_nli_jsonl(inp_file)
    oup_dir = osp.join(LASER, "data", "XNLI-1.0", "raw")
    os.makedirs(oup_dir, exist_ok=True)
    dump_to_txt(res, osp.join(oup_dir, "xnli.{}.{}"))


if __name__ == "__main__":
    main()
