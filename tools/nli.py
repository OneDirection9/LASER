from __future__ import absolute_import, division, print_function

import json
import os
import os.path as osp

from laser.data.utils import LASER


def load_nli_jsonl(inp_file: str, language: str) -> list:
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
            if "language" in example and example["language"] != language:
                continue
            # use entailment paris to do training
            if example["gold_label"] != "entailment":
                continue
            res.append(example)

    return res


def dump_to_txt(context: list, oup_file1: str, oup_file2: str) -> None:
    with open(oup_file1, "w") as f1, open(oup_file2, "w") as f2:
        for example in context:
            f1.write(example["sentence1"])
            f1.write("\n")
            f2.write(example["sentence2"])
            f2.write("\n")


def main():
    inp_file = osp.join(LASER, "data", "XNLI-1.0", "xnli.dev.jsonl")

    language, lang_pair = "en", "en1-en2"
    src, tgt = lang_pair.split("-")

    oup_dir = osp.join(LASER, "data", "XNLI-1.0", "raw")
    os.makedirs(oup_dir, exist_ok=True)
    oup_file1 = osp.join(oup_dir, f"xnli.{lang_pair}.{src}")
    oup_file2 = osp.join(oup_dir, f"xnli.{lang_pair}.{tgt}")

    res = load_nli_jsonl(inp_file, language)
    dump_to_txt(res, oup_file1, oup_file2)


if __name__ == "__main__":
    main()
