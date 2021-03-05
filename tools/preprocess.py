from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as osp
import tempfile

from laser.data.text_processing import LASER, BPEfastApply, Token
from laser.data.utils import get_bpe_codes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alias",
        type=str,
        choices=("21", "93"),
        default="93",
        help="number of languages on which model pretrained",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    args = parser.parse_args()
    return args


def process(inp_file, out_file, lang, bpe_codes, verbose=False):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = osp.join(tmp_dir, "tok")
        romanize = True if lang == "el" else False
        Token(
            inp_file,
            tmp_file,
            lang=lang,
            romanize=romanize,
            lower_case=True,
            gzip=False,
            verbose=verbose,
            over_write=False,
        )
        BPEfastApply(tmp_file, out_file, bpe_codes, verbose=verbose, over_write=False)


def main():
    args = parse_args()

    bpe_codes = get_bpe_codes(args.alias)
    datasets = {
        "Europarl": {
            "lang_pairs": ("en-it",),
            "folder": "europarl",
            "inp_tmpl": "Europarl.{}.{}",
            "oup_tmpl": "train.{}.{}",
        },
        "UNPC": {
            "lang_pairs": ("en-zh",),
            "folder": "unpc",
            "inp_tmpl": "UNPC.{}.{}.2000000",
            "oup_tmpl": "train.{}.{}",
        },
        "XNLI": {
            "lang_pairs": ("en1-en2",),
            "folder": "XNLI-1.0",
            "inp_tmpl": "xnli.{}.{}",
            "oup_tmpl": "train.{}.{}",
        },
    }

    for name, params in datasets.items():
        print(f"Processing {name}")
        raw_dir = osp.join(LASER, "data", params["folder"], "raw")
        oup_dir = osp.join(LASER, "data", params["folder"], f"bpe{args.alias}")
        os.makedirs(oup_dir, exist_ok=True)

        for lang_pair in params["lang_pairs"]:
            src, tgt = lang_pair.split("-")
            for l in (src, tgt):
                inp_file = osp.join(raw_dir, params["inp_tmpl"].format(lang_pair, l))
                oup_file = osp.join(oup_dir, params["oup_tmpl"].format(lang_pair, l))
                print(f" - processing {inp_file}")
                # for our task, we use same language as input
                if len(l) == 3 and l[-1] in {"1", "2"}:
                    l = l[:-1]
                process(inp_file, oup_file, l, bpe_codes, args.verbose)


if __name__ == "__main__":
    main()
