from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as osp
import tempfile

from laser.data.text_processing import Token, BPEfastApply, LASER
from laser.data.utils import get_bpe_codes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--alias', type=str, choices=('21', '93'), default='93',
        help='number of languages on which model pretrained'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Detailed output'
    )

    args = parser.parse_args()
    return args


def process(inp_file, out_file, lang, bpe_codes, verbose=False):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = osp.join(tmp_dir, 'tok')
        romanize = True if lang == 'el' else False
        Token(
            inp_file, tmp_file, lang=lang, romanize=romanize, lower_case=True, gzip=False,
            verbose=verbose, over_write=False
        )
        BPEfastApply(tmp_file, out_file, bpe_codes, verbose=verbose, over_write=False)


def process_europarl(lang_pairs, alias, verbose=False):
    bpe_codes = get_bpe_codes(alias)

    raw_dir = osp.join(LASER, 'data', 'europarl', 'raw')
    out_dir = osp.join(LASER, 'data', 'europarl', f'bpe{alias}')
    os.makedirs(out_dir, exist_ok=True)

    for lang_pair in lang_pairs:
        # example: en-es
        src, tgt = lang_pair.split('-')
        for l in (src, tgt):
            inp_file = osp.join(raw_dir, f'Europarl.{lang_pair}.{l}')
            out_file = osp.join(out_dir, f'train.{lang_pair}.{l}')
            process(inp_file, out_file, l, bpe_codes, verbose)


def main():
    args = parse_args()

    lang_pairs = ("en-es", "de-en", "de-es", "de-fr", "en-fr", "es-fr")
    process_europarl(lang_pairs, args.alias, args.verbose)


if __name__ == '__main__':
    main()
