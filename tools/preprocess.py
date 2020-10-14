from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import tempfile

from laser.data.text_processing import Token, BPEfastApply

# get environment
assert os.environ.get('LASER'), 'Please set the environment variable LASER'
LASER = os.environ['LASER']

BPE_CODES = osp.join(LASER, 'models', '93langs.fcodes')


def preprocess_europarl(lang_pairs, verbose=False):
    raw_dir = osp.join(LASER, 'data', 'europarl', 'raw')
    out_dir = osp.join(LASER, 'data', 'europarl', 'processed')
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for lang_pair in lang_pairs:
            src, tgt = lang_pair.split('-')
            for lang in [src, tgt]:
                filename = f'Europarl.{lang_pair}.{lang}'
                inp_file = osp.join(raw_dir, filename)
                tmp_file = osp.join(tmp_dir, filename)
                Token(
                    inp_file,
                    tmp_file,
                    lang=lang,
                    romanize=True if lang == 'el' else False,
                    lower_case=True,
                    gzip=False,
                    verbose=verbose,
                    over_write=False,
                )

                out_file = osp.join(out_dir, filename)
                BPEfastApply(tmp_file, out_file, BPE_CODES, verbose=verbose, over_write=False)


if __name__ == '__main__':
    preprocess_europarl(("en-es", "de-en", "de-es", "de-fr", "en-fr", "es-fr"))
