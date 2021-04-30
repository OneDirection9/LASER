from __future__ import absolute_import, division, print_function

import os
import os.path as osp

# get environment
assert os.environ.get("LASER"), "Please set the environment variable LASER"
LASER = os.environ["LASER"]

ALIAS_TO_CHECKPOINT_NAME = {
    "21": "bilstm.eparl21.2018-11-19.pt",
    "93": "bilstm.93langs.2018-12-26.pt",
}

ALIAS_TO_BPE_CODES_NAME = {
    "21": "eparl21.fcodes",
    "93": "93langs.fcodes",
}

ALIAS_TO_VOCAB_NAME = {
    "21": "eparl21.fvocab",
    "93": "93langs.fvocab",
}

MODEL_DIR = osp.join(LASER, "models")

# a special character indicting the start of the persona
PERSONA_SEP_CHAR = "😈"
PERSONA_SEP_CHAR_IDX = 73640


def get_checkpoint(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_CHECKPOINT_NAME[alias])


def get_bpe_codes(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_BPE_CODES_NAME[alias])


def get_vocab(alias: str) -> str:
    return osp.join(MODEL_DIR, ALIAS_TO_VOCAB_NAME[alias])
