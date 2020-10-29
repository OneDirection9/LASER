#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

data_bin="${LASER}/data/val/binarized"
fairseq-generate "${data_bin}" \
    --task translation_laser \
    --lang-pairs it-en,en-it,zh-en,en-zh \
    --gen-subset train \
    --path "${LASER}/checkpoints/laser_lstm/checkpoint_last.pt" \
    --user-dir laser/ \
    --beam 5 --batch-size 128 --remove-bpe | tee tmp.out

