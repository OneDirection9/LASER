#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi


data_bin="${LASER}/data/train"
checkpoint="${LASER}/checkpoints/laser_lstm"
mkdir -p "${checkpoint}"

fairseq-train "${data_bin}" \
  --max-epoch 17 \
  --ddp-backend=no_c10d \
  --task translation_laser --arch laser \
  --encoder-model-path "${LASER}/models/bilstm.93langs.2018-12-26.pt" \
  --fix-encoder \
  --lang-pairs en-it,it-en,en-zh,zh-en \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.001 --criterion cross_entropy \
  --dropout 0.1 --save-dir "${checkpoint}" \
  --max-tokens 12000 --fp16 \
  --valid-subset train --disable-validation \
  --no-progress-bar --log-interval 1000 \
  --user-dir laser/
