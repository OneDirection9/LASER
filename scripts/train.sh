#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi


fairseq-train \
  "${LASER}/cfgs/laser.json" \
  --user-dir laser \
  --log-interval 100 --log-format simple \
  --task laser --arch laser_lstm \
  --encoder-path "${LASER}/models/bilstm.93langs.2018-12-26.pt" \
  --fixed-encoder \
  --save-dir ./checkpoints/laser_lstm \
  --tensorboard-logdir ./checkpoints/laser_lstm/log \
  --fp16 \
  --optimizer adam \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --clip-norm 5 \
  --warmup-updates 90000 \
  --update-freq 2 \
  --dropout 0.0 \
  --encoder-dropout-out 0.1 \
  --max-tokens 2000 \
  --max-epoch 50 \
  --encoder-bidirectional \
  --encoder-layers 5 \
  --encoder-hidden-size 512 \
  --encoder-embed-dim 320 \
  --decoder-layers 1 \
  --decoder-hidden-size 2048 \
  --decoder-embed-dim 320 \
  --decoder-lang-embed-dim 32 \
  --warmup-init-lr 0.001 \
  --disable-validation
