#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

save_dir="${LASER}/checkpoints/controller_xnli_snli"
mkdir -p ${save_dir}

fairseq-train \
  "cfgs/controller.json" \
  --user-dir laser \
  --log-interval 100 --log-format simple \
  --task laser --arch laser_lstm \
  --laser-path "checkpoints/checkpoint25.pt" \
  --fixed-encoder \
  --fixed-decoder \
  --save-dir "${save_dir}" \
  --tensorboard-logdir "${save_dir}/log" \
  --optimizer adam \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --clip-norm 5 \
  --warmup-updates 1000 \
  --update-freq 2 \
  --dropout 0.0 \
  --encoder-dropout-out 0.1 \
  --ff-dim 2048 \
  --max-tokens 2000 \
  --max-epoch 80 \
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
