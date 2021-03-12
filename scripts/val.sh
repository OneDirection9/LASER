#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

data_bin="${LASER}/data/val/binarized"
oup_file="tmp.out"
fairseq-generate "${data_bin}" \
    --task translation_laser \
    --lang-pairs en-it,it-en,en-zh,zh-en \
    --source-lang zh --target-lang en \
    --gen-subset train \
    --path "${LASER}/checkpoints/laser_new_lstm/checkpoint_last.pt" \
    --num-workers 0 \
    --user-dir laser/ \
    --beam 5 --batch-size 128 --remove-bpe | tee "${oup_file}"

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

fairseq-score --sys "${oup_file}.sys" --ref "${oup_file}.ref"
