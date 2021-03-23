#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

oup_file="tmp.out"
fairseq-generate \
    "cfgs/laser.json" \
    --user-dir laser \
    --task laser \
    --gen-subset valid \
    --path "${LASER}/checkpoint_last.pt" \
    --beam 5 --batch-size 128 --remove-bpe | tee ${oup_file}

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

fairseq-score --sys "${oup_file}.sys" --ref "${oup_file}.ref"
