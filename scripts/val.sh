#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

path="${LASER}/checkpoints/laser_lstm/checkpoint30.pt"

oup_file="en-it.out"
fairseq-generate \
    "cfgs/en-it.json" \
    --user-dir laser \
    --task laser \
    --gen-subset valid \
    --path "${path}" \
    --beam 5 --batch-size 128 --remove-bpe | tee ${oup_file}

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

oup_file="it-en.out"
fairseq-generate \
    "cfgs/it-en.json" \
    --user-dir laser \
    --task laser \
    --gen-subset valid \
    --path "${path}" \
    --beam 5 --batch-size 128 --remove-bpe | tee ${oup_file}

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

oup_file="en-zh.out"
fairseq-generate \
    "cfgs/en-zh.json" \
    --user-dir laser \
    --task laser \
    --gen-subset valid \
    --path "${path}" \
    --beam 5 --batch-size 128 --remove-bpe | tee ${oup_file}

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

oup_file="zh-en.out"
fairseq-generate \
    "cfgs/zh-en.json" \
    --user-dir laser \
    --task laser \
    --gen-subset valid \
    --path "${path}" \
    --beam 5 --batch-size 128 --remove-bpe | tee ${oup_file}

grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"

fairseq-score --sys "en-it.out.sys" --ref "en-it.out.ref"
fairseq-score --sys "it-en.out.sys" --ref "it-en.out.ref"
fairseq-score --sys "en-zh.out.sys" --ref "en-zh.out.ref"
fairseq-score --sys "zh-en.out.sys" --ref "zh-en.out.ref"
