#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

path="${LASER}/checkpoints/laser_lstm/checkpoint25.pt"

generate() {
  cfg_file=$1
  oup_file=$2

  fairseq-generate \
      "${cfg_file}" \
      --user-dir laser \
      --task laser \
      --gen-subset valid \
      --path "${path}" \
      --beam 5 --batch-size 128 --remove-bpe | tee "${oup_file}"

  grep ^T "${oup_file}" | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.ref"
  grep ^H "${oup_file}" | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "${oup_file}.sys"
}

generate "cfgs/en-it.json" "output/en-it.out"
generate "cfgs/it-en.json" "output/it-en.out"
generate "cfgs/en-zh.json" "output/en-zh.out"
generate "cfgs/zh-en.json" "output/zh-en.out"

fairseq-score --sys "output/en-it.out.sys" --ref "output/en-it.out.ref"
fairseq-score --sys "output/it-en.out.sys" --ref "output/it-en.out.ref"
fairseq-score --sys "output/en-zh.out.sys" --ref "output/en-zh.out.ref"
fairseq-score --sys "output/zh-en.out.sys" --ref "output/zh-en.out.ref"
