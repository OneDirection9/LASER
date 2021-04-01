#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

BPE_VOCAB="${LASER}/models/93langs.fvocab"

function binarize() {
  bpe="$1"
  data_bin="$2"
  shift 2
  lang_pairs=( "$@" )

  for lang_pair in "${lang_pairs[@]}"; do
    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    save_dir="${data_bin}/${lang_pair}"
    mkdir -p "${save_dir}"
    /bin/rm "${save_dir}/dict.${src}.txt" "${save_dir}/dict.${tgt}.txt"
    fairseq-preprocess --source-lang "${src}" --target-lang "${tgt}" \
        --trainpref "${bpe}/train.${src}-${tgt}" \
        --joined-dictionary --tgtdict "${BPE_VOCAB}" \
        --destdir "${save_dir}" \
        --dataset-impl lazy \
        --workers 20
    cp -r "${save_dir}" "${data_bin}/${tgt}-${src}"
  done
}

lang_pairs=( "en-it" )
binarize "${LASER}/data/Europarl/bpe93" "${LASER}/data/Europarl" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/news-commentary/bpe93" "${LASER}/data/news-commentary" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/WikiMatrix/bpe93" "${LASER}/data/WikiMatrix" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/XNLI/bpe93" "${LASER}/data/XNLI" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/snli/bpe93" "${LASER}/data/snli" "${lang_pairs[@]}"

lang_pairs=( "zh-en" "ita-eng" )
binarize "${LASER}/data/tatoeba/bpe93" "${LASER}/data/tatoeba" "${lang_pairs[@]}"
