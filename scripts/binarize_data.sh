#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

BPE_VOCAB="${LASER}/models/93langs.fvocab"

function binarize() {
  echo "binarizing ${2}"
  bpe="$1"
  data_bin="$2"
  shift 2
  lang_pairs=( "$@" )

  echo "binarizing ${2}"

  for lang_pair in "${lang_pairs[@]}"; do
    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    save_dir="${data_bin}/${lang_pair}"
    mkdir -p "${save_dir}"
    /bin/rm "${save_dir}/dict.${src}.txt" "${save_dir}/dict.${tgt}.txt"
    fairseq-preprocess --source-lang "${src}" --target-lang "${tgt}" \
        --user-dir laser --task laser \
        --trainpref "${bpe}/train.${src}-${tgt}" \
        --joined-dictionary --tgtdict "${BPE_VOCAB}" \
        --destdir "${save_dir}" \
        --dataset-impl lazy \
        --workers 20
    cp -r "${save_dir}" "${data_bin}/${tgt}-${src}"
  done
  echo "  Done"
}

lang_pairs=( "en-it" )
binarize "${LASER}/data/Europarl/bpe93" "${LASER}/data/Europarl" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/news-commentary/bpe93" "${LASER}/data/news-commentary" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/WikiMatrix/bpe93" "${LASER}/data/WikiMatrix" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/mt/bpe93" "${LASER}/data/mt" "${lang_pairs[@]}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/UNPC/bpe93" "${LASER}/data/UNPC" "${lang_pairs[@]}"

lang_pairs=( "zh1-zh2" )
binarize "${LASER}/data/news-crawl/bpe93" "${LASER}/data/news-crawl" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/news-discuss/bpe93" "${LASER}/data/news-discuss" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" "zh1-zh2" )
binarize "${LASER}/data/XNLI/bpe93" "${LASER}/data/XNLI" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/snli/bpe93" "${LASER}/data/snli" "${lang_pairs[@]}"

lang_pairs=( "zh-en" "ita-eng" )
binarize "${LASER}/data/tatoeba/bpe93" "${LASER}/data/tatoeba" "${lang_pairs[@]}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/convai2/bpe93" "${LASER}/data/convai2" "${lang_pairs[@]}"
