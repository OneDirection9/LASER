#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

BPE_VOCAB="${LASER}/models/93langs.fvocab"

function binarize() {
  bpe="$1"
  data_bin="$2"
  lang_pairs="$3"

  for lang_pair in "${lang_pairs[@]}"; do
    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    /bin/rm "${data_bin}/dict.${src}.txt" "${data_bin}/dict.${tgt}.txt"
    fairseq-preprocess --source-lang "${src}" --target-lang "${tgt}" \
        --trainpref "${bpe}/train.${src}-${tgt}" \
        --joined-dictionary --tgtdict "${BPE_VOCAB}" \
        --destdir "${data_bin}" \
        --workers 20
  done
}

lang_pairs=( "en-it" )
binarize "${LASER}/data/europarl/bpe93" "${LASER}/data/europarl/binarized" "${lang_pairs}"

lang_pairs=( "en-zh" )
binarize "${LASER}/data/unpc/bpe93" "${LASER}/data/unpc/binarized" "${lang_pairs}"

lang_pairs=( "en1-en2" )
binarize "${LASER}/data/XNLI/bpe93" "${LASER}/data/XNLI/binarized" "${lang_pairs}"
