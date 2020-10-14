#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

BPE_VOCAB="${LASER}/models/93langs.fvocab"


ProcessEuroparl() {
  bpe="${LASER}/data/europarl/bpe"
  data_bin="${LASER}/data/europarl/binarized"

  lang_pairs=( "en-es" "de-en" "de-es" "de-fr" "en-fr" "es-fr" )

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

ProcessEuroparl