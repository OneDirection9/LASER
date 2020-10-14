#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

BPE_CODES="${LASER}/models/93langs.fcodes"
FASTBPE="${LASER}/tools-external/fastBPE/fast"
MOSES_DIR="${LASER}/tools-external/moses-tokenizer/"
MOSES_TOKENIZER="${MOSES_DIR}/tokenizer/tokenizer.perl"
MOSES_LC="${MOSES_DIR}/lowercase.perl"
NORM_PUNC="${MOSES_DIR}/normalize-punctuation.perl"
DESCAPE="${MOSES_DIR}/deescape-special-chars.perl"
REM_NON_PRINT_CHAR="${MOSES_DIR}/remove-non-printing-char.perl"

data_root="${LASER}/data"

###################################################################
#
# Download data
#
###################################################################

PrepareEuroparl() {
  echo "Preparing Europarl"
  raw_dir="${data_root}/europarl/raw"
  pro_dir="${data_root}/europarl/processed"
  bpe_dir="${data_root}/europarl/bpe"
  mkdir -p "${raw_dir}" "${pro_dir}" "${bpe_dir}"

  lang_pairs=( "en-es" "de-en" "de-es" "de-fr" "en-fr" "es-fr" )

  urlpref="http://opus.nlpl.eu/download.php?f=Europarl/v8/moses"
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    if [ ! -f "${raw_dir}/${f}" ] ; then
      echo " - Downloading ${f}"
      wget -q "${urlpref}/${f}" -O "${raw_dir}/${f}"
      echo " - unzip ${f}"
      unzip -q "${raw_dir}/${f}" -d "${raw_dir}"
      /bin/rm "${raw_dir}/{README,LICENSE}"
    fi
  done

  for lang_pair in "${lang_pairs[@]}" ; do
    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    lang="${src}-${tgt}"
    for l in ${src} ${tgt} ; do
      echo " - preprocessing ${lang}.${l}"
      /bin/rm -rf "${pro_dir}/train.${lang}.${l}"
      cat "${raw_dir}/Europarl.${lang}.${l}" | \
        perl "${REM_NON_PRINT_CHAR}" | \
        perl "${NORM_PUNC}" "${l}" | \
        perl "${MOSES_TOKENIZER}" -threds 20 -l "${l}" -q -no-escape | \
        perl "${MOSES_LC}" > "${pro_dir}/train.${lang}.${l}"

      FASTBPE applybpe "${bpe_dir}/train.${lang}.${l}" "${pro_dir}/train.${lang}.${l}" BPE_CODES
    done
  done
}

PrepareEuroparl

echo "Done!!!"