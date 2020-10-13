#!/bin/bash

if [ -z ${LASER} ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
data_root="${bdir}/data"
tools_ext="${bdir}/tools-external"

moses_dir="${tools_ext}/moses-tokenizer"
TOKENIZER="${moses_dir}/tokenizer/tokenizer.perl"
LC="${moses_dir}/tokenizer/lowercase.perl"
NORM_PUNC="${moses_dir}/tokenizer/normalize-punctuation.perl"
REM_NON_PRINT_CHAR="${moses_dir}/tokenizer/remove-non-printing-char.perl"

###################################################################
#
# Generic helper functions
#
###################################################################

MKDIR () {
  dname=$1
  if [ ! -d ${dname} ] ; then
    echo " - creating directory ${dname}"
    mkdir -p ${dname}
  fi
}

###################################################################
#
# Download and pre-process data
#
###################################################################

PrepareEuroparl() {
  echo "Preparing Europarl"
  europarl_root="${data_root}/europarl"
  MKDIR ${europarl_root}
  cd ${europarl_root}

  lang_pairs=( "en-es" "de-en" "de-es" "de-fr" "en-fr" "es-fr" )

  urlpref="http://opus.nlpl.eu/download.php?f=Europarl/v8/moses"
  for lang_pair in ${lang_pairs[@]} ; do
    f="${lang_pair}.txt.zip"
    if [ ! -f "${f}" ] ; then
      echo " - Downloading ${f}"
      wget -q "${urlpref}/${f}" -O ${f}
      echo " - unzip ${f}"
      unzip -q ${f}
      /bin/rm {README,LICENSE}
    fi
  done

  /bin/rm -f train.all
  for lang_pair in ${lang_pairs[@]} ; do
    src=`echo ${lang_pair} | cut -d'-' -f1`
    tgt=`echo ${lang_pair} | cut -d'-' -f2`
    lang="${src}-${tgt}"
    for l in ${src} ${tgt}; do
      cat "Europarl.${lang}.${l}" | \
        perl ${REM_NON_PRINT_CHAR} | \
        perl ${NORM_PUNC} ${l} | \
        perl ${TOKENIZER} -threads 20 -l ${l} -q -no-escape | \
        perl ${LC} >> train.all
    done
  done
}

PrepareEuroparl

echo "Done!!!"