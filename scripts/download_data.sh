#!/bin/bash

if [ -z ${LASER} ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

bdir="${LASER}"
data_root="${bdir}/data"


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
# Download data
#
###################################################################

DownloadEuroparl() {
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
}

DownloadEuroparl

echo "Done!!!"