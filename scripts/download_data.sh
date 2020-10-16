#!/bin/bash

if [ -z "${LASER}" ]; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

data_root="${LASER}/data"

###################################################################
#
# Download data
#
###################################################################

DownloadEuroparl() {
  echo "Downloading Europarl"
  raw_dir="${data_root}/europarl/raw"
  mkdir -p "${raw_dir}"

  lang_pairs=( "en-it" )

  urlpref="http://opus.nlpl.eu/download.php?f=Europarl/v8/moses"
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    if [ ! -f "${raw_dir}/${f}" ] ; then
      echo " - Downloading ${f}"
      wget -q "${urlpref}/${f}" -O "${raw_dir}/${f}"
      echo " - unzip ${f}"
      unzip -q "${raw_dir}/${f}" -d "${raw_dir}"
      /bin/rm "${raw_dir}"/{README,LICENSE}
    fi
  done
}

DownloadUNPC() {
  echo "Downloading UNPC (United Nations Parallel Corpus)"
  raw_dir="${data_root}/unpc/raw"
  mkdir -p "${raw_dir}"

  lang_pairs=( "en-zh" )

  urlpref="http://opus.nlpl.eu/download.php?f=UNPC/v1.0/moses"
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    if [ ! -f "${raw_dir}/${f}" ] ; then
      echo " - Downloading ${f}"
      wget -q "${urlpref}/${f}" -O "${raw_dir}/${f}"
      echo " - unzip ${f}"
      unzip -q "${raw_dir}/${f}" -d "${raw_dir}"
      /bin/rm "${raw_dir}"/{README,LICENSE}

      src=$(echo "${lang_pair}" | cut -d'-' -f1)
      tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
      for l in $src $tgt ; do
        head -n 2000000 "${raw_dir}/UNPC.${lang_pair}.${l}" > "${raw_dir}/UNPC.${lang_pair}.${l}.2000000"
      done
    fi
  done
}

DownloadEuroparl
DownloadUNPC

echo "Done!!!"