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

DownloadAndExtract() {
  url="$1"
  save_file="$2"
  save_dir="$(dirname "${save_file}")"

  echo " - Downloading ${url}"
  wget -q "${url}" -O "${save_file}"
  echo " - unzip ${save_file}"
  unzip -q "${save_file}" -d "${save_dir}"
  /bin/rm "${save_file}"
}

DownloadEuroparl() {
  echo "Downloading Europarl"

  urlpref="http://opus.nlpl.eu/download.php?f=Europarl/v8/moses"

  raw_dir="${data_root}/Europarl/raw"
  mkdir -p "${raw_dir}"

  lang_pairs=( "en-it" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    DownloadAndExtract "${urlpref}/${f}" "${raw_dir}/${f}"
    /bin/rm "${raw_dir}"/{LICENSE,README}
  done
}

DownloadUNPC() {
  echo "Downloading UNPC (United Nations Parallel Corpus)"

  urlpref="http://opus.nlpl.eu/download.php?f=UNPC/v1.0/moses"

  raw_dir="${data_root}/UNPC/raw"
  mkdir -p "${raw_dir}"

  lang_pairs=( "en-zh" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    DownloadAndExtract "${urlpref}/${f}" "${raw_dir}/${f}"
    /bin/rm "${raw_dir}"/{LICENSE,README}

    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    for l in $src $tgt ; do
      head -n 2000000 "${raw_dir}/UNPC.${lang_pair}.${l}" > "${raw_dir}/UNPC.${lang_pair}.${l}.2000000"
    done
  done
}


DownloadXNLI() {
  echo "Downloading XNLI"

  url="https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"

  mkdir -p "${data_root}"
  DownloadAndExtract "${url}" "${data_root}/XNLI-1.0.zip"
  /bin/rm -r "${data_root}/__MACOSX"
  /bin/rm "${data_root}/XNLI-1.0/.DS_Store"
  /bin/mv "${data_root}/XNLI-1.0" "${data_root}/XNLI"
}

DownloadEuroparl
DownloadUNPC
DownloadXNLI

echo "Done!!!"
