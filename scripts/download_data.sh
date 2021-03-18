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

DownloadAndUnpack() {
  url="$1"
  save_file="$2"
  save_dir="$(dirname "${save_file}")"
  mkdir -p "${save_dir}"

  echo " - Downloading ${url}"
  wget -q "${url}" -O "${save_file}"

  echo " - Unpacking ${save_file}"
  if [ "${save_file:(-4)}" == ".zip" ] ; then
    unzip -q "${save_file}" -d "${save_dir}"
  elif [ "${save_file:(-3)}" == ".gz" ] ; then
    gzip -d -c "${save_file}" > "${save_file%.gz}"
  fi

  /bin/rm "${save_file}"
}


DownloadEuroparl() {
  echo "Downloading Europarl"

  urlpref="http://opus.nlpl.eu/download.php?f=Europarl/v8/moses"
  save_dir="${data_root}/Europarl/raw"

  lang_pairs=( "en-it" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    DownloadAndUnpack "${urlpref}/${f}" "${save_dir}/${f}"
    /bin/rm "${save_dir}"/{LICENSE,README}
  done
}


DownloadNewsCommentary() {
  echo "Downloading News Commentary"

  urlpref="http://data.statmt.org/news-commentary/v15/training"
  save_dir="${data_root}/news-commentary/raw"

  lang_pairs=( "en-zh" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="news-commentary-v15.${lang_pair}.tsv.gz"
    DownloadAndUnpack "${urlpref}/${f}" "${save_dir}/${f}"

    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    src_file="${save_dir}/news-commentary.${lang_pair}.${src}"
    tgt_file="${save_dir}/news-commentary.${lang_pair}.${tgt}"
    awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" \
      '{if($1!=""){print $1 >> src_file; print $2 >> tgt_file}}' "${save_dir}/${f%.gz}"
  done
}


DownloadWikiMatrix() {
  echo "Downloading Wiki Matrix"

  urlpref="http://data.statmt.org/wmt20/translation-task/WikiMatrix"
  save_dir="${data_root}/WikiMatrix/raw_dir"

  lang_pairs=( "en-zh" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="WikiMatrix.v1.${lang_pair}.langid.tsv.gz"
    DownloadAndUnpack "${urlpref}/${f}" "${save_dir}/${f}"

    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    src_file="${save_dir}/WikiMatrix.${lang_pair}.${src}"
    tgt_file="${save_dir}/WikiMatrix.${lang_pair}.${tgt}"
    awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" \
      '{if($4=="en" && $5=="zh"){print $2 >> src_file; print $3 >> tgt_file}}' "${save_dir}/${f%.gz}"
  done
}


DownloadUNPC() {
  echo "Downloading UNPC (United Nations Parallel Corpus)"

  urlpref="http://opus.nlpl.eu/download.php?f=UNPC/v1.0/moses"
  save_dir="${data_root}/UNPC/raw"

  lang_pairs=( "en-zh" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="${lang_pair}.txt.zip"
    DownloadAndUnpack "${urlpref}/${f}" "${save_dir}/${f}"
    /bin/rm "${save_dir}"/{LICENSE,README}

    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    for l in $src $tgt ; do
      head -n 2000000 "${save_dir}/UNPC.${lang_pair}.${l}" > "${save_dir}/UNPC.${lang_pair}.${l}.2000000"
    done
  done
}


DownloadXNLI() {
  echo "Downloading XNLI"

  url="https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"
  xnli_root="${data_root}/XNLI"

  DownloadAndUnpack "${url}" "${data_root}/XNLI-1.0.zip"
  /bin/rm -r "${data_root}/__MACOSX"
  /bin/rm "${data_root}/XNLI-1.0/.DS_Store"
  /bin/mv "${data_root}/XNLI-1.0" "${xnli_root}"

  save_dir="${xnli_root}/raw"
  mkdir -p "${save_dir}"
  langs=( "en" )
  for lang in "${langs[@]}" ; do
    src="${lang}1"
    tgt="${lang}2"
    src_file="${save_dir}/XNLI.${src}-${tgt}.${src}"
    tgt_file="${save_dir}/XNLI.${src}-${tgt}.${tgt}"
    awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" \
      '{if($1=="en" && $2=="entailment"){print $7 >> src_file; print $8 >> tgt_file}}' "${xnli_root}/xnli.dev.tsv"
  done
}


DownloadEuroparl
DownloadNewsCommentary
DownloadWikiMatrix
DownloadXNLI

echo "Done!!!"
