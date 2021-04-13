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
  elif [ "${save_file:(-7)}" == ".tar.gz" ] || [ "${save_file:(-4)}" == ".tgz" ] ; then
    tar -zxf "${save_file}" -C "${save_dir}"
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
  save_dir="${data_root}/WikiMatrix/raw"

  lang_pairs=( "en-zh" )
  for lang_pair in "${lang_pairs[@]}" ; do
    f="WikiMatrix.v1.${lang_pair}.langid.tsv.gz"
    DownloadAndUnpack "${urlpref}/${f}" "${save_dir}/${f}"

    src=$(echo "${lang_pair}" | cut -d'-' -f1)
    tgt=$(echo "${lang_pair}" | cut -d'-' -f2)
    src_file="${save_dir}/WikiMatrix.${lang_pair}.${src}"
    tgt_file="${save_dir}/WikiMatrix.${lang_pair}.${tgt}"
    awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" -v src=${src} -v tgt=${tgt}\
      '{if($4==src && $5==tgt){print $2 >> src_file; print $3 >> tgt_file}}' "${save_dir}/${f%.gz}"
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


# Monolingual dataset

DownloadNewsCrawl() {
  echo "Downloading News Crawl"

  urlpref="http://data.statmt.org/news-crawl"
  save_dir="${data_root}/news-crawl/raw"
  langs=( "zh" )
  year="2020"

  for lang in "${langs[@]}" ; do
    f="news.${year}.${lang}.shuffled.deduped.gz"
    DownloadAndUnpack "${urlpref}/${lang}/${f}" "${save_dir}/${f}"

    head -n 2000000 "${save_dir}/${f%.gz}" > "${save_dir}/news-crawl.${lang}1-${lang}2.${lang}1"
    head -n 2000000 "${save_dir}/${f%.gz}" > "${save_dir}/news-crawl.${lang}1-${lang}2.${lang}2"
  done
}


DownloadNewsDiscuss() {
  echo "Downloading News Discussions"

  urlpref="http://data.statmt.org/news-discussions/en/"
  save_dir="${data_root}/news-discuss/raw"

  langs=( "en" )
  year="2019"
  for lang in "${langs[@]}" ; do
    f="news-discuss.${year}.${lang}.filtered.gz"
    DownloadAndUnpack "${urlpref}/${lang}/${f}" "${save_dir}/${f}"

    head -n 2000000 "${save_dir}/${f%.gz}" > "${save_dir}/news-discuss.${lang}1-${lang}2.${lang}1"
    head -n 2000000 "${save_dir}/${f%.gz}" > "${save_dir}/news-discuss.${lang}1-${lang}2.${lang}2"
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
    awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" -v lang=${lang} \
      '{if($1==lang && $2=="entailment"){print $7 >> src_file; print $8 >> tgt_file}}' "${xnli_root}/xnli.dev.tsv"
  done
}


DownloadSNLI() {
  echo "Downloading SNLI"

  url="https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
  snli_root="${data_root}/snli"

  DownloadAndUnpack "${url}" "${data_root}/snli_1.0.zip"
  /bin/rm -r "${data_root}/__MACOSX"
  /bin/rm "${data_root}/snli_1.0/.DS_Store"
  /bin/mv "${data_root}/snli_1.0" "${snli_root}"

  save_dir="${snli_root}/raw"
  mkdir -p "${save_dir}"
  lang="en"
  src="${lang}1"
  tgt="${lang}2"
  src_file="${save_dir}/snli.${src}-${tgt}.${src}"
  tgt_file="${save_dir}/snli.${src}-${tgt}.${tgt}"
  awk -F "\t" -v src_file="${src_file}" -v tgt_file="${tgt_file}" \
    '{if($1=="entailment"){print $6 >> src_file; print $7 >> tgt_file}}' "${snli_root}/snli_1.0_train.txt"
}


DownloadEuroparl
DownloadNewsCommentary
DownloadWikiMatrix
DownloadNewsCrawl
DownloadNewsDiscuss
DownloadXNLI
DownloadSNLI

echo "Done!!!"
