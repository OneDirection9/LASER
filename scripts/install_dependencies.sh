#!/usr/bin
set -e

pip install transliterate jieba
pip install torch==1.0 fairseq==0.8.0

bash ./scripts/install_models.sh
bash ./scripts/install_external_tools.sh --install-mecab

# ./scripts/download_data.sh
# python preprocess.py
# ./scripts/binarize_data.sh
