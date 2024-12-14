#!/bin/bash

# We expect to have this repo present and this script run as
# ./scripts/margin.sh

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
curl -o Miniconda3-latest-Linux-x86_64.sh -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -u -b -p ./miniconda && \
    rm -f Miniconda3-latest-Linux-x86_64.sh
source ./miniconda/bin/activate

# set up env
conda create -y -n margin python=3.11
conda activate margin
# Install ipex & pytorch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install transformers==4.35.2
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

# Install benchmarks
pip install -e .
