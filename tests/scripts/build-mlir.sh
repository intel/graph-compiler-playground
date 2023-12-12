#!/bin/bash

set -ex

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi


cd $1

${CONDA}/bin/conda env create -n mlir -f conda-dev-env.yml
${CONDA}/bin/conda install -n mlir -y pip
source ${CONDA}/bin/activate mlir

pip install -r requirements.txt
# This might require an update
pip install --pre torch==2.2.0.dev20231105 torchvision==0.17.0.dev20231105+cpu --index-url https://download.pytorch.org/whl/nightly/cpu || echo "Failed to install torchvision"

bash ./utils/build-with-imex.sh
