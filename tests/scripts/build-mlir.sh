#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi


cd $1

${CONDA}/bin/conda env create -n mlir -f conda-dev-env.yml.yml
source ${CONDA}/bin/activate mlir

pip install -r requirements.txt

source utils/build-with-imex.sh
