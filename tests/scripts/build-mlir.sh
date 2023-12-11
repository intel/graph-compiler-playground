#!/bin/bash

set -ex

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi


cd $1
git submodule init
git submodule update --recursive --init
git submodule foreach --recursive 'git fetch --tags'
ls -la
pwd
git status
pushd externals/llvm-project
ls -la
pwd
git status
popd

${CONDA}/bin/conda env create -n mlir -f conda-dev-env.yml
${CONDA}/bin/conda install -n mlir -y pip
source ${CONDA}/bin/activate mlir

pip install -r requirements.txt

bash ./utils/build-with-imex.sh
