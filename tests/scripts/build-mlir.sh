#!/bin/bash

set -ex

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi

if ${CONDA}/bin/conda env list | grep mlir > /dev/null; then
    echo "mlir conda environment already exists from cache, not creating a new one."
else
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    ${SCRIPT_DIR}/create-mlir-env.sh $1
fi

source ${CONDA}/bin/activate mlir

cd $1
bash ./utils/build-with-imex.sh
