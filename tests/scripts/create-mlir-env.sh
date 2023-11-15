#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
./create-env.sh ${SCRIPT_DIR}/../conda-envs/mlir.yaml
source ${CONDA_PREFIX}/bin/activate mlir-test
pip install -r $1/requirements.txt
