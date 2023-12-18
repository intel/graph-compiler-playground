#!/bin/bash

set -ex

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
${SCRIPT_DIR}/create-env.sh $1/conda-dev-env.yml
source ${CONDA}/bin/activate mlir
pip install -r $1/requirements.txt
pip install -r $1/torchvision-requirements.txt
