#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "Need path to torch-mlir repository as an argument."
fi

source ${CONDA}/bin/activate mlir

env
${CONDA}/bin/conda list

cd $1
pip install -r requirements.txt

cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm

cmake --build build
