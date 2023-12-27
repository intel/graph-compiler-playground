#!/bin/bash

set -ex

if (( $# != 1 )); then
    >&2 echo "Need path to environment spec file as an argument."
fi

${CONDA}/bin/conda env update --solver libmamba -f $1
