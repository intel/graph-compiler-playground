#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "Need path to environment spec file as an argument."
fi

conda env create --solver libmamba -f $1
