#!/bin/bash

set -x

export ONEDNN_VERBOSE=0

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

echo "Bfloat16 on size5"
benchmark-run -b llm -p "" --benchmark_desc "gptj" --dtype bfloat16 ${DL_BENCH_ARGS} || echo Failed
