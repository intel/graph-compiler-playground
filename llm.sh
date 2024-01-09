#!/bin/bash

set -x

export ONEDNN_VERBOSE=0

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

for DTYPE in float32 bfloat16
do
  benchmark-run -b llm -p "" --benchmark_desc "gptj" --dtype "${DTYPE}" ${DL_BENCH_ARGS} || echo Failed
done
