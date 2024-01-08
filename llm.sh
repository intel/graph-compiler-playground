#!/bin/bash

set -x

export ONEDNN_VERBOSE=0

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

benchmark-run -b llm -p "" --benchmark_desc "gptj" --dtype float32 ${DL_BENCH_ARGS} || echo Failed
benchmark-run -b llm -p "" --benchmark_desc "gptj_bfloat16" --dtype bfloat16 ${DL_BENCH_ARGS} || echo Failed
