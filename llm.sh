#!/bin/bash

set -x

export ONEDNN_VERBOSE=0

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

for NAME in llama2-7b llama2-13b gptj
do
  for BS in 1 4
  do
    for DTYPE in int8 float32 bfloat16
    do
      echo "Benchmark $NAME"
      echo "Batch size $BS"
      BS_TXT=$(printf "%04d" $BS)
      benchmark-run -b llm -p "name='${NAME}'" -bs ${BS} --benchmark_desc "${NAME}_bs${BS_TXT}" --dtype "${DTYPE}" ${DL_BENCH_ARGS} || echo Failed
    done
  done
done
