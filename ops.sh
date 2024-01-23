#!/bin/bash

set -x

export ONEDNN_VERBOSE=all
export ONEDNN_VERBOSE_TIMESTAMP=1

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

CNNS=(conv210)
for BS in 0001 0032 0128
do
    for name in "${CNNS[@]}"
    do
        echo "Benchmark $name"
        benchmark-run -b ops -p "name='${name}',batch_size='$BS'" --benchmark_desc "${name}_bs$BS" ${DL_BENCH_ARGS} || echo Failed
    done
done
