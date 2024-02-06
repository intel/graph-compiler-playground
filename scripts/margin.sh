#!/bin/sh

set -x

HOST="test"

export KMP_AFFINITY="respect,noreset,granularity=fine,balanced"
export OMP_NUM_THREADS=32
export ONEDNN_VERBOSE=0

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

CNNS=(resnet50)
for COMPILER in dynamo ipex_onednn_graph
for DTYPE in float32 bfloat16
do
  for BS in 0001 0032 0128
  do
      for name in "${CNNS[@]}"
      do
          echo "Benchmark $name with BS=$BS and DTYPE=$DTYPE"
          numactl -N 1 benchmark-run -b cnn -p "name='${name}',batch_size='$BS'" --dtype "${DTYPE}" --benchmark_desc "${name}_bs$BS" --host "${HOST}" -c "${COMPILER}"
      done
  done
done
