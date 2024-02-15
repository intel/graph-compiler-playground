#!/bin/bash

set -x

HOST="test"

export KMP_AFFINITY="respect,noreset,granularity=fine,balanced"
export OMP_NUM_THREADS=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
echo "Cores configured $OMP_NUM_THREADS"
export ONEDNN_VERBOSE=0

CNNS=(resnet50)
for COMPILER in dynamo ipex_onednn_graph
do
	for DTYPE in float32 bfloat16
	do
	  for BS in 0001 0032 0128
	  do
	      for name in "${CNNS[@]}"
	      do
			echo "Benchmark $name with BS=$BS and DTYPE=$DTYPE"
			numactl -N 1 benchmark-run -b cnn -p "name='${name}',batch_size='$BS'" --dtype "${DTYPE}" --benchmark_desc "${name}_bs$BS" --host "${HOST}" -c "${COMPILER}" --skip_verification | true
	      done
	  done
	done
done


LLMS=(gptj)
for COMPILER in dynamo ipex
do
	for DTYPE in float32 bfloat16
	do
		for name in "${LLMS[@]}"
		do
			echo "Benchmark $name with DTYPE=$DTYPE"
			numactl -N 1 benchmark-run -b llm -p "name='${name}'" --dtype "${DTYPE}" --benchmark_desc "${name}_bs$BS" --host "${HOST}" -c "${COMPILER}" --skip_verification
		done
	done
done

