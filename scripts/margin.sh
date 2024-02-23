#!/bin/bash

set -x

HOST="test"

export KMP_AFFINITY="respect,noreset,granularity=fine,balanced"
export OMP_NUM_THREADS=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
echo "Cores configured $OMP_NUM_THREADS"
export ONEDNN_VERBOSE=0

for i in 1 2 3 4 5 6 7
do
	CNNS=(resnet50)
	for COMPILER in ipex_onednn_graph
	do
		for DTYPE in bfloat16
		do
		for BS in 0001 0016 0032 0064 0128
		do
			for name in "${CNNS[@]}"
			do
				echo "Benchmark $name with BS=$BS and DTYPE=$DTYPE"
				numactl -N 1 benchmark-run -b cnn -p "name='${name}',batch_size='$BS'" --dtype "${DTYPE}" --benchmark_desc "${name}_bs$BS" --host "${HOST}" -c "${COMPILER}" --skip_verification
			done
		done
		done
	done


	LLMS=(gptj llama2-7b)
	for COMPILER in ipex
	do
		for BS in 0001 0004 0008
		do
			for DTYPE in bfloat16
			do
				for name in "${LLMS[@]}"
				do
					echo "Benchmark $name with DTYPE=$DTYPE"
					numactl -N 1 benchmark-run -b llm -p "name='${name}',batch_size=${BS}" --dtype "${DTYPE}" --benchmark_desc "${name}_bs$BS" --host "${HOST}" -c "${COMPILER}" --skip_verification
				done
			done
		done
	done
done
