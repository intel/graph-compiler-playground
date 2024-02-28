#!/bin/bash

set -x

HOST="test"

export KMP_AFFINITY="respect,noreset,granularity=fine,balanced"
export OMP_NUM_THREADS=4
echo "Cores configured $OMP_NUM_THREADS"
export ONEDNN_VERBOSE=0

for i in 1 2 3 4 5 6 7
do
	CNNS=(resnet50)
	for COMPILER in ipex_onednn_graph
	do
		for DTYPE in float32 bfloat16
		do
		for BS in 0001
		do
			for name in "${CNNS[@]}"
			do
				echo "Benchmark $name with BS=$BS and DTYPE=$DTYPE"
				export BENCH_COMMAND="benchmark-run -b cnn -p name='${name}',batch_size='$BS' --dtype ${DTYPE} --benchmark_desc ${name}_bs${BS}each4 --host ${HOST} -c ${COMPILER} --skip_verification"
				numactl -m 0 --physcpubind=4-7 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=8-11 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=12-15 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=16-19 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=20-23 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=24-27 $BENCH_COMMAND &
				numactl -m 0 --physcpubind=28-31 $BENCH_COMMAND &
				wait $(jobs -p)
			done
		done
		done
	done


	LLMS=(gptj llama2-7b)
	for COMPILER in ipex
	do
		for BS in 1
		do
			for DTYPE in bfloat16
			do
				for name in "${LLMS[@]}"
				do
					echo "Benchmark $name with DTYPE=$DTYPE"
					export BENCH_COMMAND="benchmark-run -b llm -p name='${name}',batch_size='$BS' --dtype ${DTYPE} --benchmark_desc ${name}_bs${BS}each4 --host ${HOST} -c ${COMPILER} --skip_verification"
					numactl -m 0 --physcpubind=4-7 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=8-11 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=12-15 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=16-19 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=20-23 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=24-27 $BENCH_COMMAND &
					numactl -m 0 --physcpubind=28-31 $BENCH_COMMAND &
					wait $(jobs -p)
					done
			done
		done
	done
done
