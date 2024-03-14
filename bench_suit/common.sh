#!/bin/bash

set -x

if [ -z "${COMPILER}" ]; then
  echo "Please, provide COMPILER environment variable"
  exit 1
fi

if [ -z "${DEVICE}" ]; then
  echo "Please, provide DEVICE environment variable"
  exit 1
fi

if [ -z "${OTHER_ARGS}" ]; then
  echo "Please, provide OTHER_ARGS environment variable"
  exit 1
fi

# List of all benchmark runs that failed to report
failed_commands=()

run_benchmark_suit() {
    BENCHMARK="$1"
    DTYPEs="$2"
    BSs="$3"
    NAMEs="$4"

    echo "Running suit for benchmark=\"$BENCHMARK\" DTYPEs=\"$DTYPEs\" BSs=\"$BSs\" NAMEs=\"$NAMEs\""

    for DTYPE in $DTYPEs
    do
        for BS in $BSs
        do
            for NAME in $NAMEs
            do
                echo "Benchmark $NAME with BS=$BS and DTYPE=$DTYPE"
                BS_TXT=$(printf "%04d" $BS)

                CMD="benchmark-run -c ${COMPILER} -d ${DEVICE} -b ${BENCHMARK} -p name='${NAME}' --dtype ${DTYPE} -bs $BS --benchmark_desc ${NAME}_bs${BS_TXT} ${OTHER_ARGS}"
                $CMD || failed_commands+=("$CMD")
            done
        done
    done
}


print_report() {
    if [ ${#failed_commands[@]} -gt 0 ]; then
        echo "Some benchmarks failed, here's the list:"
        for CMD in "${failed_commands[@]}"; do
            echo "$CMD"
        done
            exit 1
        fi
}
