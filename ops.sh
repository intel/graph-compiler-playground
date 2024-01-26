#!/bin/bash

set -x

export ONEDNN_VERBOSE=all
export ONEDNN_VERBOSE_TIMESTAMP=1

if [[ -z "${DL_BENCH_ARGS}" ]]; then
  echo "Please, provide DL_BENCH_ARGS environment variable"
  exit 1
fi

CNNS=(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9, conv_10, conv_11, conv_12, conv_13, conv_14, conv_15, conv_16, conv_17, conv_18, conv_19, conv_20, conv_21, conv_22, conv_23, conv_24, conv_25, conv_26, conv_27, conv_28, conv_29, conv_30, conv_31, conv_32, conv_33, conv_34, conv_35, conv_36, conv_37, conv_38, conv_39, conv_40, conv_41, conv_42, conv_43, conv_44, conv_45, conv_46, conv_47, conv_48, conv_49, conv_50, conv_51, conv_52, conv_53)
for BS in 0001 0032 0128
do
    for name in "${CNNS[@]}"
    do
        echo "Benchmark $name"
        benchmark-run -b ops -p "name='${name}',batch_size='$BS'" --benchmark_desc "${name}_bs$BS" ${DL_BENCH_ARGS} || echo Failed
    done
done
