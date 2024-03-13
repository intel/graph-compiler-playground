#!/bin/bash

. "$(dirname "$0")/common.sh"

run_benchmark_suit mlp "float32 bfloat16 int8" "1024" "size2 size3 size4 size5 size5_sigm size5_tanh size5_gelu size5_linear size5_inplace size5_bn size5_bn_gelu size5_drop_gelu 100@512 25@1024 4@16384 2@16384"
run_benchmark_suit mlp "float32 bfloat16 int8" "1 16 256 2048 8196" "size5"
print_report
