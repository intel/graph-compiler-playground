#!/bin/bash

. "$(dirname "$0")/common.sh"

run_benchmark_suit llm "bfloat16" "1 4 8" "llama2-7b llama2-13b gptj"
print_report
