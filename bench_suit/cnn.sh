#!/bin/bash

. "$(dirname "$0")/common.sh"


run_benchmark_suit cnn "float32 bfloat16 int8" "1 32 128" "vgg16 resnet18 resnet50 resnext50 resnext101 densenet121"
print_report
