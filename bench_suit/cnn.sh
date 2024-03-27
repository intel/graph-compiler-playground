#!/bin/bash

. "$(dirname "$0")/common.sh"

DTYPEs="float32"
if [ "$COMPILER" = 'ipex_onednn_graph' ]; then
    DTYPEs="$DTYPEs int8"
fi

run_benchmark_suit cnn "$DTYPEs" "1 32 128" "vgg16 resnet18 resnet50 resnext50 resnext101 densenet121"
print_report
