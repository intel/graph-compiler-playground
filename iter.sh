export ONEDNN_VERBOSE=0

# TO be removed
benchmark-run -b cnn -p "name='vgg16',batch_size='16'" --benchmark_desc "vgg16_bs16" ${DL_BENCH_ARGS} || echo Failed
