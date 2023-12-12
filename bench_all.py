import os

enable_disable_gc = [0, 1]
num_threads = [56]
dtypes = ["bfloat16"] # dtypes = ["float32", "bfloat16", "int8"]
# cases = ["name='100@512',batch_size=1024", "name='25@1024',batch_size=1024", "name='2@16384',batch_size=1024", "name='4@16384',batch_size=1024", "name='size2',batch_size=1024", "name='size3',batch_size=1024", "name='size4',batch_size=1024", "name='size5_bn',batch_size=1024", "name='size5_bn_gelu',batch_size=1024", "name='size5',batch_size=1", "name='size5',batch_size=1024", "name='size5',batch_size=16", "name='size5',batch_size=2048", "name='size5',batch_size=256", "name='size5',batch_size=8196", "name='size5_drop_gelu',batch_size=1024", "name='size5_gelu',batch_size=1024", "name='size5_inplace',batch_size=1024", "name='size5_linear',batch_size=1024", "name='size5_sigm',batch_size=1024", "name='size5_tanh',batch_size=1024"]
cases = ["name='2@16384',batch_size=1024", "name='4@16384',batch_size=1024", "name='size3',batch_size=1024", "name='size4',batch_size=1024", "name='size5',batch_size=256"]

for num_th in num_threads:
    for dtype in dtypes:
        for case in cases:
            for gc in enable_disable_gc:
                cmd = "_DNNL_DISABLE_COMPILER_BACKEND=" + str(gc) + " ONEDNN_VERBOSE=0 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=" + str(num_th) + " benchmark-run -b mlp -p \"" + case + "\"  --dtype=" + dtype + " -v --host test -c ipex_dnnl"
                print(cmd)
                os.system(cmd)
                print()
