CREATE OR REPLACE VIEW torchmlir_benchmark_view AS
SELECT
    id,
    REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
        REPLACE(
            CONCAT(host, '-', compiler, '-', dtype, '-', tag),
        'torchscript', 'ts'),
        '-ci', ''),
        'ts_onednn', 'onednn'),
        'ipex_onednn_graph', 'ipex_gc'),
        'bfloat16', 'b16'),
        'float32', 'f32'
    ) AS backend,
    host,
    device,
    compiler,
    REPLACE(REPLACE(dtype, 'bfloat16', 'b16'), 'float32', 'f32') AS dtype,
    tag,
    benchmark,
    benchmark_desc,
    samples_per_s AS items_per_s,
    flops_per_sample,
    flops_per_sample * samples_per_s / 1e12 AS tflops,
    date
FROM torchmlir_benchmark;
