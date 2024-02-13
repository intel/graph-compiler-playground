CREATE OR REPLACE VIEW torchmlir_benchmark_view AS
SELECT
    id,
    REPLACE(REPLACE(CONCAT(host, '-', compiler, '-', dtype, '-', tag), 'torchscript', 'ts'), '-ci', '') AS backend,
    host,
    device,
    compiler,
    dtype,
    tag,
    benchmark,
    benchmark_desc,
    samples_per_s AS items_per_s,
    flops_per_sample,
    flops_per_sample * samples_per_s / 1e12 AS tflops,
    date
FROM torchmlir_benchmark;
