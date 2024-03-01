import argparse
import pprint
import json
from ast import literal_eval

from dl_bench.mlp import MlpBenchmark
from dl_bench.cnn import CnnBenchmark
from dl_bench.llm import LlmBenchmark
from dl_bench.mlp_basic import MlpBasicBenchmark
from dl_bench.report.report import BenchmarkDb
from dl_bench.utils import Backend
from dl_bench.tools.compare_tensors import compare

benchmarks_table = {
    "mlp_oneiter": MlpBasicBenchmark,
    "mlp": MlpBenchmark,
    "cnn": CnnBenchmark,
    "llm": LlmBenchmark,
}


def fix_lengths(outputs, ref_outputs):
    """To speed up benchmarking we pass different number of batches for different backends.
    Need to match the lenghts."""
    min_lengths = min(len(outputs), len(ref_outputs))
    if len(outputs) != len(ref_outputs):
        print(
            f"Slicing passed batches to smallest size {len(outputs)}->{min_lengths}; {len(ref_outputs)}->{min_lengths}"
        )
        return outputs[:min_lengths], ref_outputs[:min_lengths]
    else:
        return outputs, ref_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    # Benchmark
    parser.add_argument(
        "-b",
        "--benchmark",
        choices=list(benchmarks_table),
        help="Benchmark to run.",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-bd",
        "--benchmark_desc",
        default=None,
        help="Benchmark descriptor for database.",
    )
    parser.add_argument(
        "-p",
        "--benchmark_params",
        default="",
        help="parameters for benchmark. Arguments like in dict(X). Example -p=\"name='size5',need_train=False\"",
    )

    # Backend
    parser.add_argument(
        "--backend_desc",
        default=None,
        help="Backend descriptor for database.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="Name of the host machine",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=["cpu", "xpu", "cuda", "openvino-cpu", "openvino-gpu"],
        help="Device to use for benchmark.",
    )
    parser.add_argument(
        "-c",
        "--compiler",
        default="torch",
        choices=[
            "",
            "torch",
            "dynamo",
            "torchscript",
            "torchscript_onednn",
            "ipex",
            "ipex_onednn_graph",
            "torch_mlir",
            "torch_mlir_xsmm",
        ],
        help="Compilation mode to use. No compilation by default.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=[
            "float32",
            "bfloat16",
            "int8",
        ],
        help="Dtype for computations.",
    )
    # Reporting
    parser.add_argument("-t", "--tag", default="", help="Tag to mark this result in DB")

    parser.add_argument(
        "-u",
        "--url",
        default="sqlite:///results.db",
        help="Database url in sqlalchemy format, like sqlite://./database.db'",
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Path to output report file."
    )
    parser.add_argument(
        "--skip_verification",
        required=False,
        action="store_true",
        help="Skip output verification.",
    )
    return parser.parse_args()


def parse_benchmark_params(params_txt):
    print(f"{params_txt}")
    key2val = {}
    for row in params_txt.split(","):
        if len(row.strip()) == 0:
            continue
        print(row)
        key, val = row.split("=")
        key2val[key] = literal_eval(val)

    return key2val


def main():
    args = parse_args()

    benchmark_name = args.benchmark
    benchmark_desc = args.benchmark_desc or f"{benchmark_name}_{args.benchmark_params}"
    benchmark_params = parse_benchmark_params(args.benchmark_params)

    host = args.host
    device = args.device
    compiler = args.compiler
    if compiler == "":
        compiler = "torch"
    dtype = args.dtype
    backend_desc = args.backend_desc or f"{host}_{device}_{compiler}"
    if dtype != "float32":
        backend_desc += "_" + str(dtype)
    benchmark_params["dtype"] = dtype
    benchmark_params["batch_size"] = args.batch_size

    backend = Backend(device=device, compiler=compiler, dtype=dtype)
    benchmark = benchmarks_table[benchmark_name](benchmark_params)
    if args.skip_verification:
        results, _ = benchmark.inference(backend)
    else:
        ref_device = "cpu" if device not in "cuda" else device
        reference_backend = Backend(device=ref_device, compiler="torch", dtype=dtype)
        _, ref_outputs = benchmark.inference(reference_backend)
        results, outputs = benchmark.inference(backend)
        outputs, ref_outputs = fix_lengths(outputs, ref_outputs)
        cmp_res = compare(outputs, ref_outputs)

    print(f"Benchmark {benchmark_name} completed")

    report = {
        "tag": args.tag,
        "benchmark": benchmark_name,
        "batch_size": args.batch_size,
        "benchmark_desc": benchmark_desc,
        "benchmark_params": benchmark_params,
        "backend_desc": backend_desc,
        "host": host,
        "device": device,
        "compiler": compiler,
        "dtype": dtype,
        **{
            c: results[c]
            for c in [
                "duration_s",
                "samples_per_s",
                "flops_per_sample",
                "n_items",
                "p00",
                "p50",
                "p90",
                "p100",
            ]
        },
    }

    db = BenchmarkDb(args.url)

    print("Report:")
    print("FPS: {:.1f}".format(results.get("samples_per_s", 0)))
    print(
        "TFLOPS: {:.3}".format(
            results.get("flops_per_sample", 0)
            * results.get("samples_per_s", 0)
            / (10**12)
        )
    )
    pprint.pprint(report)
    pprint.pprint(results)

    if args.output is not None:
        with open(args.output, "w", encoding="UTF-8") as out:
            json.dump(report, out)

    db.report(**report)


if __name__ == "__main__":
    main()
