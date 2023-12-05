import argparse
import pprint
import json
from ast import literal_eval

from dl_bench.mlp import MlpBenchmark
from dl_bench.mlp_basic import MlpBasicBenchmark
from dl_bench.report.report import BenchmarkDb
from dl_bench.utils import Backend

benchmarks_table = {
    "mlp_oneiter": MlpBasicBenchmark,
    "mlp": MlpBenchmark,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Benchmark
    parser.add_argument(
        "-b",
        "--benchmark",
        choices=list(benchmarks_table.keys()),
        help="Benchmark to run.",
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
        "-d",
        "--device",
        default="cpu",
        choices=["cpu", "xpu", "cuda", "openvino-cpu", "openvino-gpu"],
        help="Device to use for benchmark.",
    )
    parser.add_argument(
        "-c",
        "--compiler",
        default="",
        choices=[
            "",
            "dynamo",
            "torchscript",
            "torchscript_onednn",
            "ipex",
            "torch_mlir",
        ],
        help="Compilation mode to use. No compilation by default.",
    )
    parser.add_argument(
        "--host",
        default="",
        help="Name of the host machine",
    )

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
        "-v", "--verbose", required=False, action="store_true", help="Verbose mode."
    )
    return parser.parse_args()


def parse_benchmark_params(params_txt):
    print(f"{params_txt}")
    key2val = {}
    for row in params_txt.split(","):
        print(row)
        key, val = row.split("=")
        key2val[key] = literal_eval(val)

    return key2val


def main():
    args = parse_args()

    benchmark_name = args.benchmark
    benchmark_desc = args.benchmark_desc or f"{benchmark_name}_{args.benchmark_params}"
    benchmark_params = parse_benchmark_params(args.benchmark_params)

    device = args.device
    compiler = args.compiler
    if compiler == "":
        compiler = "torch"
    host = args.host
    backend_desc = args.backend_desc or f"{host}_{device}_{compiler}"


    backend = Backend(device=device, compiler=compiler)
    benchmark = benchmarks_table[benchmark_name]()
    results = benchmark.run(backend=backend, params=benchmark_params)

    print(f"Benchmark {benchmark_name} completed")


    report = {
        "benchmark": benchmark_name,
        "benchmark_desc": benchmark_desc,
        "benchmark_params": benchmark_params,
        "backend_desc": backend_desc,
        "host": host,
        "device": device,
        "compiler": compiler,
        **{c: results.get(c, 0) for c in ["warmup_s", "duration_s", "samples_per_s", "flops_per_sample"]},
    }

    db = BenchmarkDb(args.url)

    if args.verbose:
        print("Report:")
        print("TFLOPS: {:.3}".format(results.get("flops_per_sample", 0) * results.get('samples_per_s', 0) / (10**12)))
        pprint.pprint(report)

    if args.output is not None:
        with open(args.output, "w", encoding="UTF-8") as out:
            json.dump(report, out)

    db.report(**report)


if __name__ == "__main__":
    main()
