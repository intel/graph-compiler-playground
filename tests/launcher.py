import argparse
import pprint
import json
from ast import literal_eval

from mlp import MlpBenchmark
from mlp_basic import MlpBasicBenchmark
from utils import Backend

benchmarks_table = {
    "mlp_oneiter": MlpBasicBenchmark,
    "mlp": MlpBenchmark,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", choices=list(benchmarks_table.keys()), help="Benchmark to run.")
    parser.add_argument("-p", "--benchmark_params", default="", help="parameters for benchmark. Arguments like in dict(X). Example -p=\"name='size5',need_train=False\"")
    parser.add_argument(
        "-d",
        "--device",
        required=False,
        default="cpu",
        choices=["cpu", "xpu", "cuda", "openvino-cpu", "openvino-gpu"],
        help="Device to use for benchmark.",
    )
    parser.add_argument("-o", "--output", required=False, help="Path to output report file.")
    parser.add_argument("-v", "--verbose", required=False, action="store_true", help="Verbose mode.")
    parser.add_argument(
        "-c",
        "--compiler",
        required=False,
        default="",
        choices=["", "dynamo", "torchscript", "torchscript_onednn", "ipex", "torch_mlir"],
        help="Compilation mode to use. No compilation by default.",
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
    device = args.device
    compiler = args.compiler
    benchmark_name = args.benchmark
    benchmark_params = parse_benchmark_params(args.benchmark_params)

    backend = Backend(device=device, compile_mode=compiler)
    benchmark = benchmarks_table[benchmark_name]()
    benchmark_report = benchmark.run(backend=backend, params=benchmark_params)

    print(f"Benchmark {benchmark_name} completed")
    report = {
        "device": args.device,
        "compiler": args.compiler,
        "benchmark": benchmark_name,
        "benchmark_params": benchmark_params,
        "results": benchmark_report,
    }

    if args.verbose:
        print("Report:")
        pprint.pprint(report)

    if args.output is not None:
        with open(args.output, "w", encoding="UTF-8") as out:
            json.dump(report, out)


if __name__ == "__main__":
    main()
