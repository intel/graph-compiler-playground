import argparse
import time
import json
from collections import OrderedDict

from mlp import MPLBenchmark


def main():
    """Main"""
    benchmarks_table = {
        "MLP": MPLBenchmark,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        nargs="+",
        required=False,
        default=["all"],
        choices=(list(benchmarks_table.keys()) + ["all"]),
        help="Benchmark mode(s) to run. Default is to run all benchmarks.",
    )
    parser.add_argument(
        "-inf",
        "--inference",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run inference.",
    )
    parser.add_argument(
        "--train",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run train.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        required=False,
        default="CPU",
        choices=["CPU", "IPEX-CPU", "IPEX-XPU", "CUDA", "OPENVINO-CPU", "OPENVINO-GPU"],
        help="Execution engine to use.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        required=False,
        default=1,
        help="Number of iterations to perform.",
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Name of output report file."
    )
    parser.add_argument(
        "-v", "--verbose", required=False, action="store_true", help="Verbose mode."
    )
    parser.add_argument(
        "-j",
        "--jit",
        required=False,
        default="Vanilla",
        choices=["Vanilla", "Dynamo", "TorchScript", "TorchScriptOneDNN", "IPEX", "TorchMLIR"],
        help="JIT compilers.",
    )
    args = parser.parse_args()

    if "all" in args.mode:
        args.mode = benchmarks_table.keys()

    report = OrderedDict()
    for benchmark_mode in args.mode:
        benchmark = benchmarks_table[benchmark_mode](args.engine, args.jit)
        start_time = time.time()
        benchmark.execute(args.train)
        execution_time = round(time.time() - start_time, 2)
        print(f"Benchmark {benchmark_mode} completed in {execution_time} seconds")
        report[benchmark_mode] = execution_time

    if args.verbose:
        print("Summary:")
        for bn, et in report.items():
            print(f"Benchmark {bn} completed in {et} seconds.")

    if args.output is not None:
        with open(args.output, "w", encoding="UTF-8") as out:
            json.dump(report, out)


if __name__ == "__main__":
    main()
