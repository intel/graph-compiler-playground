import runpy
from pathlib import Path
from setuptools import setup, find_packages, find_namespace_packages

root = Path(__file__).resolve().parent

with open(root / "README.md", encoding="utf-8") as f:
    long_description = f.read()

with open(root / "requirements.txt", "r") as f:
    reqs = f.readlines()


name = "dl_bench"
version = "v0.1"

setup(
    name=name,
    version=version,
    description="Benchmarks for torch compilers.",
    long_description=long_description,
    packages=[
        *find_packages(include=["dl_bench"]),
    ],
    install_requires=reqs,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            # "report-xlsx = timedf.scripts.report_xlsx:main",
            "benchmark-run = dl_bench.cli.launcher:main",
            # "benchmark-load = timedf.scripts.benchmark_load:main",
        ]
    },
)
