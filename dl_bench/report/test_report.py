import pytest
from sqlalchemy import create_engine, select

from dl_bench.report import BenchmarkDb
from dl_bench.report.report import metadata_obj


engine_string = "sqlite://"


def test_schema():
    engine = create_engine(engine_string)
    metadata_obj.create_all(engine)


def test_dbreport():
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    db = BenchmarkDb(engine_string)

    db.report(
        benchmark="benchmark",
        benchmark_desc="benchmark_x",
        benchmark_params={},
        backend_desc="backend_pam",
        backend_params={},
        warmup_s=11.1,
        duration_s=1.1,
        samples_per_s=2233.1,
    )
