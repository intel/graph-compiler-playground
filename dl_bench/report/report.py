from sqlalchemy import create_engine, insert, Engine
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    DateTime,
    String,
    Float,
    Integer,
    JSON,
    func,
)

STRING_LENGTH = 200
LARGE_STRING_LENGTH = 500


metadata_obj = MetaData()


def make_string(name, nullable=False):
    return Column(name, String(STRING_LENGTH), nullable=nullable)


results_table = Table(
    "torchmlir_benchmark",
    metadata_obj,
    # Basic data
    Column("id", Integer, primary_key=True),
    Column("date", DateTime(), nullable=False, server_default=func.now()),
    make_string("tag"),
    # Benchmark info
    make_string("benchmark_desc"),
    make_string("benchmark"),
    Column("batch_size", Integer, nullable=True),
    Column("benchmark_params", JSON, nullable=False),
    # Backend info
    make_string("backend_desc"),
    make_string("host"),
    make_string("device"),
    make_string("compiler"),
    make_string("dtype"),
    # Results
    Column("duration_s", Float, nullable=False),
    Column("samples_per_s", Float, nullable=False),
    Column("samples_per_s_dirty", Float, nullable=True),
    Column("n_items", Integer, nullable=True),
    Column("p00", Float),
    Column("p50", Float),
    Column("p90", Float),
    Column("p100", Float),
    # This is actually benchmark property
    Column("flops_per_sample", Float, nullable=False),
)


class BenchmarkDb:
    def __init__(self, engine_string):
        self.engine = create_engine(engine_string)
        metadata_obj.create_all(self.engine, checkfirst=True)

    def report(self, **value_dict):
        stmt = insert(results_table).values(**value_dict)
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
