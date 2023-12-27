# Should be imported from
# from torch_mlir_e2e_test.reporting import ValueReport, ErrorContext
import torch
from typing import Any, List, Optional, Set


class TensorSummary:
    """A summary of a tensor's contents."""

    def __init__(self, tensor):
        self.min = torch.min(tensor.type(torch.float64))
        self.max = torch.max(tensor.type(torch.float64))
        self.mean = torch.mean(tensor.type(torch.float64))
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __str__(self):
        return f"Tensor with shape={self.shape}, dtype={self.dtype}, min={self.min:+0.4}, max={self.max:+0.4}, mean={self.mean:+0.4}"


class ErrorContext:
    """A chained list of error contexts.

    This is useful for tracking errors across multiple levels of detail.
    """

    def __init__(self, contexts: List[str]):
        self.contexts = contexts

    @staticmethod
    def empty():
        """Create an empty error context.

        Used as the top-level context.
        """
        return ErrorContext([])

    def chain(self, additional_context: str):
        """Chain an additional context onto the current error context."""
        return ErrorContext(self.contexts + [additional_context])

    def format_error(self, s: str):
        return "@ " + "\n@ ".join(self.contexts) + "\n" + "ERROR: " + s


class ValueReport:
    """A report for a single value processed by the program."""

    def __init__(self, value, golden_value, context: ErrorContext):
        self.value = value
        self.golden_value = golden_value
        self.context = context
        self.failure_reasons = []
        self._evaluate_outcome()

    @property
    def failed(self):
        return len(self.failure_reasons) != 0

    def error_str(self):
        return "\n".join(self.failure_reasons)

    def _evaluate_outcome(self):
        value, golden = self.value, self.golden_value
        if isinstance(golden, float):
            if not isinstance(value, float):
                return self._record_mismatch_type_failure("float", value)
            if abs(value - golden) / golden > 1e-4:
                return self._record_failure(
                    f"value ({value!r}) is not close to golden value ({golden!r})"
                )
            return
        if isinstance(golden, int):
            if not isinstance(value, int):
                return self._record_mismatch_type_failure("int", value)
            if value != golden:
                return self._record_failure(
                    f"value ({value!r}) is not equal to golden value ({golden!r})"
                )
            return
        if isinstance(golden, str):
            if not isinstance(value, str):
                return self._record_mismatch_type_failure("str", value)
            if value != golden:
                return self._record_failure(
                    f"value ({value!r}) is not equal to golden value ({golden!r})"
                )
            return
        if isinstance(golden, tuple):
            if not isinstance(value, tuple):
                return self._record_mismatch_type_failure("tuple", value)
            if len(value) != len(golden):
                return self._record_failure(
                    f"value ({len(value)!r}) is not equal to golden value ({len(golden)!r})"
                )
            reports = [
                ValueReport(v, g, self.context.chain(f"tuple element {i}"))
                for i, (v, g) in enumerate(zip(value, golden))
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, list):
            if not isinstance(value, list):
                return self._record_mismatch_type_failure("list", value)
            if len(value) != len(golden):
                return self._record_failure(
                    f"value ({len(value)!r}) is not equal to golden value ({len(golden)!r})"
                )
            reports = [
                ValueReport(v, g, self.context.chain(f"list element {i}"))
                for i, (v, g) in enumerate(zip(value, golden))
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, dict):
            if not isinstance(value, dict):
                return self._record_mismatch_type_failure("dict", value)
            gkeys = list(sorted(golden.keys()))
            vkeys = list(sorted(value.keys()))
            if gkeys != vkeys:
                return self._record_failure(
                    f"dict keys ({vkeys!r}) are not equal to golden keys ({gkeys!r})"
                )
            reports = [
                ValueReport(
                    value[k],
                    golden[k],
                    self.context.chain(f"dict element at key {k!r}"),
                )
                for k in gkeys
            ]
            for report in reports:
                if report.failed:
                    self.failure_reasons.extend(report.failure_reasons)
            return
        if isinstance(golden, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                return self._record_mismatch_type_failure("torch.Tensor", value)

            if value.shape != golden.shape:
                return self._record_failure(
                    f"shape ({value.shape}) is not equal to golden shape ({golden.shape})"
                )
            if value.dtype != golden.dtype:
                return self._record_failure(
                    f"dtype ({value.dtype}) is not equal to golden dtype ({golden.dtype})"
                )
            if not torch.allclose(
                value, golden, rtol=1e-03, atol=1e-04, equal_nan=True
            ):
                return self._record_failure(
                    f"value ({TensorSummary(value)}) is not close to golden value ({TensorSummary(golden)})"
                )
            return
        return self._record_failure(
            f"unexpected golden value of type `{golden.__class__.__name__}`"
        )

    def _record_failure(self, s: str):
        self.failure_reasons.append(self.context.format_error(s))

    def _record_mismatch_type_failure(self, expected: str, actual: Any):
        self._record_failure(
            f"expected a value of type `{expected}` but got `{actual.__class__.__name__}`"
        )


def compare(taken, golden):
    if len(taken) != len(golden):
        raise ValueError(
            f"Golden comparison mismatch: tensor lengths are different, {len(golden)} != {len(taken)}"
        )

    res = ValueReport(taken, golden, ErrorContext([]))
    if res.failed:
        print(res.error_str())
        raise ValueError("Golden comparison mismatch")
    else:
        print("CMP Match")
    return res
