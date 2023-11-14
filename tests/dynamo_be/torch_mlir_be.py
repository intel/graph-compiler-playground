import torch_mlir
import torch
from torch_mlir.dynamo import make_simple_dynamo_backend
from functorch.compile import draw_graph, make_boxed_func
import torch._dynamo as dynamo
from typing import List
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

import os

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# for debug
# from functorch.compile import draw_graph, make_boxed_func, ts_compile
# import torch.fx as fx
# def graph_drawer(name):
#     def f(fx_g: fx.GraphModule, inps):
#         draw_graph(fx_g, name)
#         return fx_g

#     return f


@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(
    fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    # graph_drawer("mlir_fw")
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors"
    )
    print(mlir_module)
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        if not isinstance(result, tuple):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
        return result

    return compiled_callable


# model.train(False)
# dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(model)
