import os
import time
import math

import torch
# import intel_extension_for_pytorch as ipex
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from dl_bench.utils import Benchmark, get_report, get_time, str_to_dtype


def get_llm(name, dtype):
    name2params = {
        "gptj": ("EleutherAI/gpt-j-6B", AutoModelForCausalLM, AutoTokenizer),
        "llama2-7b": ("meta-llama/Llama-2-7b-hf", LlamaForCausalLM, LlamaTokenizer),
        "llama2-13b": ("meta-llama/Llama-2-13b-hf", LlamaForCausalLM, LlamaTokenizer),
    }

    if name not in name2params:
        raise ValueError("Unsupported model name")

    kwargs = {}
    if name.startswith("llama2") and "HF_TOKEN" in os.environ:
        kwargs = {"HF_TOKEN": os.environ.get("HF_TOKEN")}

    model_name, M, T = name2params[name]

    model = M.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    tokenizer = T.from_pretrained(model_name)
    return tokenizer, model


class LlmBenchmark(Benchmark):
    def __init__(self, params) -> None:
        name = params.get("name", "gptj")
        dtype = params.get("dtype")
        self.batch_size = int(params.get("batch_size", 1))
        self.n_iter = params.get("n_iter", 5)
        self.warmup_batches = params.get("warmup", 2)

        self.tokenizer, self.model = get_llm(name, dtype=str_to_dtype(dtype))
        prompt = "Here is a story about a person that find out he was adopted: one day little Timmy was looking through old"
        self.prompt = [prompt] * self.batch_size
        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": 128,
            "min_new_tokens": 30,
            "num_beams": 4,
        }

    def generate(self, backend):
        backend.sync()
        start = time.perf_counter()
        input_tokens = self.tokenizer(self.prompt, return_tensors="pt").input_ids
        input_tokens = backend.to_device(input_tokens)
        gen_tokens = self.model.generate(
            input_tokens, **self.gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
        )
        backend.sync()
        text = self.tokenizer.batch_decode(gen_tokens)[0]
        total_time = time.perf_counter() - start

        # new tokens are a subset of all tokens
        output_tokens = gen_tokens[:, input_tokens.shape[1] :]
        return output_tokens, total_time

    def inference(self, backend):
        # TODO: Recover MACs computation
        # generate requires several forward passes, so need addtional algo to estimate
        # self.flops_per_sample = get_macs(self.model, self.in_shape, backend) * 2
        self.model = backend.prepare_eval_transformer(self.model)

        enabled = backend.dtype != torch.float32

        n_items = 0
        outputs = []
        fw_times = []


        # Ipex gives error with eval, other backends have no effect
        # self.model.eval()
        for i in range(self.n_iter):
            print(f"Epoch {i+1}/{self.n_iter}")
            cast = torch.autocast(enabled=enabled, device_type=backend.device_name)
            with torch.inference_mode(), cast:
                tokens, total_time = self.generate(backend)

            print(f"Fw time: {total_time:.1f}")

            if i < self.warmup_batches:
                # We restart timer because that was just a warmup
                start = get_time()
                continue

            fw_times.append(total_time)
            n_items += math.prod(tokens.shape)
            outputs.append(tokens)

        stop = get_time()

        report = get_report(
            fw_times=fw_times,
            duration_s=stop - start,
            n_items=n_items,
            flops_per_sample=1,
        )
        return report, outputs
