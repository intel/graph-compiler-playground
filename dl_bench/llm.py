import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dl_bench.utils import TimerManager, Benchmark


def get_llm(name, dtype=torch.float32):
    if name != "gptj":
        raise ValueError("Unsupported model name")

    model_name = "EleutherAI/gpt-j-6B"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return tokenizer, model


class LlmBenchmark(Benchmark):
    def __init__(self, params) -> None:
        name = params.get("name", "gptj")
        self.tokenizer, self.model = get_llm(name)
        self.warmup_prompt = "There are several ways to travel, but my favourite is"
        self.prompt = "Here is a story about a person that find out he was adopted: one day little Timmy was looking through old"
        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": 128,
            "min_new_tokens": 30,
            "num_beams": 4,
        }

    def generate(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        start = time.perf_counter()
        gen_tokens = self.model.generate(
            input_ids, **self.gen_kwargs, pad_token_id=self.tokenizer.eos_token_id
        )
        total_time = time.perf_counter() - start

        # text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_tokens[0], total_time

    def inference(self, backend):
        tm = TimerManager()

        # Recover MACs computation
        # generate requires several forward passes, so need addtional algo to estimate
        # self.flops_per_sample = get_macs(self.model, self.in_shape, backend) * 2

        self.model = backend.prepare_eval_transformer(self.model)

        print("Warmup started")
        with torch.no_grad(), tm.timeit("warmup_s"):
            self.model.eval()
            self.generate(self.warmup_prompt)
        print("Warmup done")

        self.model.eval()
        with torch.no_grad(), tm.timeit("duration_s"):
            tokens, total_time = self.generate(self.prompt)
            outputs = [tokens]

        results = tm.get_results()
        results["samples_per_s"] = len(tokens) / total_time
        results["flops_per_sample"] = 1

        return results, outputs
