### Generation with Quantization
import argparse
import asyncio
import json
import os
import logging
import numpy as np
from pathlib import Path
import time
import torch

from transformers import AutoTokenizer

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi import BuildConfig, CalibConfig, QuantAlgo, QuantConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.llmapi.build_cache import BuildCache, BuildCacheConfig

# os.environ['TLLM_LLMAPI_BUILD_CACHE'] == '1'
# os.environ['TLLM_LLMAPI_BUILD_CACHE_ROOT'] = ''


major, minor = torch.cuda.get_device_capability()
enable_fp8 = major > 8 or (major == 8 and minor >= 9)

quant_and_calib_configs = []

if False:
    # Example 1: Specify int4 AWQ quantization to QuantConfig.
    # We can skip specifying CalibConfig or leave a None as the default value.
    quant_and_calib_configs.append(
        (QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ), None))

if enable_fp8:
    # Example 2: Specify FP8 quantization to QuantConfig.
    # We can create a CalibConfig to specify the calibration dataset and other details.
    # Note that the calibration dataset could be either HF dataset name or a path to local HF dataset.
    quant_and_calib_configs.append(
        (QuantConfig(quant_algo=QuantAlgo.FP8,
                     kv_cache_quant_algo=QuantAlgo.FP8),
         CalibConfig(calib_dataset='cnn_dailymail',
                     calib_batches=32,
                     calib_max_seq_length=126976)))



async def gen_async(llm, prompt, sampling_params, stream):

    async for output in llm.generate_async(prompt, streaming=stream, sampling_params = sampling_params):
        for out in output.outputs:
            print(out)

    ttfts = []
    decodes = []
    for iter in range(10):
        ttft = None
        last_time = time.perf_counter()
        async for output in llm.generate_async(prompt, streaming=stream, sampling_params = sampling_params):
            for out in output.outputs:
                print(out)
                # out = out.result(timeout=10)
                # print(
                #     f"Prompt: {output.prompt!r}, Generated text: {out.outputs[0].text!r}"
                # )
                latency = round(time.perf_counter() - last_time, 5)*1000
                if ttft == None:
                    ttft = latency
                    ttfts.append(ttft)
                else:
                    decodes.append(latency)
                print(f"i: {out.index}, len: {len(out.token_ids)}, text: {out.text}, finish_reason: {out.finish_reason}, latency: {round(time.perf_counter() - last_time, 5)*1000} ms")
                last_time = time.perf_counter()
    print(f"avg ttft: {round(np.mean(ttfts), 3)} ms, decode: {round(np.mean(decodes), 3)} ms.") 

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_prompt(model_path):
    messages = load_json("/mnt/data/zhangjun/mydev/auto_bench/dataset/gemini-32k-64k.json")[10]['messages']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    prompt = prompt[:4096]
    return prompt

def main(args):

    # The built-in end-to-end quantization is triggered according to the passed quant_config.
    quant_config = quant_and_calib_configs[0][0]
    calib_config = quant_and_calib_configs[0][1]
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.95,
        enable_block_reuse = True,
    )
    # read fron engine_dir or set explictly
    build_config = BuildConfig(
        max_num_tokens=4096,
        max_batch_size=48,
        max_seq_len=32768,
        max_input_len=32768-2048,
        max_beam_width=1
    )
    build_cache_config = BuildCacheConfig(cache_root=Path("./tmp/trt_caches"))
    llm_args = {
        "workspace": "./tmp/",
        "enable_build_cache": build_cache_config,
    }
    model_path = "/mnt/data/zhangjun/mydev/models/qw25_2050_agent_ppt_coder_0520_0528"
    llm = LLM(
        # model="/mnt/data/zhangjun/mydev/trtllm/tmp/qwen32b/Qwen/Qwen2.5-32B-Instruct/tp_2_pp_1/",
        model=model_path,
        tokenizer = model_path,
        tensor_parallel_size=2,
        quant_config=quant_config,
        # calib_config=calib_config,
        build_config=build_config,
        kv_cache_config=kv_cache_config,
        **llm_args
    )
    if args.build:
        engine_dir = "./tmp/engines/tp2pp1"
        llm.save(engine_dir)
        print(f"Finish build engine. Save to {engine_dir}")
        return

    # Sample prompts.
    prompt = get_prompt(model_path)
    prompts = [
        prompt,
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9,  max_tokens=8)

    stream = True
    if stream:
        # for output in llm.generate_async(prompts[2], streaming=stream, sampling_params = sampling_params):
        #     for out in output:
        #         out = out.result(timeout=10)
        #         print(
        #             f"Prompt: {output.prompt!r}, Generated text: {out.outputs[0].text!r}"
        #         )
        asyncio.run(gen_async(llm, prompts[0], sampling_params, True))  
    else:
        for output in llm.generate(prompts, sampling_params):
            print(
                f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
            )
    llm.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build',
                        action='store_true',
                        default=False,
                        help='Building and save engines')
    args = parser.parse_args()    
    main(args)