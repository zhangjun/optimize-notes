# https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html
MODEL_PATH=tmp/search_model/Qwen/Qwen2.5-32B-Instruct/tp_2_pp_1
TOKENIZER_PATH=/mnt/data/zhangjun/mydev/models/search_model/

MODEL_PATH=tmp/engines/tp2pp1/
TOKENIZER_PATH=/mnt/data/zhangjun/mydev/models/ppt_coder/

# --backend pytorch \
# --reasoning_parser deepseek-r1
trtllm-serve \
    ${MODEL_PATH} \
    --host localhost --port 8000 \
    --max_batch_size 48 --max_num_tokens 4096 \
    --tp_size 2 --ep_size 1 --pp_size 1 \
    --kv_cache_free_gpu_memory_fraction 0.95 \
    --extra_llm_api_options ./extra-llm-api-config.yml \
    --tokenizer ${TOKENIZER_PATH}