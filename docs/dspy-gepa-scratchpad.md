export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager


export MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
CUDA_VISIBLE_DEVICES=3 vf-vllm --model $MODEL \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager


export MODEL="Qwen/Qwen3-8B"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager

export MODEL="Qwen/Qwen3-32B"
CUDA_VISIBLE_DEVICES=3 vf-vllm --model $MODEL \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager

export MODEL="Qwen/Qwen3-8B"
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager

export MODEL="Qwen/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enforce-eager
        
export MODEL="Qwen/Qwen3-8B"
CUDA_VISIBLE_DEVICES=3 vf-vllm --model $MODEL \
    --port 8001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enforce-eager
        