install:
    echo "Setting up environment..."
    uv venv --allow-existing
    uv sync
    uv pip install flash-attn --no-build-isolation

patch:
    cp -f ./patches/vllm/api_server.py .venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py 

install-envs:
    vf-install vf-musique 
    vf-install vf-musique-multi 
    vf-install math-python 
    vf-install gsm8k --from-repo

setup:
    just install
    just patch
    just install-envs
    mkdir -p outputs/logs
    mkdir -p outputs/dspy
    mkdir -p outputs/musique-eval
    mkdir -p tmp/

start-services:
    docker-compose up --build