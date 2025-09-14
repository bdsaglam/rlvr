install:
    echo "Setting up environment..."
    uv venv --allow-existing
    uv sync
    uv pip install flash-attn --no-build-isolation
    vf-install vf-musique
    vf-install vf-musique-structured

patch-vllm:
    cp -f ./services/vllm/api_server.py .venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py 

setup:
    just install
    just patch-vllm
    mkdir -p outputs/logs
    mkdir -p tmp/

