install:
    echo "Setting up environment..."
    uv venv --allow-existing
    uv sync
    uv pip install flash-attn --no-build-isolation

install-envs:
    vf-install vf-musique
    vf-install vf-musique-structured
    vf-install vf-musique-multi

patch:
    cp -f ./patches/vllm/api_server.py .venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py 
    cp -f ./patches/verifiers/stateful_tool_env.py .venv/lib/python3.12/site-packages/verifiers/envs/stateful_tool_env.py

setup:
    just install
    just patch
    just install-envs
    mkdir -p outputs/logs
    mkdir -p tmp/

