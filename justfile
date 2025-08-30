setup:
    echo "Setting up environment..."
    uv venv --allow-existing
    uv sync
    uv pip install flash-attn --no-build-isolation
    vf-install vf-musique -p environments
