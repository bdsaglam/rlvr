# Troubleshooting

## NCCL Weight Synchronization Errors

### Problem Description

During RLVR training with weight synchronization between vLLM server and training processes, you may encounter NCCL errors like:

```
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
RuntimeError: NCCL error: unhandled system error
```

This happens during the weight sync initialization phase when the training client tries to establish NCCL communication with the vLLM server workers. Regular chat completions work fine - the issue only occurs during online RL training that requires weight updates.

### Root Cause Analysis

**Hardware Context:**
Our system uses NVIDIA A100 80GB PCIe cards with the following specifications:
- Product: `NVIDIA A100 80GB PCIe` (Part #900-21001-0020-100)
- GPU Topology: All connections are `SYS` (system interconnect via PCIe + CPU)
- NVLink Status: `Unable to retrieve NVLink information as all links are inActive`

**GPU Interconnect Types:**

**SYS (System Interconnect) - Our Hardware:**
- Communication path: GPU → PCIe → CPU → System Bus → CPU → PCIe → GPU
- Bandwidth: ~32 GB/s (PCIe 4.0 x16)
- Latency: High (~microseconds)
- Used by: PCIe form factor GPUs

**NVLink (Not Available on Our Hardware):**
- Communication path: GPU → NVLink → GPU (direct)
- Bandwidth: 300-900 GB/s depending on generation
- Latency: Low (~nanoseconds)
- Used by: SXM form factor GPUs in DGX/HGX systems

**Why NCCL Fails:**
1. NCCL attempts peer-to-peer (P2P) GPU memory access
2. PCIe A100s cannot do P2P through SYS interconnect
3. NCCL falls back to unsupported communication methods
4. Driver compatibility issues with older NVIDIA drivers (535.x series)

### Solution

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_GDR_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring
```

**What Each Variable Does:**
- `NCCL_P2P_DISABLE=1`: Disables peer-to-peer GPU memory access (main fix)
- `NCCL_IB_DISABLE=1`: Disables InfiniBand networking (not available)
- `NCCL_SOCKET_IFNAME=lo`: Forces loopback interface for local communication
- `NCCL_NET_GDR_DISABLE=1`: Disables GPU Direct RDMA
- `NCCL_TREE_THRESHOLD=0` + `NCCL_ALGO=Ring`: Forces ring algorithm for collectives

**vLLM Server Startup:**
```bash
vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
      --tensor-parallel-size 1  --gpu-memory-utilization 0.7 \
      --enforce-eager --disable-log-requests
```

### Hardware Limitations

**Cannot Enable NVLink:**
NVLink requires specific hardware that our PCIe A100s do not have:
- Physical NVLink connectors/bridges between GPUs
- SXM form factor (not PCIe)
- Integrated baseboard (DGX/HGX systems)

Our PCIe cards have NVLink controllers on-chip but they are physically disconnected.

### References
- [GitHub Issue #181](https://github.com/willccbb/verifiers/issues/181)
- Hardware confirmed via `nvidia-smi nvlink --status` showing "inActive" links
- System topology confirmed via `nvidia-smi topo -m` showing SYS interconnect


## vLLM Server NaN Serialization Issues

```sh
export MODEL="willcb/Qwen3-8B"

CUDA_VISIBLE_DEVICES=0 uv run scripts/vllm_server.py \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```