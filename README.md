# Gemma NVFP4 simple vLLM inference benchmark

This repository contains the standalone inference and latency runner used for
`nvidia/Gemma-4-26B-A4B-NVFP4`. It starts a local vLLM server, sends the
storybook prompt as streaming OpenAI-compatible chat requests, and reports
time-to-first-token (TTFT), end-to-end latency, and output-token throughput.

This is the simple inference benchmark only. It does not contain or run
Promptfoo.

## Files

- `run_vllm_simple_latency.py`: command-line entrypoint and summary table
- `run_vllm_latency_matrix.py`: vLLM server lifecycle and concurrent request engine
- `prompt_cases.json`: 64 available storybook inputs; the default is the original 20-page Ava case
- `requirements.txt`: verified vLLM version

## Requirements

- Linux with an NVIDIA GPU and enough VRAM for the model
- A compatible NVIDIA driver and CUDA toolkit
- Python 3.10+
- Hugging Face access to the model, if its license requires acceptance

The verified environment used Python 3.10.12, vLLM 0.25.1,
PyTorch 2.11.0+cu130, and Transformers 5.14.1.

## Install

```bash
git clone https://github.com/kumarshivam-intellifai/gemma-storybook-promptfoo-eval.git
cd gemma-storybook-promptfoo-eval

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Hugging Face authentication is required, export a token in your shell; do
not commit it:

```bash
export HF_TOKEN=your_token_here
```

## Run the same inference test

Activate the vLLM virtual environment and run:

```bash
python run_vllm_simple_latency.py \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --gpu-memory-utilization 0.90
```

The default concurrency sweep is `1,4,8,16,32,64`, with one request per active
slot. For a quick single-request check:

```bash
python run_vllm_simple_latency.py \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --concurrency 1 \
  --gpu-memory-utilization 0.90
```

Prefix caching is enabled explicitly by default. Run the same test with it
disabled without editing source code:

```bash
python run_vllm_simple_latency.py \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

If CUDA is not installed at `/usr/local/cuda`, pass its location:

```bash
python run_vllm_simple_latency.py \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --cuda-home /usr/local/cuda-13.1
```

To inspect the generated vLLM command without starting the model:

```bash
python run_vllm_simple_latency.py \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --dry-run
```

## Results

Each run creates `vllm_runs/<run-id>/` containing:

- `summary.jsonl`: aggregate TTFT, latency, throughput, and GPU metrics
- `requests.jsonl`: per-request metrics and errors
- `quality_outputs.jsonl`: complete generated text for inspection
- `server_*.log`: the vLLM server log

The runner also prints a compact table after the benchmark completes. Model
weights, Hugging Face credentials, caches, and benchmark outputs are excluded
from Git.
