# Gemma NVFP4 storybook evaluation

This repository runs the complete 64-prompt storybook evaluation against
[`nvidia/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4)
through a local vLLM OpenAI-compatible server. Promptfoo produces JSON and HTML
reports, and the included Python grader checks page structure, title/author
fidelity, character/object coverage, moral coverage, image prompts, and cover
requirements.

The repository contains test code and prompts only. It does not contain model
weights, Hugging Face credentials, virtual environments, caches, or generated
reports.

## Requirements

- Linux with an NVIDIA GPU and enough VRAM for the model
- A compatible NVIDIA driver/CUDA installation
- Python 3.10+
- Node.js 20+
- Hugging Face access to the model, if its license requires acceptance

The verified environment used Python 3.10.12, vLLM 0.25.1,
PyTorch 2.11.0+cu130, Transformers 5.14.1, Node.js 24.18.0, and Promptfoo
0.121.19. The Python requirements pin vLLM; its package dependencies install a
compatible serving stack for the selected platform.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd promptfoo_eval
npm ci
```

If required, authenticate with Hugging Face without saving a token in this
repository:

```bash
export HF_TOKEN=your_token_here
```

## Run all 64 prompts

From `promptfoo_eval` with the Python environment activated:

```bash
./run_gemma_promptfoo_full_eval.sh
```

You can also point the runner at an existing vLLM environment explicitly:

```bash
VLLM_BIN=/path/to/venv/bin/vllm \
PYTHON_BIN=/path/to/venv/bin/python \
./run_gemma_promptfoo_full_eval.sh
```

The defaults reproduce the verified server configuration:

- model: `nvidia/Gemma-4-26B-A4B-NVFP4`
- GPU memory utilization: `0.90`
- maximum model length: `32768`
- maximum sequences: `1`
- maximum batched tokens: `8192`
- Promptfoo concurrency: `1`
- temperature: `0.7`
- top-p: `0.9`
- prefix caching: enabled explicitly

Supported overrides include:

```bash
GPU_MEMORY_UTILIZATION=0.85 MAX_MODEL_LEN=24576 ./run_gemma_promptfoo_full_eval.sh
```

To compare with prefix caching disabled, do not edit any source file:

```bash
PREFIX_CACHING=false ./run_gemma_promptfoo_full_eval.sh
```

## Results

Each invocation creates a timestamped directory under
`promptfoo_eval/results/`. Important files are:

- `results.html`: shareable Promptfoo HTML report
- `results.json`: complete machine-readable Promptfoo output
- `summary.json` and `cases.csv`: compact raw-result summaries
- `regraded_summary.json` and `regraded_cases.csv`: independently audited scores
- `vllm_server.log`: server startup and runtime log

Promptfoo may return a non-zero exit code when one or more quality assertions
fail. That does not mean the evaluation failed to run; inspect `results.html`
and the audited summary. Server startup or dependency errors are printed with a
path to the vLLM log.

## Repository layout

```text
.
├── prompt_cases.json
├── system_prompt.txt
├── requirements.txt
└── promptfoo_eval/
    ├── promptfooconfig_full.yaml
    ├── run_gemma_promptfoo_full_eval.sh
    ├── generate_prompt.py
    ├── generate_full_tests.py
    ├── storybook_assertions_full.py
    ├── summarize_full_results.py
    └── regrade_full_results.py
```
