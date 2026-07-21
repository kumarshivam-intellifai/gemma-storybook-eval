#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VLLM_PY = Path(sys.executable)
MATRIX = ROOT / 'run_vllm_latency_matrix.py'
DEFAULT_CASE = 'latency_20page_ava_and_the_glowing_pebble'
DEFAULT_HF_HOME = Path(
    os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')
).expanduser()
DEFAULT_CUDA_HOME = os.environ.get(
    'CUDA_HOME', os.environ.get('CUDA_PATH', '/usr/local/cuda')
)


def parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return values


def model_label(model: str) -> str:
    if '=' in model:
        return model.split('=', 1)[0]
    return model.rstrip('/').split('/')[-1].lower().replace('.', '-').replace('_', '-')


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def print_table(summary_path: Path) -> None:
    rows = [json.loads(line) for line in summary_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    headers = ['concurrency', 'ok/total', 'ttft_p50_s', 'ttft_p95_s', 'latency_p50_s', 'latency_p95_s', 'output_tok_s']
    print('\n' + ' | '.join(headers))
    print(' | '.join(['---'] * len(headers)))
    for row in rows:
        values = [
            row.get('concurrency'),
            f"{row.get('successful_requests')}/{row.get('total_requests')}",
            fmt(row.get('ttft_p50_s')),
            fmt(row.get('ttft_p95_s')),
            fmt(row.get('latency_p50_s')),
            fmt(row.get('latency_p95_s')),
            fmt(row.get('output_tok_s'), 2),
        ]
        print(' | '.join(str(v) for v in values))


def main() -> int:
    parser = argparse.ArgumentParser(description='Simple vLLM latency runner: pass model + concurrency, get TTFT/latency/output_tok_s table.')
    parser.add_argument('--model', required=True, help='HF-format model id/path, optionally label=model. Do not use GGUF for vLLM here.')
    parser.add_argument('--concurrency', default='1,4,8,16,32,64', help='Comma-separated concurrent request counts.')
    parser.add_argument('--case', default=DEFAULT_CASE, help='Prompt case id. Defaults to a standalone 20-page latency case.')
    parser.add_argument('--max-num-seqs', type=int, default=0, help='vLLM max active sequences. 0 means max(concurrency).')
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192)
    parser.add_argument('--requests-per-concurrency', type=int, default=0, help='0 means concurrency * repeats-per-slot.')
    parser.add_argument('--repeats-per-slot', type=int, default=1, help='With requests-per-concurrency=0, total requests = concurrency * repeats.')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Maximum generated tokens per request.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--max-model-len', type=int, default=12288)
    parser.add_argument('--cuda-home', default=DEFAULT_CUDA_HOME)
    parser.add_argument('--hf-home', type=Path, default=DEFAULT_HF_HOME, help='HF cache root; reused to avoid repeated downloads.')
    parser.add_argument(
        '--enable-prefix-caching',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable vLLM prefix caching; use --no-enable-prefix-caching to disable it.',
    )
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--extra-serve-arg', action='append', default=[])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    concurrencies = parse_ints(args.concurrency)
    max_num_seqs = args.max_num_seqs or max(concurrencies)
    run_id = args.run_id or f"simple_{model_label(args.model)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    cmd = [
        str(VLLM_PY),
        str(MATRIX),
        '--model', args.model,
        '--case', args.case,
        '--max-num-seqs', str(max_num_seqs),
        '--max-num-batched-tokens', str(args.max_num_batched_tokens),
        '--concurrency', ','.join(str(v) for v in concurrencies),
        '--requests-per-concurrency', str(args.requests_per_concurrency),
        '--repeats-per-slot', str(args.repeats_per_slot),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--max-model-len', str(args.max_model_len),
        '--cuda-home', args.cuda_home,
        '--run-id', run_id,
        f"--extra-serve-arg={'--enable-prefix-caching' if args.enable_prefix_caching else '--no-enable-prefix-caching'}",
    ]
    if args.max_tokens is not None:
        cmd += ['--max-tokens', str(args.max_tokens)]
    for item in args.extra_serve_arg:
        cmd.append(f'--extra-serve-arg={item}')
    if args.dry_run:
        cmd.append('--dry-run')

    env = os.environ.copy()
    env['HF_HOME'] = str(args.hf_home)
    env['HF_HUB_CACHE'] = str(args.hf_home / 'hub')
    env['CUDA_HOME'] = args.cuda_home
    env['CUDA_PATH'] = args.cuda_home

    print('Using vLLM python:', VLLM_PY)
    print('Using HF cache:', env['HF_HOME'])
    print('Prefix caching:', args.enable_prefix_caching)
    print('Run id:', run_id)
    print('Command:', ' '.join(cmd))
    completed = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if completed.returncode != 0:
        return completed.returncode
    if args.dry_run:
        return 0

    summary_path = ROOT / 'vllm_runs' / run_id / 'summary.jsonl'
    if not summary_path.exists():
        print(f'Missing summary file: {summary_path}', file=sys.stderr)
        return 2
    print_table(summary_path)
    print(f'\nSummary: {summary_path}')
    print(f'Requests: {ROOT / "vllm_runs" / run_id / "requests.jsonl"}')
    print(f'Quality outputs: {ROOT / "vllm_runs" / run_id / "quality_outputs.jsonl"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
