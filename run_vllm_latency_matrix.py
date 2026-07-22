#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

aiohttp = None

warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

try:
    import pynvml
except Exception:  # noqa: BLE001 - optional monitor.
    pynvml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_FILE = ROOT / "prompt_cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "vllm_runs"
DEFAULT_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
DEFAULT_CASE_ID = "latency_20page_ava_and_the_glowing_pebble"
DEFAULT_CUDA_HOME = os.environ.get(
    "CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")
)

SYSTEM_PROMPT = """# You are a Creative and Detail-Oriented Children's Storybook AI

Your job is to generate a complete storybook package in three structured
stages:

## 🔹 Stage 1 -- Story Creation (Generator)

**Inputs You Will Accept:**

-   Story Title or Summary
-   Story Type -- Fiction, Non-Fiction, Moral, Mystery, Fantasy, Comic,
    Fairy Tale, Biography, Poetry, Science Fiction, Ghost Story, Horror
-   Target Age Group -- Picture Books (3--5), Early Readers (5--7),
    Young Readers (7--12), Young Adults (12+)
-   Number of Characters -- 1--2
-   Number of Pages -- 1--32

**Optional Inputs:**
- Book Title (if not provided → system generates one)
- Author Name (if not provided → system generates placeholder like "By
[Author's Name]")
- Child's Name (if provided → story must make this child the hero/main
character)

**Rules:**
- If title/author are provided → use them exactly.
- If not provided → auto-generate a creative title and placeholder
author.
- Always include child's name in the story text if user provides one.
- Language must match age group.
- Each page = at least 2--3 sentences.
- Moral/educational stories must end with a clear lesson.

**Output Format:**

    Story Title: [Given or auto-generated]  
    Author: [Given or auto-generated]  

    Page Content:  
    Page 1  
     Content: [...]  
    Page 2  
     Content: [...]  

------------------------------------------------------------------------

## 🔹 Stage 2 -- Story Processing (Processor)

**Step 1: Character Design Sheet**
Identify all recurring characters.
For each, write a highly descriptive paragraph including:

-   Age & gender
-   Hair style & color
-   Eye color & facial expressions
-   Clothing, shoes, accessories
-   Distinct quirks/mannerisms

**Example:**
Finnley is a 6-year-old boy with a mop of curly blonde hair that sticks
out in every direction and bright, sparkling blue eyes that light up
with wonder. He wears a faded blue t-shirt with a picture of a rocket
ship on it and a pair of worn-out sneakers that have seen their fair
share of adventures.

**Step 2: Story Page Division**
- Divide story into 1--3 sentences per page.
- Smooth pacing and clarity.

**Step 3: Page-by-Page Output**
For each page:
- Page Story Text → rewritten text.
- Page Image Prompt → ultra-detailed scene description.
- After Image Prompt → always paste character description(s) for every
character present on that page.

**Image Prompt Rules:**
- Describe environment, time of day, mood, actions, props.
- Always paste the full character description(s) directly after the
image scene.
- If 1 character appears → paste that description.
- If 2 characters appear → paste both.
- If 3+ characters appear → paste all of them.
- Always end with:
`Style: 3d-Pixar Style`

**Output Example:**

    🌱 Story Title: Milo and the Talking Tree  
    Author: [Given or Auto-generated]  

    🔹 Character Design Sheet  
    Milo:  
    Milo is a 5-year-old boy with messy brown hair that curls slightly at the ends and wide hazel eyes full of curiosity. He wears a green dinosaur hoodie that's a little too big for him, tan shorts, and yellow rain boots. His cheeks are always flushed, and he carries a red backpack covered in stickers.  

    The Talking Tree:  
    The Talking Tree is a grand old oak with deep grooves in its bark that resemble a wise face. Its eyes are made of golden sap that glows faintly, and its leaves shimmer in different shades of green. Its roots curl gently like open arms.  

    📘 Page-by-Page Output  

    **Page 1: Story Text**  
    Milo wandered into the forest with his backpack, looking for something magical. He had heard stories about a tree that could talk.  

    **Page 1: Image Prompt**  
    A little boy walks through a dense forest filled with sunbeams and rustling leaves. Tall trees tower around him, and a small path winds ahead. Milo clutches his red backpack as he gazes into the woods, full of excitement and wonder.  
    Milo is a 5-year-old boy with messy brown hair that curls slightly at the ends and wide hazel eyes full of curiosity. He wears a green dinosaur hoodie that's a little too big for him, tan shorts, and yellow rain boots. His cheeks are always flushed, and he carries a red backpack covered in stickers.  
    Style: 3d-Pixar Style  

------------------------------------------------------------------------

## 🔹 Stage 3 -- Book Cover Creation (Cover Designer)

Using the character design sheet + story essence + title + author:

Write one detailed paragraph describing the front cover illustration.

**Must include:**
- Characters (all design details from sheet)
- Story theme environment
- Atmosphere/emotion
- If title and author are given → use them exactly.
- If not → auto-generate a creative title + placeholder author.
- Title at the top in bold lettering.
- Author name below title.
- Always end with:
`Style: 3d-Pixar Style`

**Example:**

    Image Prompt:  
    A warm, sunlit garden filled with colorful blossoms, radiating joy and wonder. At the center, Sophie, a curious 6-year-old girl with golden-brown pigtails, bright green eyes, and a cheerful yellow sundress with daisy patterns, stands in her bright red mary jane shoes. Beside her, Grandpa, a kind man in his late 60s with soft gray hair and warm blue eyes, wears a light blue shirt with rolled sleeves, brown gardening overalls, and sturdy boots. They smile as they admire the flowers together. At the top of the cover, the title appears in bold playful lettering: “Sophie’s Garden Surprise.” Below it, the author’s name is clearly written: “By Emily Carter.”  
    Style: 3d-Pixar Style  

------------------------------------------------------------------------

## ✅ Final Output Order

1.  Story (Stage 1)
2.  Character Design Sheet + Page-by-Page Story with Prompts (Stage 2)
3.  Each Image Prompt must always be followed by the full description(s)
    of every character appearing on that page.
4.  Cover Illustration Prompt (Stage 3)
"""


@dataclass
class RequestResult:
    source_model: str
    model_label: str
    case_id: str
    max_num_seqs: int
    max_num_batched_tokens: int
    concurrency: int
    request_index: int
    ok: bool
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_chars: int
    output_preview: str
    input_payload: dict[str, Any]
    output_text: str
    prompt_variant_index: int | None = None
    error: str | None = None


def request_metrics_record(result: RequestResult) -> dict[str, Any]:
    record = asdict(result)
    record.pop("source_model")
    record.pop("input_payload")
    record.pop("output_text")
    return record


def quality_output_record(result: RequestResult) -> dict[str, Any]:
    return {
        "model": result.source_model,
        "model_label": result.model_label,
        "case_id": result.case_id,
        "max_num_seqs": result.max_num_seqs,
        "max_num_batched_tokens": result.max_num_batched_tokens,
        "concurrency": result.concurrency,
        "request_index": result.request_index,
        "prompt_variant_index": result.prompt_variant_index,
        "input": result.input_payload,
        "output": result.output_text,
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "output_chars": result.output_chars,
    }


class NvmlMonitor:
    def __init__(self, interval_s: float = 0.05, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: list[dict[str, float]] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._handle = None

    async def __aenter__(self) -> "NvmlMonitor":
        if pynvml is None:
            return self
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        if pynvml is not None:
            pynvml.nvmlShutdown()

    async def _run(self) -> None:
        assert pynvml is not None
        while not self._stop.is_set():
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except Exception:  # noqa: BLE001
                power_w = 0.0
            self.samples.append(
                {
                    "t": time.perf_counter(),
                    "used_mib": mem.used / (1024 * 1024),
                    "gpu_util_pct": float(util.gpu),
                    "mem_util_pct": float(util.memory),
                    "power_w": power_w,
                }
            )
            await asyncio.sleep(self.interval_s)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "peak_vram_mib": None,
                "avg_vram_mib": None,
                "avg_gpu_util_pct": None,
                "peak_gpu_util_pct": None,
                "avg_power_w": None,
                "peak_power_w": None,
            }
        return {
            "peak_vram_mib": max(s["used_mib"] for s in self.samples),
            "avg_vram_mib": statistics.mean(s["used_mib"] for s in self.samples),
            "avg_gpu_util_pct": statistics.mean(s["gpu_util_pct"] for s in self.samples),
            "peak_gpu_util_pct": max(s["gpu_util_pct"] for s in self.samples),
            "avg_power_w": statistics.mean(s["power_w"] for s in self.samples),
            "peak_power_w": max(s["power_w"] for s in self.samples),
        }


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def slugify(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-._") or "item"


def parse_model(value: str) -> tuple[str, str]:
    if "=" in value:
        label, model = value.split("=", 1)
    else:
        model = value
        label = Path(model).stem if Path(model).suffix else model.split("/")[-1]
    return slugify(label), model


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases file must contain a JSON list")
    return data


def select_cases(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = cases
    if args.difficulty:
        wanted = set(args.difficulty)
        selected = [case for case in selected if case["difficulty"] in wanted]
    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(sorted(missing))}")
    if not args.all and not args.case and not args.difficulty:
        selected = [case for case in selected if case["id"] == DEFAULT_CASE_ID]
    return selected


def request_input(case: dict[str, Any], request_index: int) -> tuple[str, int | None]:
    variants = case.get("variants") or []
    if variants:
        variant_index = request_index % len(variants)
        return str(variants[variant_index]), variant_index
    return str(case["input"]), None


def build_messages(user_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, timeout_s: float = 2.0) -> dict[str, Any] | None:
    try:
        req = Request(url, headers={"accept": "application/json"})
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - local benchmark URL.
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def wait_for_server(base_url: str, proc: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        if http_json(url) is not None:
            return
        time.sleep(1.0)
    raise TimeoutError(f"vLLM server did not become ready within {timeout_s}s: {url}")


def vllm_command(
    *,
    args: argparse.Namespace,
    model_label: str,
    model: str,
    port: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> list[str]:
    if args.vllm_cmd:
        cmd = shlex.split(args.vllm_cmd)
    else:
        vllm_from_env = Path(sys.executable).parent / "vllm"
        vllm_bin = str(vllm_from_env) if vllm_from_env.exists() else shutil.which("vllm")
        cmd = [vllm_bin, "serve"] if vllm_bin else [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    cmd += [
        model,
        "--served-model-name",
        model_label,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
    ]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.hf_config_path:
        cmd += ["--hf-config-path", args.hf_config_path]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    if args.kv_cache_dtype:
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    if args.extra_serve_arg:
        for item in args.extra_serve_arg:
            cmd += shlex.split(item)
    return [part for part in cmd if part]


def start_server(
    *,
    args: argparse.Namespace,
    model_label: str,
    model: str,
    run_dir: Path,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> tuple[subprocess.Popen[str], str, Path]:
    port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    cmd = vllm_command(
        args=args,
        model_label=model_label,
        model=model,
        port=port,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    log_path = run_dir / f"server_{model_label}_seq{max_num_seqs}_tok{max_num_batched_tokens}.log"
    env = os.environ.copy()
    active_bin = str(Path(sys.executable).parent)
    cuda_home = args.cuda_home
    env["PATH"] = active_bin + os.pathsep + str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    cuda_root = Path(cuda_home)
    include_candidates = [
        cuda_root / "targets" / "x86_64-linux" / "include",
        cuda_root / "include",
    ]
    library_candidates = [
        cuda_root / "targets" / "x86_64-linux" / "lib",
        cuda_root / "lib64",
    ]
    cuda_include = next((path for path in include_candidates if path.is_dir()), None)
    cuda_lib = next((path for path in library_candidates if path.is_dir()), None)
    if cuda_include is not None:
        env["CPATH"] = str(cuda_include) + os.pathsep + env.get("CPATH", "")
        env["CPLUS_INCLUDE_PATH"] = str(cuda_include) + os.pathsep + env.get("CPLUS_INCLUDE_PATH", "")
    if cuda_lib is not None:
        env["LIBRARY_PATH"] = str(cuda_lib) + os.pathsep + env.get("LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(cuda_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("VLLM_USE_V1", args.vllm_use_v1)
    print("Starting vLLM server:", " ".join(shlex.quote(p) for p in cmd), flush=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(ROOT),
        start_new_session=True,
    )
    try:
        wait_for_server(base_url, proc, args.server_timeout_s)
    except Exception:
        log.close()
        stop_server(proc)
        raise
    log.close()
    return proc, base_url, log_path


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


async def one_request(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    source_model: str,
    model_label: str,
    case: dict[str, Any],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    concurrency: int,
    request_index: int,
    args: argparse.Namespace,
) -> RequestResult:
    user_input, prompt_variant_index = request_input(case, request_index)
    payload: dict[str, Any] = {
        "model": model_label,
        "messages": build_messages(user_input),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens if args.max_tokens is not None else int(case["max_tokens"]),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.extra_body:
        payload.update(json.loads(args.extra_body))
    start = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, int] = {}
    pieces: list[str] = []
    try:
        async with session.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=args.request_timeout_s) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {body[:1000]}")
            async for raw in resp.content:
                for line in raw.decode("utf-8", "ignore").splitlines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        continue
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if piece:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            pieces.append(piece)
        end = time.perf_counter()
        text = "".join(pieces)
        return RequestResult(
            source_model=source_model,
            model_label=model_label,
            case_id=case["id"],
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            concurrency=concurrency,
            request_index=request_index,
            ok=True,
            latency_s=end - start,
            ttft_s=None if first_token_at is None else first_token_at - start,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            output_chars=len(text),
            output_preview=text[:500],
            input_payload=payload,
            output_text=text,
            prompt_variant_index=prompt_variant_index,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark records request failures.
        text = "".join(pieces)
        return RequestResult(
            source_model=source_model,
            model_label=model_label,
            case_id=case["id"],
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            concurrency=concurrency,
            request_index=request_index,
            ok=False,
            latency_s=time.perf_counter() - start,
            ttft_s=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            output_chars=len(text),
            output_preview=text[:500],
            input_payload=payload,
            output_text=text,
            prompt_variant_index=prompt_variant_index,
            error=str(exc),
        )


async def run_concurrency(
    *,
    base_url: str,
    source_model: str,
    model_label: str,
    case: dict[str, Any],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    concurrency: int,
    total_requests: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    connector = aiohttp.TCPConnector(limit=max(concurrency, 16))
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        async with NvmlMonitor(args.monitor_interval_s, args.gpu_index) as monitor:
            wall_start = time.perf_counter()

            async def guarded(i: int) -> RequestResult:
                async with sem:
                    return await one_request(
                        session,
                        base_url=base_url,
                        source_model=source_model,
                        model_label=model_label,
                        case=case,
                        max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_num_batched_tokens,
                        concurrency=concurrency,
                        request_index=i,
                        args=args,
                    )

            results = await asyncio.gather(*(guarded(i) for i in range(total_requests)))
            wall_s = time.perf_counter() - wall_start
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    latencies = [r.latency_s for r in ok]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": model_label,
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "prompt_variant_count": len(case.get("variants") or []) or 1,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "wall_s": wall_s,
        "request_per_s": len(ok) / wall_s if wall_s else 0.0,
        "prompt_tok_s": prompt_tokens / wall_s if wall_s else 0.0,
        "output_tok_s": completion_tokens / wall_s if wall_s else 0.0,
        "total_tok_s": (prompt_tokens + completion_tokens) / wall_s if wall_s else 0.0,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "ttft_avg_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "latency_avg_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        **monitor.summary(),
        "requests": [request_metrics_record(r) for r in results],
        "quality_outputs": [quality_output_record(r) for r in ok],
    }
    return summary


async def warmup(
    base_url: str,
    source_model: str,
    model_label: str,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.warmup_requests <= 0:
        return
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(args.warmup_requests):
            result = await one_request(
                session,
                base_url=base_url,
                source_model=source_model,
                model_label=model_label,
                case=case,
                max_num_seqs=0,
                max_num_batched_tokens=0,
                concurrency=1,
                request_index=i,
                args=args,
            )
            if not result.ok:
                print(f"Warmup failed: {result.error}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM server + concurrent latency matrix for storybook prompts.")
    parser.add_argument("--model", action="append", default=None, metavar="LABEL=MODEL", help="HF model id/path or GGUF path. May be repeated.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer id/path, primarily for GGUF models.")
    parser.add_argument("--hf-config-path", default=None, help="HF config id/path, useful for GGUF.")
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    parser.add_argument("--case", action="append")
    parser.add_argument("--difficulty", action="append", choices=["easy", "medium", "hard"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 2, 4, 8], help="Comma-separated concurrent request counts.")
    parser.add_argument("--requests-per-concurrency", type=int, default=0, help="0 means concurrency * repeats_per_slot.")
    parser.add_argument("--repeats-per-slot", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=parse_int_list, default=[8], help="vLLM scheduler max sequences, comma-separated.")
    parser.add_argument("--max-num-batched-tokens", type=parse_int_list, default=[8192], help="vLLM scheduler token budget, comma-separated.")
    parser.add_argument("--pair-server-settings", action="store_true", help="Pair seq/token lists by index instead of cartesian product.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--extra-body", default=None, help="JSON merged into each OpenAI request body.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--extra-serve-arg", action="append", help="Extra args appended to vLLM serve command. May be repeated.")
    parser.add_argument("--vllm-cmd", default=None, help="Override launch command, e.g. 'vllm serve'.")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port per server config.")
    parser.add_argument("--server-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--monitor-interval-s", type=float, default=0.05)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME)
    parser.add_argument("--vllm-use-v1", default="1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def server_settings(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.pair_server_settings:
        tokens = args.max_num_batched_tokens
        if len(tokens) == 1 and len(args.max_num_seqs) > 1:
            tokens = tokens * len(args.max_num_seqs)
        if len(tokens) != len(args.max_num_seqs):
            raise SystemExit("With --pair-server-settings, token list must have one value or match seq list length")
        return list(zip(args.max_num_seqs, tokens))
    return [(seqs, toks) for seqs in args.max_num_seqs for toks in args.max_num_batched_tokens]


async def main_async() -> int:
    global aiohttp
    args = build_parser().parse_args()
    cases = load_cases(args.cases_file.expanduser())
    if args.list_cases:
        for case in cases:
            variant_count = len(case.get("variants") or []) or 1
            print(f"{case['id']}\t{case['difficulty']}\tmax_tokens={case['max_tokens']}\tvariants={variant_count}\t{case.get('description', '')}")
        return 0
    selected_cases = select_cases(args, cases)
    models = [parse_model(item) for item in (args.model or [DEFAULT_MODEL])]
    settings = server_settings(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.expanduser() / run_id

    print(f"Run id: {run_id}")
    print(f"Output root: {run_dir}")
    print("Models:")
    for label, model in models:
        print(f"  - {label}: {model}")
    print("Server settings:")
    for seqs, toks in settings:
        print(f"  - max_num_seqs={seqs}, max_num_batched_tokens={toks}")
    print("Client concurrency:", ",".join(str(x) for x in args.concurrency))
    print("Cases:")
    for case in selected_cases:
        print(f"  - {case['id']} ({case['difficulty']})")

    if args.dry_run:
        return 0

    try:
        import aiohttp as aiohttp_module
    except ImportError as exc:
        raise SystemExit("aiohttp is required for live vLLM latency runs. Install it in the vLLM env.") from exc
    aiohttp = aiohttp_module

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.jsonl"
    summary_path.write_text("", encoding="utf-8")
    request_path = run_dir / "requests.jsonl"
    request_path.write_text("", encoding="utf-8")
    quality_output_path = run_dir / "quality_outputs.jsonl"
    quality_output_path.write_text("", encoding="utf-8")

    for model_label, model in models:
        for max_num_seqs, max_num_batched_tokens in settings:
            proc: subprocess.Popen[str] | None = None
            try:
                proc, base_url, log_path = start_server(
                    args=args,
                    model_label=model_label,
                    model=model,
                    run_dir=run_dir,
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                )
                print(f"Server ready: {base_url} (log: {log_path})", flush=True)
                for case in selected_cases:
                    await warmup(base_url, model, model_label, case, args)
                    for concurrency in args.concurrency:
                        total_requests = args.requests_per_concurrency or concurrency * args.repeats_per_slot
                        print(
                            f"Benchmark model={model_label} case={case['id']} seqs={max_num_seqs} "
                            f"batched_tokens={max_num_batched_tokens} concurrency={concurrency} requests={total_requests}",
                            flush=True,
                        )
                        summary = await run_concurrency(
                            base_url=base_url,
                            source_model=model,
                            model_label=model_label,
                            case=case,
                            max_num_seqs=max_num_seqs,
                            max_num_batched_tokens=max_num_batched_tokens,
                            concurrency=concurrency,
                            total_requests=total_requests,
                            args=args,
                        )
                        requests = summary.pop("requests")
                        quality_outputs = summary.pop("quality_outputs")
                        with summary_path.open("a", encoding="utf-8") as out:
                            out.write(json.dumps(summary, sort_keys=True) + "\n")
                        with request_path.open("a", encoding="utf-8") as out:
                            for request in requests:
                                out.write(json.dumps(request, sort_keys=True) + "\n")
                        with quality_output_path.open("a", encoding="utf-8") as out:
                            for quality_output in quality_outputs:
                                out.write(json.dumps(quality_output, ensure_ascii=False, sort_keys=True) + "\n")
                        print(
                            f"  ok={summary['successful_requests']}/{summary['total_requests']} "
                            f"ttft_p50={summary['ttft_p50_s']} latency_p50={summary['latency_p50_s']} "
                            f"out_tok_s={summary['output_tok_s']:.2f}",
                            flush=True,
                        )
            finally:
                if proc is not None:
                    stop_server(proc)

    print(f"Summary: {summary_path}")
    print(f"Requests: {request_path}")
    print(f"Quality outputs: {quality_output_path}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
