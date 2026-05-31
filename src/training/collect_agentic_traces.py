"""Collect AGENTIC teacher traces — model uses write_file/read_file/bash tools.

This is the rebuilt trace collector that fixes the structural mistake in the
prior `collect_vl_traces.py`: the teacher now actually CALLS TOOLS to write
artifacts, render them, and read them back. The harness only dispatches tool
calls; it does not do the file I/O or rendering work itself.

Loop per task:
  1. Spin up fresh Workspace tmpdir
  2. messages = [system, user_task]
  3. Repeat:
       a. Call teacher with messages + tools schema
       b. Parse tool_calls from assistant response
       c. For each tool_call: dispatch via Workspace, append tool result message
       d. If model emits <final>OK</final> with no tool_calls: stop
  4. Save trajectory (messages, tool_calls, tool_results, image paths)

Notes on API:
  - OpenRouter: pass tools=[...] in payload. Model returns assistant.tool_calls
    in OpenAI format (or content with <tool_call> XML — we handle both).
  - read_file returning image: dispatch result includes a PIL.Image. We save
    it to disk and embed it as a content part in a follow-up user message
    (since OpenAI tool-role messages can't carry image content over HTTP).

Usage:
    OPENROUTER_API_KEY=... python -m src.training.collect_agentic_traces \\
        --pilot --workers 1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image

from src.evaluation.code_eval import CodeEvaluator
from src.training.agent_workspace import (
    TOOLS_SCHEMA, Workspace, dispatch_tool_call,
)

logger = logging.getLogger(__name__)

TEACHER = "qwen/qwen3-vl-235b-a22b-instruct"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 600.0
MAX_TOKENS = 8000

# --- agentic teacher system prompt ----------------------------------------

_VERBOSE_TAIL = """
Common rendering commands you can use:
  # Playwright (preferred, usually pre-installed):
  bash(command="python3 -c \\"from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); pg=b.new_page(viewport={'width':1920,'height':1080}); pg.goto('file://' + __import__('os').getcwd() + '/slide.html', wait_until='networkidle'); pg.screenshot(path='slide.png', full_page=False); b.close(); p.stop()\\"")
  # Or chromium headless directly:
  bash(command="chromium --headless --no-sandbox --screenshot=slide.png --window-size=1920,1080 file://$PWD/slide.html")
"""

AGENT_SYSTEM_PROMPT_VERBOSE = """You are an autonomous engineer agent. You have three tools: write_file, read_file, and bash.

For ANY task, follow this verification loop:
1. Use write_file to create your initial artifact (e.g. solution.py, slide.html, index.html).
2. Verify your work using the appropriate method:
   - For code tasks: write the test file, then use bash to run it (e.g. `bash("python test_solution.py")`).
   - For visual tasks (HTML/CSS/SVG): use bash to render the artifact to a PNG (e.g. with chromium --headless --screenshot, or playwright via python -c '...'), then use read_file to view the resulting image.
3. Read the result. If a screenshot, examine it carefully and describe what you see.
4. If there are defects, write_file the revised artifact and verify again.
5. Repeat until the artifact passes verification (tests pass, or the rendered image meets the spec).
6. End with `<final>OK</final>` (a single line, no tools) when satisfied.

Hard rules:
- You MUST verify every artifact. Do not assume code or design works without running it.
- For visual tasks, you MUST use read_file on the rendered PNG. Looking at the HTML alone is not verification.
- Each revision must be a substantive edit addressing a specific defect you identified, not a re-emission of the same content.
- Use absolute filenames relative to the workspace root: write_file(path="slide.html", ...), bash(command="chromium --headless --screenshot=slide.png --window-size=1920,1080 file://$PWD/slide.html"), read_file(path="slide.png").
- The workspace starts empty. Create whatever files you need.
""" + _VERBOSE_TAIL

AGENT_SYSTEM_PROMPT_COMPACT = AGENT_SYSTEM_PROMPT_VERBOSE.replace(
    "- The workspace starts empty. Create whatever files you need.",
    """- The workspace starts empty. Create whatever files you need.

COMPACTNESS RULES (critical for downstream training):
- Each `write_file` call MUST keep `content` under 5000 characters. Long verbose CSS/HTML wastes tokens and breaks downstream consumers.
- Use minimal, terse CSS — no extensive utility class systems, no decorative comments, no verbose color palettes. Inline styles are fine when shorter.
- Prefer semantic HTML defaults over excessive class hierarchies. One CSS rule per element is usually enough.
- Do NOT pad with placeholder content beyond what the spec requires.
- If you find yourself writing >5000 chars for a single artifact, you are over-engineering. Cut the styling/markup until it fits."""
)

# Default = verbose (matches original Phase 6); set via --compact at CLI.
AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT_VERBOSE


# --- per-category user task framing ---------------------------------------

def build_user_task(task: dict, category: str, compact: bool = False) -> str:
    compact_hint = (
        " KEEP IT COMPACT — under 5000 chars total. Use minimal CSS, semantic defaults."
        if compact else ""
    )
    if category == "code":
        return (
            f"Task: {task['description']}\n\n"
            f"The workspace already contains test.py with the test suite (do not modify it). "
            f"Write your fix to solution.py — the test imports `from solution import *`. "
            f"Then run `bash python test.py` to verify. Iterate until the test prints 'All tests passed!' (exit code 0). "
            f"End with <final>OK</final> when done."
        )
    if category == "webpages":
        reqs = "\n".join(f"- {r}" for r in task.get("requirements", []))
        vps = task.get("viewport_sizes", [1920])
        vp_main = vps[0] if vps else 1920
        return (
            f"Task: {task['description']}\n\n"
            f"Requirements:\n{reqs}\n\n"
            f"Build a single self-contained HTML file (inline CSS/JS only, no external resources)."
            f"{compact_hint} "
            f"Render it at {vp_main}x1080 viewport using a headless browser, then read_file the PNG to verify. "
            f"Iterate until the page meets the spec. End with <final>OK</final>."
        )
    # slides
    reqs = "\n".join(f"- {r}" for r in task.get("requirements", []))
    return (
        f"Task: {task['description']}\n\n"
        f"Requirements:\n{reqs}\n\n"
        f"Build a single self-contained HTML file for a 1920x1080 slide (inline CSS only, no external resources)."
        f"{compact_hint} "
        f"Render it to a PNG at 1920x1080, then read_file the PNG to verify. "
        f"Iterate until the slide meets the spec. End with <final>OK</final>."
    )


# --- API call -------------------------------------------------------------

def call_teacher(messages: list[dict], extra_user_images: list[Image.Image] | None = None) -> dict:
    """Send messages to teacher with tools. Returns the raw assistant message dict."""
    payload = {
        "model": TEACHER,
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    resp = httpx.post(
        OR_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


# --- response parsing -----------------------------------------------------

TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*OK\s*</final>", re.IGNORECASE)


def _tolerant_parse_args(args_str: str, tool_name: str) -> dict:
    """Try strict JSON first; on failure, try regex-extracted fields per known tool."""
    if not isinstance(args_str, str):
        return args_str if isinstance(args_str, dict) else {}
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        pass
    # Heuristic fallback: pull out path/content/command via regex.
    # Common failure mode: model embeds Python code with unescaped " inside content.
    out = {}
    if tool_name == "write_file":
        m_path = re.search(r'"path"\s*:\s*"([^"]+)"', args_str)
        if m_path:
            out["path"] = m_path.group(1)
        # Content: from `"content": "` to the LAST `"` before the closing brace.
        m_c = re.search(r'"content"\s*:\s*"', args_str)
        if m_c:
            start = m_c.end()
            # Find the last `"` before final `}` in the args string
            tail = args_str[start:]
            # walk back from the end to find closing "}
            close = tail.rfind('"}')
            if close < 0:
                close = tail.rfind('"\n}')
            if close < 0:
                close = len(tail)
            content = tail[:close]
            # Unescape common JSON escapes the model probably DID get right
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            out["content"] = content
    elif tool_name == "bash":
        m_cmd = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', args_str)
        if m_cmd:
            out["command"] = m_cmd.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        m_to = re.search(r'"timeout"\s*:\s*(\d+)', args_str)
        if m_to:
            out["timeout"] = int(m_to.group(1))
    elif tool_name == "read_file":
        m_path = re.search(r'"path"\s*:\s*"([^"]+)"', args_str)
        if m_path:
            out["path"] = m_path.group(1)
    return out


def extract_tool_calls(message: dict) -> list[dict]:
    """Return [{name, arguments}] for every tool call in the message,
    handling both OpenAI tool_calls field AND inline <tool_call> XML."""
    out = []
    # Path A: structured tool_calls
    for tc in (message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = _tolerant_parse_args(fn.get("arguments"), name)
        out.append({
            "id": tc.get("id") or f"call_{len(out)}",
            "name": name,
            "arguments": args,
        })
    # Path B: inline XML in content (Qwen3-VL native format)
    content = message.get("content") or ""
    if isinstance(content, str):
        for m in TOOL_CALL_XML_RE.finditer(content):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {}
            out.append({
                "id": f"xml_call_{len(out)}",
                "name": name,
                "arguments": args,
            })
    return out


# --- image embedding ------------------------------------------------------

def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_tool_response_message(call_id: str, result: dict) -> tuple[dict, dict | None]:
    """Build the tool-role message for a dispatch result.

    Returns (tool_message, follow_up_user_message_or_None).
    For image results, the tool message gets text saying 'image follows' and a
    follow-up user message carries the image content (since OpenAI tool-role
    can't reliably carry image content over the wire).
    """
    if not result.get("ok", False):
        # Failure
        body = json.dumps({"error": result.get("error", "unknown")})
        return {"role": "tool", "tool_call_id": call_id, "content": body}, None
    kind = result.get("kind")
    if kind == "image":
        # Tool message: short text result; user follow-up: the image
        text_summary = json.dumps({
            "kind": "image",
            "path": result.get("path"),
            "size": result.get("size"),
            "see": "next user message for the rendered image content",
        })
        tool_msg = {"role": "tool", "tool_call_id": call_id, "content": text_summary}
        img = result["image"]
        follow_up = {"role": "user", "content": [
            {"type": "text", "text": f"Image content of {result.get('path')!r}:"},
            {"type": "image_url", "image_url": {"url": _png_data_url(img)}},
        ]}
        return tool_msg, follow_up
    elif kind == "text":
        body = json.dumps({"path": result.get("path"), "content": result.get("content"),
                           "truncated": result.get("truncated", False)})
        return {"role": "tool", "tool_call_id": call_id, "content": body}, None
    else:
        # write_file / bash
        return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}, None


# --- collection loop ------------------------------------------------------

def run_one_trajectory(task: dict, category: str, max_steps: int,
                       evaluator: CodeEvaluator, traj_dir: Path,
                       compact: bool = False) -> dict:
    """Run one task end-to-end. Returns trajectory record."""
    ws = Workspace()
    # For code tasks, pre-populate test.py so the model doesn't transcribe wrong.
    if category == "code":
        test_src = (
            "from solution import *\n\n"
            f"{task['test_code']}\n"
        )
        ws.write_file("test.py", test_src)
    user_task = build_user_task(task, category, compact=compact)
    sys_prompt = AGENT_SYSTEM_PROMPT_COMPACT if compact else AGENT_SYSTEM_PROMPT_VERBOSE
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_task},
    ]
    record_steps: list[dict] = []
    saved_images: list[str] = []  # paths to images saved to traj_dir
    final_seen = False

    try:
        for step in range(max_steps):
            # 1. Call teacher
            t0 = time.time()
            try:
                msg = call_teacher(messages)
            except Exception as e:
                logger.warning("teacher call failed on %s step %d: %s",
                               task["task_id"], step, e)
                return {
                    "task_id": task["task_id"], "category": category,
                    "status": "api_error", "error": str(e),
                    "steps": record_steps,
                }
            gen_s = round(time.time() - t0, 2)

            content = msg.get("content") or ""
            tool_calls = extract_tool_calls(msg)
            record_steps.append({
                "step": step,
                "assistant_content": content,
                "tool_calls": [{"name": t["name"], "arguments": t["arguments"]}
                               for t in tool_calls],
                "gen_s": gen_s,
            })

            # 2. Append assistant message to history
            #    Use OpenAI tool_calls format for the API turn
            if tool_calls:
                api_assistant = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{
                        "id": t["id"],
                        "type": "function",
                        "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])},
                    } for t in tool_calls],
                }
            else:
                api_assistant = {"role": "assistant", "content": content}
            messages.append(api_assistant)

            # 3. If <final>OK</final> with no tool_calls → done
            if not tool_calls and FINAL_RE.search(content):
                final_seen = True
                break

            # 4. If no tool_calls and no final → bail
            if not tool_calls:
                logger.info("[%s/%s] step %d: no tool calls, no final → ending",
                            category, task["task_id"], step)
                break

            # 5. Dispatch each tool_call, append tool message + optional follow-up
            for tc in tool_calls:
                result = dispatch_tool_call(ws, tc["name"], tc["arguments"])
                # Save images to disk so we keep them for SFT
                if result.get("kind") == "image" and "image" in result:
                    img_path = traj_dir / f"step{step}_{len(saved_images):03d}.png"
                    result["image"].save(img_path)
                    saved_images.append(str(img_path))

                tool_msg, follow_up = make_tool_response_message(tc["id"], result)
                messages.append(tool_msg)
                if follow_up is not None:
                    messages.append(follow_up)
                record_steps[-1].setdefault("tool_results", []).append({
                    "tool": tc["name"],
                    "ok": result.get("ok"),
                    "kind": result.get("kind"),
                    "summary": str({k: v for k, v in result.items() if k != "image"})[:300],
                })

        # 6. Final code-eval if applicable: run task['test_code'] against last solution.py
        final_passed = None
        if category == "code":
            sol_path = ws.root / "solution.py"
            if sol_path.exists():
                code = sol_path.read_text(encoding="utf-8", errors="replace")
                try:
                    res = evaluator.evaluate(code, task["test_code"], timeout=10)
                    final_passed = res.passed
                except Exception:
                    final_passed = False

        return {
            "task_id": task["task_id"], "category": category,
            "status": "final" if final_seen else "max_steps",
            "n_steps": len(record_steps),
            "final_passed": final_passed,
            "saved_images": saved_images,
            "steps": record_steps,
            "messages": messages,    # full conversation for later SFT prep
        }
    finally:
        ws.cleanup()


# --- main -----------------------------------------------------------------

def load_tasks(category: str) -> list[dict]:
    paths = {
        "code": ["data/hard_benchmarks/code/code_tasks.json"],
        "webpages": ["data/hard_benchmarks/webpages/webpage_tasks.json",
                     "data/training/vl_webpage_tasks_gen.json"],
        "slides": ["data/hard_benchmarks/slides/slide_tasks.json",
                   "data/training/vl_slide_tasks_gen.json"],
    }
    out = []
    for p in paths[category]:
        if Path(p).exists():
            out.extend(json.loads(Path(p).read_text()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="1 task per category for plumbing test")
    ap.add_argument("--n-per-category", type=int, default=0,
                    help="override pilot size; 0 = all tasks")
    ap.add_argument("--categories", nargs="+",
                    default=["code", "webpages", "slides"])
    ap.add_argument("--max-steps", type=int, default=20,
                    help="max tool-call cycles per trajectory")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", default="data/training/agentic_traces.json")
    ap.add_argument("--images-dir", default="data/training/agentic_images")
    ap.add_argument("--log", default="logs/collect_agentic.log")
    ap.add_argument("--compact", action="store_true",
                    help="Use compact teacher prompt (cap content at 5KB)")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set"); sys.exit(1)

    images_root = Path(args.images_dir)
    images_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = CodeEvaluator()

    all_jobs = []
    for cat in args.categories:
        tasks = load_tasks(cat)
        n = args.n_per_category or (1 if args.pilot else len(tasks))
        for t in tasks[:n]:
            all_jobs.append((cat, t))
    logger.info("Queued %d agentic trajectories across %s",
                len(all_jobs), args.categories)

    traces = []
    if out_path.exists():
        traces = json.loads(out_path.read_text())
        done = {(t["category"], t["task_id"]) for t in traces}
        all_jobs = [(c, t) for c, t in all_jobs if (c, t["task_id"]) not in done]
        logger.info("Resuming. %d done; %d remaining.", len(traces), len(all_jobs))

    lock = threading.Lock()

    def worker(cat: str, task: dict):
        traj_dir = images_root / f"{cat}_{task['task_id']}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rec = run_one_trajectory(task, cat, args.max_steps, evaluator, traj_dir,
                                  compact=args.compact)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        with lock:
            traces.append(rec)
            out_path.write_text(json.dumps(traces, indent=2, default=str))
            logger.info(
                "[done %d] %s/%s status=%s steps=%d elapsed=%.1fs final_passed=%s",
                len(traces), cat, task["task_id"], rec.get("status"),
                rec.get("n_steps", 0), rec["elapsed_s"], rec.get("final_passed"),
            )

    if args.workers <= 1:
        for cat, task in all_jobs:
            worker(cat, task)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker, c, t) for c, t in all_jobs]
            for _ in as_completed(futures):
                pass

    logger.info("Done. %d trajectories -> %s", len(traces), out_path)


if __name__ == "__main__":
    main()
