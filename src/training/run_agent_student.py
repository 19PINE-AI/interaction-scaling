"""Run the agentic student (Qwen3-VL-8B-Thinking ± LoRA) on tasks with tool
dispatch through a per-trajectory workspace.

Mirrors `collect_agentic_traces.py` but the agent is a local HF model rather
than a remote OpenRouter call. The model emits `<tool_call>{json}</tool_call>`
XML which we parse, dispatch via `agent_workspace.Workspace`, and feed back as
`tool` role messages.

Usage:
    PYTHONPATH=. python -m src.training.run_agent_student \\
        --adapter models/qwen3-vl-8b-agentic-sft-v1 \\
        --tasks data/training/heldout_phase5.json \\
        --output results/phase6/student_traces.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from src.training.agent_workspace import Workspace, dispatch_tool_call, TOOLS_SCHEMA
from src.training.collect_agentic_traces import (
    AGENT_SYSTEM_PROMPT, build_user_task,
)
from src.training.run_vl_student import (
    load_model, ForceCloseThinking, CLOSE_THINK_ID,
)

logger = logging.getLogger(__name__)

FINAL_RE = re.compile(r"<final>\s*OK\s*</final>", re.IGNORECASE)
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*", re.DOTALL)
TOOL_CALL_CLOSE_RE = re.compile(r"\s*</tool_call>", re.DOTALL)


STUDENT_SYSTEM_PROMPT = (
    "You are an autonomous engineer agent. Use the provided tools "
    "(write_file, read_file, bash) to solve the user's task. "
    "For visual tasks, render the artifact to a PNG via bash and "
    "use read_file to view it before revising. "
    "When the task is complete, emit <final>OK</final> in your final "
    "assistant message and stop calling tools."
)


def messages_to_inputs(processor, messages, tools):
    """Build model inputs from chat messages + tool schemas."""
    images: list[Image.Image] = []
    msgs_flat = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) or c is None:
            d = {"role": m["role"], "content": c or ""}
            if m.get("tool_calls"):
                d["tool_calls"] = m["tool_calls"]
            msgs_flat.append(d)
            continue
        parts = []
        for p in c:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                # may be PIL.Image or filepath str
                img = p["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images.append(img)
                parts.append({"type": "image"})
        msgs_flat.append({"role": m["role"], "content": parts})
    text = processor.apply_chat_template(
        msgs_flat, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=images if images else None,
        return_tensors="pt", padding=False,
    )
    return inputs


@torch.no_grad()
def generate(processor, model, messages, tools, max_new_tokens=20000,
             temperature=0.0, repetition_penalty=1.0,
             force_close_think_after=0):
    inputs = messages_to_inputs(processor, messages, tools).to(model.device)
    do_sample = temperature > 0
    logits_processors = []
    if force_close_think_after > 0:
        logits_processors.append(ForceCloseThinking(
            prompt_len=inputs["input_ids"].shape[1],
            force_after=force_close_think_after,
        ))
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        repetition_penalty=repetition_penalty,
        pad_token_id=processor.tokenizer.pad_token_id,
        logits_processor=logits_processors if logits_processors else None,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def parse_tool_calls_from_text(text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """Extract `<tool_call>{json}</tool_call>` blocks robustly.

    Uses json.JSONDecoder.raw_decode to find the JSON object boundary, so it
    works even if the closing `</tool_call>` tag is missing (which happens
    when the model hits max_new_tokens mid-emission).

    Returns (calls, spans) where calls is a list of {name, arguments} dicts
    and spans is a list of (start, end) text indices for each parsed tool_call
    block (used by strip_tool_calls).
    """
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    decoder = json.JSONDecoder()
    while pos < len(text):
        open_m = TOOL_CALL_OPEN_RE.search(text, pos)
        if not open_m:
            break
        json_start = open_m.end()
        # Skip whitespace
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1
        if json_start >= len(text) or text[json_start] != "{":
            pos = open_m.end()
            continue
        # Find the close tag, if any (block_end is bound for tolerant fallback).
        close_search_m = TOOL_CALL_CLOSE_RE.search(text, json_start)
        close_pos = close_search_m.start() if close_search_m else len(text)
        try:
            obj, end_idx = decoder.raw_decode(text, json_start)
            block_end = (close_search_m.end() if close_search_m
                         and close_search_m.start() == end_idx else end_idx)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
        except json.JSONDecodeError:
            # Malformed JSON (commonly: model emitted Python code with
            # un-escaped `"` inside the `content` string field). Try a tolerant
            # regex extraction from the bytes between the tags.
            inside = text[json_start:close_pos]
            tol = _tolerant_extract_tool_call(inside)
            if tol is None:
                pos = open_m.end()
                continue
            name, args = tol
            block_end = close_search_m.end() if close_search_m else close_pos
        calls.append({"name": name, "arguments": args})
        spans.append((open_m.start(), block_end))
        pos = block_end
    return calls, spans


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"', re.DOTALL)
_PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"', re.DOTALL)
_COMMAND_RE = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _tolerant_extract_tool_call(s: str) -> tuple[str, dict] | None:
    """Best-effort extraction when JSON is malformed (model emitted unescaped
    quotes inside `content`). Returns (name, args_dict) or None.

    For `write_file`: extracts path via JSON regex, content via brace-balanced
    scan from `"content": "` to the final `"` before `}}`.
    For `bash`: extracts command via JSON regex.
    For `read_file`: extracts path via JSON regex.
    """
    nm = _NAME_RE.search(s)
    if not nm:
        return None
    name = nm.group(1)
    if name == "write_file":
        path_m = _PATH_RE.search(s)
        if not path_m:
            return None
        # Find the content field, then scan to the LAST `"}}` (close of content
        # string + close of arguments + close of outer object). This is robust
        # to unescaped `"` inside the content body.
        i = s.find('"content"', path_m.end())
        if i < 0:
            return None
        i = s.find('"', i + len('"content"'))
        if i < 0:
            return None
        content_start = s.find('"', i + 1)
        if content_start < 0:
            return None
        content_start += 1
        # Scan from end backwards for `"}}` or `"}`
        for marker in ('"\n}}', '"}}', '"}\n}', '"}'):
            j = s.rfind(marker)
            if j > content_start:
                content = s[content_start:j]
                # Decode escapes minimally (\n, \t, \\, \"). Other escapes
                # left as-is.
                content = (content.replace("\\n", "\n").replace("\\t", "\t")
                                  .replace('\\"', '"').replace("\\\\", "\\"))
                return (name, {"path": path_m.group(1), "content": content})
        return None
    if name == "bash":
        cm = _COMMAND_RE.search(s)
        if not cm:
            return None
        cmd = cm.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        return (name, {"command": cmd})
    if name == "read_file":
        pm = _PATH_RE.search(s)
        if not pm:
            return None
        return (name, {"path": pm.group(1)})
    return None


def strip_tool_calls(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove tool_call blocks from text using their (start,end) spans."""
    if not spans:
        return text.strip()
    out = []
    cur = 0
    for s, e in spans:
        out.append(text[cur:s])
        cur = e
    out.append(text[cur:])
    return "".join(out).strip()


def make_tool_response(result: dict) -> tuple[dict, dict | None]:
    """Build a `tool` role message. If result is image, return (tool_msg,
    None) where tool_msg has image content directly inline (Qwen3-VL chat
    template renders this correctly).
    """
    if result.get("kind") == "image" and "image" in result:
        img = result["image"]
        parts = [{"type": "text", "text": "screenshot:"},
                 {"type": "image", "image": img}]
        return {"role": "tool", "content": parts}, img
    # text/error result — JSON-stringify (drop PIL.Image if any)
    safe = {k: v for k, v in result.items() if k != "image"}
    return {"role": "tool", "content": json.dumps(safe)}, None


def run_one_trajectory(processor, model, task: dict, max_steps: int = 16,
                       temperature: float = 0.0,
                       repetition_penalty: float = 1.1,
                       force_close_think_after: int = 4000,
                       save_image_dir: Path | None = None) -> dict:
    category = task["category"]
    ws = Workspace()
    if category == "code":
        # Pre-populate test.py so the model just makes it pass
        test_src = "from solution import *\n\n" + task["test_code"]
        (ws.root / "test.py").write_text(test_src)

    user_task = build_user_task(task, category)
    messages: list[dict] = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]
    tools = TOOLS_SCHEMA
    saved_images: list[str] = []
    saved_image_idx = 0
    steps_log: list[dict] = []
    final_status = "max_steps"
    final_passed = None

    try:
        for step_idx in range(max_steps):
            t0 = time.time()
            try:
                gen = generate(processor, model, messages, tools,
                               temperature=temperature,
                               repetition_penalty=repetition_penalty,
                               force_close_think_after=force_close_think_after)
            except Exception as e:
                logger.warning("generate failed: %s", e)
                final_status = "gen_error"
                break
            gen_s = round(time.time() - t0, 2)

            tcs, spans = parse_tool_calls_from_text(gen)
            content_text = strip_tool_calls(gen, spans)

            assistant_msg = {"role": "assistant", "content": content_text}
            if tcs:
                assistant_msg["tool_calls"] = [
                    {"type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in tcs
                ]
            messages.append(assistant_msg)

            step_log = {"step": step_idx, "gen_s": gen_s,
                        "assistant_content": content_text,
                        "tool_calls": [
                            {"function": {"name": tc["name"],
                                          "arguments": json.dumps(tc["arguments"])}}
                            for tc in tcs
                        ],
                        "tool_results": []}

            # Final detection: if no tool calls AND <final>OK</final>, stop
            if not tcs:
                if FINAL_RE.search(content_text):
                    final_status = "final"
                steps_log.append(step_log)
                break

            for tc in tcs:
                result = dispatch_tool_call(ws, tc["name"], tc["arguments"])
                # Build tool result summary for logging
                if result.get("kind") == "image" and "image" in result:
                    # Save image to disk for trace replay/inspection
                    if save_image_dir:
                        save_image_dir.mkdir(parents=True, exist_ok=True)
                        img_path = save_image_dir / f"step{step_idx}_{saved_image_idx:03d}.png"
                        result["image"].save(img_path, "PNG", optimize=True)
                        saved_images.append(str(img_path))
                        saved_image_idx += 1
                    summary = {"ok": result.get("ok"), "kind": "image",
                               "path": result.get("path"),
                               "size": result.get("size")}
                else:
                    summary_obj = {k: v for k, v in result.items() if k != "image"}
                    s = json.dumps(summary_obj)[:400]
                    summary = {"ok": result.get("ok"),
                               "kind": result.get("kind"),
                               "summary": s}
                step_log["tool_results"].append({"tool": tc["name"], **summary})

                tool_msg, _ = make_tool_response(result)
                messages.append(tool_msg)

            steps_log.append(step_log)

        # Code outcome: if a solution.py exists and tests pass, set final_passed
        if category == "code":
            sol_path = ws.root / "solution.py"
            if sol_path.exists():
                try:
                    proc = ws.bash(f"python test.py", timeout=60)
                    final_passed = bool(proc.get("ok"))
                except Exception:
                    final_passed = False

    finally:
        ws.cleanup()

    return {
        "task_id": task["task_id"],
        "category": category,
        "status": final_status,
        "n_steps": len(steps_log),
        "final_passed": final_passed,
        "saved_images": saved_images,
        "steps": steps_log,
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; omit for base-model eval")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--force-close-think-after", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-dir", default="results/phase6/student_images")
    ap.add_argument("--log", default="logs/run_agent_student.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    torch.manual_seed(args.seed)
    processor, model = load_model(args.model, args.adapter,
                                   load_in_4bit=args.load_in_4bit,
                                   load_in_8bit=args.load_in_8bit)
    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("Loaded %d tasks", len(tasks))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_dir)
    results: list[dict] = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done = {(r["category"], r["task_id"]) for r in results}
        tasks = [t for t in tasks if (t["category"], t["task_id"]) not in done]
        logger.info("Resuming. %d tasks remaining.", len(tasks))

    for i, t in enumerate(tasks):
        cat = t["category"]
        t0 = time.time()
        save_dir = image_root / f"{cat}_{t['task_id']}"
        trace = run_one_trajectory(
            processor, model, t,
            max_steps=args.max_steps,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            force_close_think_after=args.force_close_think_after,
            save_image_dir=save_dir,
        )
        trace["elapsed_s"] = round(time.time() - t0, 2)
        results.append(trace)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info(
            "[%d/%d] %s/%s status=%s steps=%d final_passed=%s elapsed=%.1fs",
            len(results), len(results) + len(tasks) - i - 1,
            cat, t["task_id"], trace["status"], trace["n_steps"],
            trace.get("final_passed"), trace["elapsed_s"],
        )

    logger.info("Done. %d traces -> %s", len(results), out_path)


if __name__ == "__main__":
    main()
