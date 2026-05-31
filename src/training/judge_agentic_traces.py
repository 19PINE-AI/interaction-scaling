"""LLM-as-judge filter for AGENTIC teacher traces (Gemini 3 Flash via OpenRouter).

Adapted from `judge_vl_traces.py` for the agentic-tools trace format produced
by `collect_agentic_traces.py` (steps with write_file/bash/read_file).

For each trace the judge gets:
  - Task spec
  - Per-step summary: what the agent said + which tool it called
  - For visual tasks: the final saved screenshot (the last image the agent saw
    via read_file). This is the agent's actual last view — if that screenshot
    is broken, the agent's "looks good" claim is wrong.
  - For code tasks: deterministic `final_passed` from the trace metadata
    (the test was run against the workspace's solution.py at trace end).

Three dimensions:
  review_specific          — post-screenshot critiques cite concrete defects
  revision_addresses_review — subsequent write_file/bash actions target those defects
  final_meets_spec         — final artifact satisfies task spec (vision check for visual)

A trace is `keep=true` iff all three are true.

Usage:
    OPENROUTER_API_KEY=... python -m src.training.judge_agentic_traces \\
        --traces data/training/agentic_traces.json \\
        --output data/training/agentic_traces_judged.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx

from src.training.judge_vl_traces import load_task_by_id

logger = logging.getLogger(__name__)

JUDGE_MODEL = "google/gemini-3-flash-preview"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 180.0

JUDGE_SYSTEM = """You are evaluating an AGENTIC engineer's trace where the agent solved a task using three tools: write_file, read_file, bash. The agent renders artifacts to PNG with bash and views them with read_file.

You receive:
- Task spec (description + requirements)
- Per-step summary: what the agent said in each step, and which tools it called
- For visual tasks (webpages/slides): the FINAL screenshot the agent saw (the last read_file image), so you can independently verify the agent's claim that the artifact meets spec.
- For code tasks: the trace's deterministic final_passed flag (whether the actual test suite passed on the final solution.py).

Return ONE JSON object with exactly these fields (no prose outside the JSON):

{
  "review_specific": <bool>,
  "review_specific_why": "<one sentence>",
  "revision_addresses_review": <bool>,
  "revision_addresses_review_why": "<one sentence>",
  "final_meets_spec": <bool>,
  "final_meets_spec_why": "<one sentence>",
  "keep": <bool>
}

Rules:
- `review_specific`: true iff at least one post-render assistant turn cites concrete observations from the screenshot (specific colors, positions, element names, overlaps, missing parts) OR for code, cites concrete error text from the bash output. "The output looks good" without details is NOT specific. The first turn (before any feedback) does not count.
- `revision_addresses_review`: true iff the agent's subsequent write_file/bash calls clearly target the cited defects (not random thrashing, not identical re-emits). If only one render+critique cycle happened, judge whether that critique was used to revise.
- `final_meets_spec`: for CODE, true iff trace.final_passed == true. For WEBPAGES/SLIDES, LOOK AT THE FINAL SCREENSHOT and decide whether it satisfies the task spec. If layout is broken (overlaps, missing sections, unreadable text, wrong proportions), set FALSE and explain.
- `keep = review_specific AND revision_addresses_review AND final_meets_spec`."""


def format_trace_for_judge(trace: dict, task: dict) -> str:
    parts = [
        f"CATEGORY: {trace.get('category')}",
        f"TASK_ID: {trace.get('task_id')}",
        f"SPEC:\n{task.get('description', '')[:2000]}",
    ]
    reqs = task.get("requirements")
    if reqs:
        parts.append("REQUIREMENTS:\n" + "\n".join(f"- {r}" for r in reqs))
    parts.append("")
    for step in trace.get("steps", []):
        idx = step.get("step")
        content = (step.get("assistant_content") or "")[:1500]
        parts.append(f"--- STEP {idx} ---")
        parts.append(f"AGENT_SAID:\n{content}")
        tcs = step.get("tool_calls", []) or []
        tool_summary = []
        for tc in tcs:
            fn = tc.get("function") or tc
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
            except Exception:
                args = {}
            if name == "write_file":
                content_len = len(args.get("content", ""))
                tool_summary.append(f"  write_file path={args.get('path')!r} bytes={content_len}")
            elif name == "bash":
                cmd = args.get("command", "")[:200]
                tool_summary.append(f"  bash {cmd!r}")
            elif name == "read_file":
                tool_summary.append(f"  read_file path={args.get('path')!r}")
            else:
                tool_summary.append(f"  {name} {str(args)[:150]}")
        if tool_summary:
            parts.append("TOOL_CALLS:\n" + "\n".join(tool_summary))
        # Tool result summary (compact)
        for tr in step.get("tool_results", []) or []:
            if isinstance(tr, dict):
                k = tr.get("kind")
                ok = tr.get("ok")
                summ = (tr.get("summary") or "")[:300]
                parts.append(f"TOOL_RESULT: ok={ok} kind={k} -> {summ}")
        parts.append("")
    parts.append(f"FINAL_STATUS: {trace.get('status')} "
                 f"final_passed={trace.get('final_passed')} "
                 f"n_steps={trace.get('n_steps')}")
    return "\n".join(parts)


def _png_to_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def get_final_image(trace: dict) -> bytes | None:
    """Return raw PNG bytes for the LAST saved screenshot (what the agent saw
    most recently). For code tasks, returns None.
    """
    if trace.get("category") == "code":
        return None
    saved = trace.get("saved_images", [])
    if not saved:
        return None
    last = Path(saved[-1])
    if not last.exists():
        return None
    return last.read_bytes()


def call_judge(prompt: str, image_png: bytes | None) -> dict | None:
    if image_png is not None:
        content = [
            {"type": "text", "text": prompt},
            {"type": "text",
             "text": "\n\nBelow: the FINAL screenshot the agent saw (the most "
                     "recent read_file result). Inspect it directly to decide "
                     "final_meets_spec."},
            {"type": "image_url", "image_url": {"url": _png_to_url(image_png)}},
        ]
    else:
        content = prompt

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }
    try:
        r = httpx.post(
            OR_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload, timeout=TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.warning("judge call failed: %s", e)
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", default="logs/judge_agentic_traces.log")
    ap.add_argument("--resume", action="store_true",
                    help="If output exists, skip already-judged trace ids")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set"); sys.exit(1)

    traces = json.loads(Path(args.traces).read_text())
    out_path = Path(args.output)
    judged: list[dict] = []
    done_keys: set = set()
    if args.resume and out_path.exists():
        judged = json.loads(out_path.read_text())
        done_keys = {(t["category"], t["task_id"]) for t in judged}
        logger.info("Resuming with %d already-judged traces", len(judged))

    for trace in traces:
        key = (trace.get("category"), trace.get("task_id"))
        if key in done_keys:
            continue
        if "messages" not in trace:
            # api_error or otherwise structurally incomplete; mark and skip
            trace["judge"] = {"keep": False, "error": "no_messages"}
            judged.append(trace)
            continue

        task = load_task_by_id(trace["category"], trace["task_id"])
        if task is None:
            logger.warning("skipping %s/%s: task spec not found",
                           trace["category"], trace["task_id"])
            trace["judge"] = {"keep": False, "error": "task_not_found"}
            judged.append(trace)
            continue

        # Code: deterministic outcome already known. We still ask the judge
        # to assess review_specific + revision_addresses_review.
        prompt = format_trace_for_judge(trace, task)
        image_png = get_final_image(trace)
        verdict = call_judge(prompt, image_png)
        if verdict is None:
            verdict = {"keep": False, "error": "judge_failed"}

        # For code, outcome is deterministic. Trust final_passed and keep
        # iff the test passed. One-shot solves (no review/revise cycle) are
        # still valuable SFT signal — they teach efficient solve + early
        # termination — so don't penalize them for review_specific=False.
        if trace.get("category") == "code":
            fp = bool(trace.get("final_passed"))
            verdict["final_meets_spec"] = fp
            verdict["keep"] = fp

        trace["judge"] = verdict
        judged.append(trace)

        # Incremental save
        out_path.write_text(json.dumps(judged, indent=2, default=str))

        logger.info(
            "%s/%s keep=%s spec=%s addr=%s final=%s  why_final=%r",
            trace["category"], trace["task_id"],
            verdict.get("keep"),
            verdict.get("review_specific"),
            verdict.get("revision_addresses_review"),
            verdict.get("final_meets_spec"),
            (verdict.get("final_meets_spec_why") or "")[:120],
        )

    n_kept = sum(1 for t in judged if (t.get("judge") or {}).get("keep"))
    logger.info("Judged %d traces, kept %d -> %s", len(judged), n_kept, out_path)


if __name__ == "__main__":
    main()
