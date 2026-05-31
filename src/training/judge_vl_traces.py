"""LLM-as-judge filter for VL teacher traces (Gemini 3 Flash via OpenRouter).

Reads traces produced by `collect_vl_traces.py` and asks the judge three
binary questions per trace:

  review_specific         -- does every post-feedback turn cite concrete
                             observations from the tool return (not vague
                             "looks good")?
  revision_addresses_review -- do revisions target the defects called out in
                             the immediately preceding review?
  final_meets_spec        -- does the final artifact satisfy the task spec?

For webpages/slides the judge is given the actual rendered screenshot of the
FINAL artifact (re-rendered locally, not the teacher's description of it).
That makes `final_meets_spec` a real vision check, not hearsay. For code,
the trace already carries deterministic pass/fail info from the harness.

A trace is kept iff all three are true.

Usage:
    OPENROUTER_API_KEY=... python -m src.training.judge_vl_traces \\
        --traces data/training/vl_teacher_pilot.json \\
        --output data/training/vl_teacher_pilot_judged.json
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

from src.training.collect_vl_traces import (
    extract_artifact, _render_one,
)

logger = logging.getLogger(__name__)

JUDGE_MODEL = "google/gemini-3-flash-preview"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 180.0

JUDGE_SYSTEM = """You are evaluating whether a teacher model's trace demonstrates GENUINE self-review behavior — critical, specific, grounded in observed feedback — vs shallow thrashing or vague praise.

You receive:
- Task spec (description + requirements)
- Ordered assistant messages (teacher's critiques + revised artifacts)
- Execution/render feedback summaries between assistant turns
- For webpages/slides: the ACTUAL rendered screenshot(s) of the teacher's FINAL artifact, so you can verify the teacher's "it looks good" claim directly with your own vision.

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
- `review_specific`: true ONLY if at least one post-feedback turn cites concrete observations (variable names, line numbers, exception text for code; colors, positions, element names, overlaps for visual). "The output is wrong" is not specific.
- `revision_addresses_review`: true ONLY if each revision targets the previously identified defect. If the teacher oscillates or re-emits identical code, this is false.
- `final_meets_spec`: for CODE, true iff final_passed=true in the trace metadata. For WEBPAGES/SLIDES, YOU MUST LOOK AT THE FINAL SCREENSHOT and check whether it actually satisfies the task spec. If the teacher claimed "no issues" but you can see overlapping elements / missing sections / broken layout / unreadable text in the screenshot, set this to FALSE and explain what you see that the teacher missed.
- `keep = review_specific AND revision_addresses_review AND final_meets_spec`."""


def summarize_feedback(fb: dict | None) -> str:
    if not fb:
        return "(no feedback yet — first attempt)"
    if fb.get("type") == "text":
        return f"EXECUTION: passed={fb.get('passed')} :: {(fb.get('text_payload') or '')[:600]}"
    return f"RENDER: {fb.get('image_count', 0)} screenshot(s) shown to the teacher"


def format_trace_for_judge(trace: dict, task: dict) -> str:
    parts = [
        f"CATEGORY: {trace['category']}",
        f"TASK_ID: {trace['task_id']}",
        f"SPEC:\n{task.get('description', '')[:2000]}",
    ]
    reqs = task.get("requirements")
    if reqs:
        parts.append("REQUIREMENTS:\n" + "\n".join(f"- {r}" for r in reqs))
    parts.append("")
    for i, turn in enumerate(trace.get("turns", [])):
        fb_summary = summarize_feedback(turn.get("feedback"))
        parts.append(f"--- TURN {i} ---")
        parts.append(f"FEEDBACK_TO_TEACHER: {fb_summary}")
        content = (turn.get("assistant") or "")
        # keep critique prose, trim code blocks to their first 8 lines
        content = re.sub(
            r"```(python|html)\n(.*?)```",
            lambda m: "```" + m.group(1) + "\n" + "\n".join(
                m.group(2).splitlines()[:8]) + "\n... [code truncated]\n```",
            content, flags=re.DOTALL,
        )
        parts.append(f"TEACHER_SAID:\n{content[:3000]}")
        parts.append("")
    parts.append(f"FINAL_STATUS: {trace.get('status')} "
                 f"final_passed={trace.get('final_passed')}")
    return "\n".join(parts)


def _png_to_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def render_final_images(trace: dict, task: dict) -> list[tuple[int, int, bytes]]:
    """Re-render the trace's final artifact so the judge can see it directly.

    Returns list of (width, height, png_bytes). Empty for code or when
    no artifact is extractable.
    """
    category = trace["category"]
    if category == "code":
        return []
    # Last artifact across all turns (final turn may be critique-only)
    art = None
    for turn in reversed(trace.get("turns", [])):
        a = extract_artifact(turn.get("assistant") or "", category)
        if a:
            art = a
            break
    if art is None:
        return []
    viewports = task.get("viewport_sizes", [1920]) if category == "webpages" else [1920]
    imgs = []
    for w in viewports:
        h = 1080 if w >= 1024 else (1024 if w >= 700 else 812)
        try:
            png = _render_one(art, w, h, full_page=(category == "webpages"))
            imgs.append((w, h, png))
        except Exception as e:
            logger.warning("judge-render failed at %dx%d: %s", w, h, e)
    return imgs


def call_judge(prompt: str, images: list[tuple[int, int, bytes]]) -> dict | None:
    if images:
        content = [{"type": "text", "text": prompt},
                   {"type": "text",
                    "text": "\n\nBelow: the actual rendered screenshot(s) of the teacher's FINAL artifact. Inspect these yourself to decide final_meets_spec."}]
        for w, h, png in images:
            content.append({"type": "text", "text": f"viewport {w}x{h}:"})
            content.append({"type": "image_url", "image_url": {"url": _png_to_url(png)}})
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


def load_task_by_id(category: str, task_id: str) -> dict | None:
    paths_by_cat = {
        "code": [
            "data/hard_benchmarks/code/code_tasks.json",
            "data/hard_benchmarks/code/code_tasks_heldout.json",
            "data/hard_benchmarks/code/code_tasks_heldout_v2.json",
        ],
        "webpages": [
            "data/hard_benchmarks/webpages/webpage_tasks.json",
            "data/training/vl_webpage_tasks_gen.json",
        ],
        "slides": [
            "data/hard_benchmarks/slides/slide_tasks.json",
            "data/training/vl_slide_tasks_gen.json",
        ],
    }
    for p in paths_by_cat.get(category, []):
        if Path(p).exists():
            for t in json.loads(Path(p).read_text()):
                if t["task_id"] == task_id:
                    return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", default="logs/judge_vl_traces.log")
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
    judged = []
    kept = 0
    for trace in traces:
        task = load_task_by_id(trace["category"], trace["task_id"])
        if task is None:
            logger.warning("skipping %s: task not found", trace["task_id"])
            continue
        prompt = format_trace_for_judge(trace, task)
        images = render_final_images(trace, task)
        verdict = call_judge(prompt, images)
        if verdict is None:
            verdict = {"keep": False, "error": "judge_failed"}
        trace["judge"] = verdict
        judged.append(trace)
        if verdict.get("keep"):
            kept += 1
        logger.info(
            "%s/%s keep=%s spec=%s addr=%s final=%s  why_final=%r",
            trace["category"], trace["task_id"],
            verdict.get("keep"),
            verdict.get("review_specific"),
            verdict.get("revision_addresses_review"),
            verdict.get("final_meets_spec"),
            (verdict.get("final_meets_spec_why") or "")[:120],
        )

    Path(args.output).write_text(json.dumps(judged, indent=2, default=str))
    logger.info("Judged %d traces, kept %d -> %s", len(judged), kept, args.output)


if __name__ == "__main__":
    main()
