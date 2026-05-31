"""Teacher trace collection for VL self-review thesis (code / webpages / slides).

Uses Qwen3-VL-235B-A22B-Thinking via OpenRouter as the teacher. The system
prompt combines (a) the generation instructions for the category and (b) a
specific review rubric -- the teacher is told to GENERATE, then REVIEW the
rendered artifact against the rubric, then REVISE until the review is clean,
then emit <final>...</final>.

For webpages / slides the harness renders HTML with Playwright and feeds the
screenshot back as a user-turn image. For code the harness executes the Python
block and feeds back stdout/stderr.

Pilot run -- 5 tasks per category -- should be inspected manually before
scaling to the full set, to verify traces demonstrate real self-review (not
going through the motions).

Usage:
    OPENROUTER_API_KEY=... python -m src.training.collect_vl_traces \\
        --pilot --workers 3
"""

from __future__ import annotations

import argparse
import base64
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

from src.evaluation.code_eval import CodeEvaluator

logger = logging.getLogger(__name__)

TEACHER = "qwen/qwen3-vl-30b-a3b-thinking"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 600.0
MAX_TOKENS = 12000

# --- combined generation + review system prompts ---------------------------

_TURN_STRUCTURE = """\
Your FIRST assistant message contains only the initial ```<LANG> block (no prose before or after).

Every SUBSEQUENT assistant message MUST have EXACTLY this structure and nothing else:

  <CRITIQUE paragraph: {critique_rule}>

  ```<LANG>
  <revised full artifact>
  ```

OR, only if you are fully satisfied with the LAST rendered result:

  <CRITIQUE paragraph describing what you verified in the last feedback>

  <final>OK</final>

Do not emit a message that contains ONLY a critique without either a revised code block OR <final>OK</final>. That is incomplete. Either revise or declare done — never end a turn mid-loop."""

CODE_SYSTEM = ("""You are a senior Python engineer debugging code. You iterate: GENERATE -> REVIEW execution output -> REVISE. Never re-emit an identical function.

""" + _TURN_STRUCTURE.replace("<LANG>", "python").replace(
    "{critique_rule}",
    "2-4 sentences referencing the actual execution output received. Quote the specific AssertionError / exception line and diagnose the real bug. If the test passed, quote which assertion the fix resolved and why. Never write 'let me try again' without a concrete diagnosis.",
))

WEBPAGE_SYSTEM = ("""You are a senior frontend engineer. You iterate: GENERATE self-contained HTML -> REVIEW rendered screenshots -> REVISE. Use inline <style>/<script> only; no external resources. Never regress fixes you just made.

Review axes you must check on every rendered screenshot: element presence, layout (columns/alignment/spacing), colors and contrast, responsive behavior across viewports, overflow / clipping / overlap.

""" + _TURN_STRUCTURE.replace("<LANG>", "html").replace(
    "{critique_rule}",
    "3-6 sentences citing specific visual observations (colors you see, element positions, overlaps, missing elements, what differs between the 1920 and 375 viewports). Use concrete selectors or coordinates. Generic praise like 'looks good' is forbidden — if you truly see no issue, describe what you verified for each axis.",
))

SLIDE_SYSTEM = ("""You are a senior designer producing a single 1920x1080 HTML/CSS slide. You iterate: GENERATE -> REVIEW screenshot at 1920x1080 -> REVISE. Inline <style> only; no external resources.

Review axes on every screenshot: fit (no scrollbars), component presence, non-overlap (arrows vs text, boxes vs boxes, labels vs containers), typography (min 12px, super/subscript rendered via CSS), composition.

""" + _TURN_STRUCTURE.replace("<LANG>", "html").replace(
    "{critique_rule}",
    "3-6 sentences citing what you literally see in the screenshot (which boxes are placed where, which labels are readable, which arrows cross text). Generic praise forbidden.",
))

SYSTEM_PROMPTS = {
    "code": CODE_SYSTEM,
    "webpages": WEBPAGE_SYSTEM,
    "slides": SLIDE_SYSTEM,
}

# --- artifact extraction ---------------------------------------------------

PY_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
HTML_RE = re.compile(r"```html\s*\n(.*?)```", re.DOTALL)
FINAL_RE = re.compile(r"<final>\s*OK\s*</final>", re.IGNORECASE)


def extract_artifact(text: str, category: str) -> str | None:
    rx = PY_RE if category == "code" else HTML_RE
    matches = rx.findall(text)
    return matches[-1].strip() if matches else None


# --- per-category feedback harnesses ---------------------------------------


def run_code(code: str, test_code: str, evaluator: CodeEvaluator) -> dict:
    res = evaluator.evaluate(code, test_code, timeout=10)
    return {
        "type": "text",
        "passed": res.passed,
        "payload": (
            f"execution result:\npassed={res.passed}\n"
            f"stdout:\n{(res.stdout or '')[:1500]}\n"
            f"stderr:\n{(res.stderr or '')[:1500]}\n"
            f"error: {res.error_message or 'None'}"
        ),
    }


def _png_to_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _render_one(html: str, w: int, h: int, full_page: bool = False) -> bytes:
    """Thread-safe: launch+close a fresh Playwright per call (~1s overhead)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.set_content(html, wait_until="networkidle")
            return page.screenshot(type="png", full_page=full_page)
        finally:
            browser.close()


def run_webpage(html: str, viewports: list[int]) -> dict:
    images = []
    for w in viewports:
        h = 1080 if w >= 1024 else (1024 if w >= 700 else 812)
        try:
            # full_page=True so teacher sees the entire scrolled page (hero + features + pricing + footer)
            png = _render_one(html, w, h, full_page=True)
            images.append((w, h, png))
        except Exception as e:
            logger.warning("render failed at %dx%d: %s", w, h, e)
    return {"type": "image", "payload": images}


def run_slide(html: str) -> dict:
    try:
        png = _render_one(html, 1920, 1080, full_page=False)
        return {"type": "image", "payload": [(1920, 1080, png)]}
    except Exception as e:
        return {"type": "image", "payload": [], "error": str(e)}


def build_feedback_user_turn(fb: dict) -> dict:
    if fb["type"] == "text":
        return {"role": "user", "content": fb["payload"]}
    parts = [{"type": "text", "text": "rendered screenshot(s):"}]
    for w, h, png in fb["payload"]:
        parts.append({"type": "text", "text": f"viewport {w}x{h}:"})
        parts.append({"type": "image_url", "image_url": {"url": _png_to_url(png)}})
    if not fb["payload"]:
        parts = [{"type": "text", "text": f"render failed: {fb.get('error','')}"}]
    return {"role": "user", "content": parts}


# --- OpenRouter call -------------------------------------------------------


def call_teacher(messages: list[dict]) -> tuple[str, str]:
    """Return (content, reasoning). reasoning is the <think> block."""
    payload = {
        "model": TEACHER,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    resp = httpx.post(
        OR_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", msg.get("reasoning") or ""


# --- task loading ----------------------------------------------------------


def load_tasks(category: str) -> list[dict]:
    paths = {
        "code": ["data/hard_benchmarks/code/code_tasks.json"],
        "webpages": ["data/hard_benchmarks/webpages/webpage_tasks.json",
                     "data/training/vl_webpage_tasks_gen.json"],
        "slides": ["data/hard_benchmarks/slides/slide_tasks.json",
                   "data/training/vl_slide_tasks_gen.json"],
    }
    out: list[dict] = []
    for p in paths[category]:
        if Path(p).exists():
            out.extend(json.loads(Path(p).read_text()))
    return out


def build_user_turn(task: dict, category: str) -> str:
    if category == "code":
        return (
            f"{task['description']}\n\n"
            f"Your fix will be validated against this test harness:\n"
            f"```python\n{task['test_code']}\n```"
        )
    reqs = "\n".join(f"- {r}" for r in task.get("requirements", []))
    if category == "webpages":
        vps = task.get("viewport_sizes", [1920, 375])
        vps_str = ", ".join(f"{w}px" for w in vps)
        return (
            f"{task['description']}\n\nRequirements:\n{reqs}\n\n"
            f"You will see the page rendered at these viewport widths: {vps_str}.\n"
            f"Review each screenshot against the requirements and revise until the page is clean."
        )
    return (
        f"{task['description']}\n\nRequirements:\n{reqs}\n\n"
        f"You will see the slide rendered at 1920x1080. Review and revise."
    )


# --- trace collection ------------------------------------------------------


def collect_one(task: dict, category: str, max_turns: int,
                evaluator: CodeEvaluator) -> dict:
    system = SYSTEM_PROMPTS[category]
    user0 = build_user_turn(task, category)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user0},
    ]
    turns = []
    last_feedback = None
    for turn_idx in range(max_turns):
        try:
            content, reasoning = call_teacher(messages)
        except Exception as e:
            logger.warning("teacher call failed on %s turn %d: %s",
                           task["task_id"], turn_idx, e)
            return {
                "task_id": task["task_id"], "category": category,
                "status": "error", "error": str(e), "turns": turns,
            }
        turns.append({"assistant": content, "reasoning": reasoning,
                      "feedback": last_feedback})
        messages.append({"role": "assistant", "content": content})

        if FINAL_RE.search(content):
            # Last artifact could be in this turn OR a previous one
            # (model may emit <final>OK</final> in a critique-only turn when
            # the previous render/exec already passed).
            art = None
            for past in reversed(turns):
                a = extract_artifact(past.get("assistant") or "", category)
                if a:
                    art = a
                    break
            if art and category == "code":
                fb = run_code(art, task["test_code"], evaluator)
                final_passed = fb["passed"]
            else:
                final_passed = None  # visual or missing artifact
            return {
                "task_id": task["task_id"], "category": category,
                "status": "final", "final_artifact": art,
                "final_passed": final_passed, "turns": turns,
            }

        art = extract_artifact(content, category)
        if art is None:
            return {
                "task_id": task["task_id"], "category": category,
                "status": "no_artifact", "turns": turns,
            }

        if category == "code":
            fb = run_code(art, task["test_code"], evaluator)
        elif category == "webpages":
            fb = run_webpage(art, task.get("viewport_sizes", [1920, 375]))
        else:
            fb = run_slide(art)

        user_turn = build_feedback_user_turn(fb)
        messages.append(user_turn)
        last_feedback = {
            "type": fb["type"],
            "passed": fb.get("passed"),
            "text_payload": fb["payload"] if fb["type"] == "text" else None,
            "image_count": len(fb["payload"]) if fb["type"] == "image" else 0,
        }

    return {
        "task_id": task["task_id"], "category": category,
        "status": "max_turns", "turns": turns,
    }


# --- main ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="5 tasks per category instead of full sets")
    ap.add_argument("--categories", nargs="+",
                    default=["code", "webpages", "slides"])
    ap.add_argument("--n-per-category", type=int, default=0,
                    help="override: N tasks per category (0 = all or pilot)")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", default="data/training/vl_teacher_traces.json")
    ap.add_argument("--log", default="logs/collect_vl_traces.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set"); sys.exit(1)

    evaluator = CodeEvaluator()

    all_jobs = []
    for cat in args.categories:
        tasks = load_tasks(cat)
        n = args.n_per_category or (5 if args.pilot else len(tasks))
        for t in tasks[:n]:
            all_jobs.append((cat, t))
    logger.info("Queued %d trace jobs across %s", len(all_jobs), args.categories)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    if out_path.exists():
        traces = json.loads(out_path.read_text())
        done_ids = {(t["category"], t["task_id"]) for t in traces}
        all_jobs = [(c, t) for c, t in all_jobs if (c, t["task_id"]) not in done_ids]
        logger.info("Resuming. %d already done; %d remaining.", len(traces), len(all_jobs))

    lock = threading.Lock()
    total = len(traces) + len(all_jobs)

    def worker(cat: str, task: dict):
        t0 = time.time()
        trace = collect_one(task, cat, args.max_turns, evaluator)
        trace["elapsed_s"] = round(time.time() - t0, 2)
        with lock:
            traces.append(trace)
            out_path.write_text(json.dumps(traces, indent=2, default=str))
            logger.info(
                "[%d/%d] %s/%s status=%s turns=%d elapsed=%.1fs",
                len(traces), total,
                cat, task["task_id"], trace["status"],
                len(trace.get("turns", [])), trace["elapsed_s"],
            )
        return trace

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, c, t) for c, t in all_jobs]
        for _ in as_completed(futures):
            pass

    logger.info("Done. %d traces -> %s", len(traces), out_path)


if __name__ == "__main__":
    main()
