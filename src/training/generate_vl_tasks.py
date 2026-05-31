"""Generate fresh webpage and slide tasks via Qwen3-235B (OpenRouter).

Produces task specs in the same shape as
`data/hard_benchmarks/webpages/webpage_tasks.json` and
`data/hard_benchmarks/slides/slide_tasks.json`.

These task specs are *spec-only* (no buggy_code / fixed_code as in the code
generator). They will be rendered into teacher traces by collect_vl_traces.

Usage:
    OPENROUTER_API_KEY=... python -m src.training.generate_vl_tasks \\
        --target-webpages 25 --target-slides 25 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MODEL = "qwen/qwen3-235b-a22b-2507"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 300.0
MAX_TOKENS = 4000
TEMPERATURE = 0.8

WEBPAGE_THEMES = [
    "SaaS analytics dashboard with charts and metrics",
    "documentation site with sidebar TOC and code examples",
    "personal blog with author bio and article cards",
    "marketing landing page with hero, features, pricing",
    "e-commerce product detail page with gallery + specs",
    "news magazine multi-column layout",
    "data table viewer with filters and sortable columns",
    "team directory grid with avatars and bios",
    "FAQ accordion page with categories",
    "portfolio site with project tiles and case studies",
    "settings page with form sections and toggles",
    "comparison table for product tiers",
    "stats/timeline page tracking project history",
    "academic course catalog page",
    "events calendar with month/week views",
]

SLIDE_THEMES = [
    "explain a CS algorithm with diagram + pseudocode + complexity",
    "compare ML model architectures in a table",
    "research methodology overview with phases + deliverables",
    "system architecture diagram with components and arrows",
    "math derivation with aligned equations",
    "results bar chart with annotations",
    "concept map with nodes and labeled edges",
    "before/after comparison with two columns",
    "process flowchart with 5-7 steps",
    "scientific image + 3-paragraph caption",
    "API reference card for a single endpoint",
    "design system swatches grid (colors + typography)",
    "pricing or feature matrix",
    "cited references slide with bibliography formatting",
    "geographic data map with legend",
]

WEBPAGE_SYSTEM = """You are generating evaluation tasks for an HTML/CSS coding agent. Each task asks the agent to build a self-contained HTML+CSS page satisfying a detailed spec.

Reply with ONE JSON object, no prose, no markdown fences:

{
  "task_id": "web_genXXX",
  "description": "<2-4 paragraphs describing the page concept and visible content>",
  "requirements": ["<8-12 bullet-style requirements covering layout, breakpoints, colors, sizes, behavior>"],
  "expected_issues": ["<5-7 bullet-style common mistakes a junior engineer would make on this page>"],
  "viewport_sizes": [1920, 768, 375],
  "difficulty": "hard"
}

Rules:
1. Description must be concrete: name actual headings, content categories, color hexes, typography sizes.
2. Requirements must be testable visually (e.g. "3 columns at 1920px, 1 column at 375px", "hero gradient #1a1a4e to #6b21a8", "minimum 16px body text").
3. Include responsive breakpoints — at least one for desktop and one for mobile.
4. Include at least one numeric color (hex), one numeric size (px), and one positional requirement (top-left, centered, etc.).
5. The task must be doable in pure inline HTML+CSS+JS (no external assets).
6. Avoid duplicating existing tasks. Pick a NEW theme/concept.

Return ONLY the JSON object."""

SLIDE_SYSTEM = """You are generating evaluation tasks for an HTML/CSS coding agent producing a single 1920x1080 slide.

Reply with ONE JSON object, no prose, no markdown fences:

{
  "task_id": "slide_genXXX",
  "description": "<2-3 paragraphs describing the slide content: title, body elements, visual structure>",
  "requirements": ["<6-10 testable bullets: layout, sizes, colors, positions, visible elements>"],
  "expected_issues": ["<5-7 common mistakes a junior designer would make>"],
  "difficulty": "hard"
}

Rules:
1. Slide must fit 1920x1080 with NO scrollbar.
2. Description must enumerate concrete content: actual labels, equations, table headers, etc.
3. Requirements must be testable visually (sizes in px, alignment, spacing, color hexes).
4. Use only HTML+CSS+SVG inline; no external assets.
5. Avoid duplicating existing slide tasks. Pick a NEW theme.
6. Expected_issues should describe REAL common mistakes (overlap, fits-too-small, wrong superscript rendering, etc).

Return ONLY the JSON object."""


def _strip_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text


def _extract_json_object(text: str) -> str:
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return text[start:]


def call_qwen(category: str, theme: str, exemplar: dict | None) -> dict | None:
    system = WEBPAGE_SYSTEM if category == "webpages" else SLIDE_SYSTEM
    exemplar_block = ""
    if exemplar:
        ex_desc = exemplar.get("description", "")[:1200]
        exemplar_block = f"\n\nOne exemplar in the same domain (DO NOT copy — produce a different concept):\n{ex_desc}"

    user_text = (
        f"Generate ONE task on the theme: {theme}.{exemplar_block}\n\n"
        f"Return exactly one JSON object following the schema."
    )
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    }
    try:
        resp = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("API call failed for %s/%s: %s", category, theme, e)
        return None
    text = (data["choices"][0]["message"].get("content") or "").strip()
    obj_text = _extract_json_object(text)
    try:
        return json.loads(obj_text)
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed: %s (head: %r)", e, obj_text[:200])
        return None


def validate(task: dict, category: str) -> tuple[bool, str]:
    required = {"task_id", "description", "requirements", "expected_issues", "difficulty"}
    if category == "webpages":
        required.add("viewport_sizes")
    missing = required - set(task.keys())
    if missing:
        return False, f"missing fields: {missing}"
    if len(task["description"]) < 200:
        return False, "description too short"
    if not isinstance(task["requirements"], list) or len(task["requirements"]) < 5:
        return False, "requirements must be a list of >=5 items"
    if not isinstance(task["expected_issues"], list) or len(task["expected_issues"]) < 3:
        return False, "expected_issues must be a list of >=3 items"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-webpages", type=int, default=25)
    ap.add_argument("--target-slides", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-attempts-per-cat", type=int, default=80)
    ap.add_argument("--out-webpages", default="data/training/vl_webpage_tasks_gen.json")
    ap.add_argument("--out-slides", default="data/training/vl_slide_tasks_gen.json")
    ap.add_argument("--log", default="logs/generate_vl_tasks.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set"); sys.exit(1)

    # Load exemplars (existing tasks)
    web_existing = json.loads(Path("data/hard_benchmarks/webpages/webpage_tasks.json").read_text())
    slide_existing = json.loads(Path("data/hard_benchmarks/slides/slide_tasks.json").read_text())

    def collect(category: str, target: int, themes: list[str], exemplars: list[dict],
                out_path: str):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        collected = json.loads(out.read_text()) if out.exists() else []
        if collected:
            logger.info("[%s] resuming with %d existing tasks", category, len(collected))
        existing_descs = {t["description"][:200] for t in collected}
        existing_descs |= {t["description"][:200] for t in exemplars}
        attempts = [0]
        next_id = [len(collected) + 1]
        lock = threading.Lock()

        def submit_one():
            with lock:
                if len(collected) >= target or attempts[0] >= args.max_attempts_per_cat:
                    return
                attempts[0] += 1
                theme = random.choice(themes)
                exemplar = random.choice(exemplars) if exemplars else None
            task = call_qwen(category, theme, exemplar)
            if task is None:
                return
            ok, reason = validate(task, category)
            if not ok:
                logger.info("[%s] REJECT: %s", category, reason)
                return
            sig = task["description"][:200]
            with lock:
                if sig in existing_descs:
                    logger.info("[%s] REJECT: duplicate description", category)
                    return
                prefix = "web_gen" if category == "webpages" else "slide_gen"
                task["task_id"] = f"{prefix}_{next_id[0]:03d}"
                next_id[0] += 1
                collected.append(task)
                existing_descs.add(sig)
                out.write_text(json.dumps(collected, indent=2))
                logger.info("[%s] ACCEPT [%d/%d] %s attempts=%d",
                            category, len(collected), target,
                            task["task_id"], attempts[0])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            while True:
                with lock:
                    if (len(collected) >= target or
                        attempts[0] >= args.max_attempts_per_cat):
                        break
                    in_flight = sum(1 for f in futures if not f.done())
                    capacity = args.workers * 2 - in_flight
                if capacity <= 0:
                    import time; time.sleep(0.5); continue
                for _ in range(capacity):
                    futures.append(ex.submit(submit_one))
                for f in list(futures):
                    if f.done():
                        futures.remove(f)
                import time; time.sleep(0.2)
            for f in as_completed(futures):
                pass
        logger.info("[%s] Done: %d/%d in %d attempts",
                    category, len(collected), target, attempts[0])

    collect("webpages", args.target_webpages, WEBPAGE_THEMES, web_existing,
            args.out_webpages)
    collect("slides", args.target_slides, SLIDE_THEMES, slide_existing,
            args.out_slides)


if __name__ == "__main__":
    main()
