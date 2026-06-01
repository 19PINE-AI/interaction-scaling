"""Diagram/figure harness with a DETERMINISTIC geometric reward (Type 3b, but
grounded in DOM measurement rather than a VLM's reading of a screenshot).

Motivation: VLM judging of rendered figures is unreliable for exactly the
defects that matter -- it scored 14/15 dense paper figures "perfect" when a
DOM-geometry check finds only 3/15 are actually clean (text collisions,
clipped labels, canvas overflow). So here the environment feedback E is the
exact, reproducible list of geometric defects extracted from the rendered DOM:
which text labels overlap, which are clipped, whether it overflows 1920x1080.
The reviewer signal is that defect list; the proposer revises to remove them.

Reward = number of geometric defects (lower is better); 0 == clean.

single_shot : 1 proposer call, measured, no revision.
reviewed    : propose -> measure defects -> feed exact defects back -> revise,
              up to max_iters, keep the iteration with the FEWEST defects.

Usage:
  python -m scripts.run_geometric_harness --tasks <tasks.json> \
      --out-prefix paperfig_geom --run 1 --condition both --max-iters 3
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path

from src.config import ModelConfig
from src.evaluation.geometric_checker import (
    geometric_defects, web_geometric_defects, animation_geometric_defects)
from src.rendering.browser import BrowserRenderer
from src.utils.code_utils import extract_code
from src.utils.llm_client import get_client
from scripts.run_diagram_benchmark import DIAGRAM_SYSTEM_PROMPT
from src.experiments.hard_benchmark_runner import (
    SLIDE_SYSTEM_PROMPT, WEBPAGE_SYSTEM_PROMPT, ANIMATION_SYSTEM_PROMPT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("geomharness")
logger.setLevel(logging.INFO)


def _feedback(g: dict) -> str:
    lines = []
    for a, b in g.get("overlap_examples", []):
        lines.append(f"- OVERLAP: the text \"{a}\" overlaps the text \"{b}\" "
                     "-- move them apart so their bounding boxes do not intersect.")
    for ex in g.get("overflow_examples", []):
        lines.append(f"- OVERFLOW: the text \"{ex}\" extends outside its container box "
                     "-- enlarge the box or shrink/wrap the text.")
    for ex in g.get("oob_examples", []):
        lines.append(f"- CLIPPED: \"{ex}\" extends beyond the 1920x1080 viewport edge "
                     "-- move it fully inside with >=10px margin.")
    for ex in g.get("misalignment_examples", []):
        lines.append(f"- MISALIGNMENT: {ex} -- make the boxes in that group equal "
                     "size, share top/bottom (or left/right) edges, and use uniform gutters.")
    if g.get("scrollbar"):
        ow, oh = g.get("overflow_px", [0, 0])
        lines.append(f"- SCROLLBAR: the page is {ow}px too wide and {oh}px too tall; "
                     "the whole figure must fit in 1920x1080 with NO scrollbar "
                     "(set body margin:0 and size the canvas to fit).")
    return "\n".join(lines)


def _feedback_web(g: dict) -> str:
    """Width-labeled deterministic feedback for a scrollable responsive page."""
    lines = []
    for wname, gw in g.get("per_width", {}).items():
        if not isinstance(gw, dict) or "error" in gw:
            continue
        px = wname.split("x")[0]
        tag = f"at {px}px width"
        for a, b in gw.get("overlap_examples", []):
            lines.append(f"- OVERLAP ({tag}): \"{a}\" overlaps \"{b}\" -- separate them.")
        for ex in gw.get("overflow_examples", []):
            lines.append(f"- OVERFLOW ({tag}): \"{ex}\" spills outside its container -- contain or wrap it.")
        for ex in gw.get("oob_examples", []):
            lines.append(f"- HORIZONTAL CLIP ({tag}): \"{ex}\" extends past the right edge -- keep all content within {px}px.")
        for ex in gw.get("misalignment_examples", []):
            lines.append(f"- MISALIGNMENT ({tag}): {ex} -- equalize sizes, share edges, use uniform gutters.")
        if gw.get("scrollbar"):
            ow = gw.get("overflow_px", [0, 0])[0]
            lines.append(f"- HORIZONTAL SCROLLBAR ({tag}): the page is {ow}px too wide -- "
                         f"nothing may exceed {px}px width (vertical scrolling is fine).")
    return "\n".join(lines)


def _feedback_anim(g: dict) -> str:
    """Frame-labeled deterministic feedback for an SVG/CSS animation."""
    lines = []
    for t, gf in g.get("per_frame", {}).items():
        if not isinstance(gf, dict) or "error" in gf or not gf.get("n_defects"):
            continue
        tag = f"at t={t}ms"
        for a, b in gf.get("overlap_examples", []):
            lines.append(f"- OVERLAP ({tag}): \"{a}\" overlaps \"{b}\" -- they must not collide during the animation.")
        for ex in gf.get("oob_examples", []):
            lines.append(f"- OFF-SCREEN ({tag}): \"{ex}\" leaves the 1920x1080 viewport -- keep every element fully on-screen throughout the motion.")
        for ex in gf.get("overflow_examples", []):
            lines.append(f"- OVERFLOW ({tag}): \"{ex}\" spills outside its container -- contain it.")
        for ex in gf.get("misalignment_examples", []):
            lines.append(f"- MISALIGNMENT ({tag}): {ex} -- keep the group equal-size and edge-aligned during the animation.")
    return "\n".join(lines)


REVISION_PROMPT = """\
Your previous figure (shown above) was rendered and measured. These EXACT \
geometric defects were detected in the DOM:

{feedback}

Fix ONLY these geometric defects; preserve all correct content and structure. \
Re-emit the COMPLETE updated HTML in a ```html code block."""


def run_task(task, condition, max_iters, proposer, renderer, client,
             system_prompt=DIAGRAM_SYSTEM_PROMPT, include_alignment=False, web=False,
             anim=False):
    client.reset_counters()
    start = time.time()
    if anim:
        ftimes = task.get("frame_times_ms", [0, 1000, 2000, 3000, 4000])
        scorer = lambda h: animation_geometric_defects(
            h, ftimes, renderer=renderer, include_alignment=include_alignment)
        feedbacker = _feedback_anim
    elif web:
        scorer = lambda h: web_geometric_defects(
            h, renderer=renderer, include_alignment=include_alignment)
        feedbacker = _feedback_web
    else:
        scorer = lambda h: geometric_defects(
            h, renderer=renderer, include_alignment=include_alignment)
        feedbacker = _feedback
    prompt = task["description"]
    resp = client.generate(config=proposer, system=system_prompt,
                           messages=[{"role": "user", "content": prompt}])
    html = extract_code(resp.content, "html")
    if not html.strip().startswith("<"):
        html = resp.content

    max_i = 1 if condition == "single_shot" else max_iters
    best = None  # (n_defects, html, g)
    iterations = 1
    for i in range(max_i):
        try:
            g = scorer(html)
        except Exception as e:  # noqa: BLE001
            logger.warning("geom failed %s: %s", task["task_id"], e)
            break
        if "error" in g or "n_defects" not in g:
            logger.warning("geom probe error %s: %s", task["task_id"], g.get("error"))
            break
        nd = g["n_defects"]
        if best is None or nd < best[0]:
            best = (nd, html, g)
        if condition == "single_shot" or nd == 0:
            break
        if i < max_i - 1:
            fb = feedbacker(g)
            if not fb.strip():
                break
            resp = client.generate(config=proposer, system=system_prompt,
                                   messages=[{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": f"```html\n{html}\n```"},
                                             {"role": "user", "content": REVISION_PROMPT.format(feedback=fb)}])
            new = extract_code(resp.content, "html")
            if new.strip():
                html = new
            iterations = i + 2

    failed = best is None  # never produced a measurable render
    nd, html, g = best if best else (None, html, {})
    try:
        png = renderer.render_html(html)
        shot = base64.b64encode(png).decode()
    except Exception:  # noqa: BLE001
        shot = ""
    usage = client.get_usage_summary()
    return {"task_id": task["task_id"],
            "category": task.get("category") or task.get("name", ""),
            "condition": condition, "n_defects": nd, "failed": failed,
            "text_overlap": g.get("text_overlap"), "clipped": g.get("out_of_bounds"),
            "overflow": g.get("overflow"), "scrollbar": g.get("scrollbar"),
            "iterations": iterations, "total_tokens": usage["total_tokens"],
            "wall_time_seconds": time.time() - start, "final_html": html,
            "screenshot_b64": shot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-prefix", default="geom")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--condition", choices=["single_shot", "reviewed", "both"], default="both")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", choices=["diagram", "slide", "web", "animation"], default="diagram",
                    help="system prompt for the proposer")
    ap.add_argument("--include-alignment", action="store_true",
                    help="fold misalignment into the geometric reward and feedback")
    ap.add_argument("--web", action="store_true",
                    help="score as a scrollable responsive page (multi-width, horizontal-overflow only)")
    ap.add_argument("--animation", action="store_true",
                    help="score per-frame geometry across the task's frame_times_ms")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    system_prompt = {"slide": SLIDE_SYSTEM_PROMPT, "web": WEBPAGE_SYSTEM_PROMPT,
                     "animation": ANIMATION_SYSTEM_PROMPT}.get(
        args.prompt, DIAGRAM_SYSTEM_PROMPT)
    proposer = ModelConfig(provider=ModelConfig.claude_sonnet().provider,
                           model_id="claude-sonnet-4-20250514",
                           max_tokens=8192, temperature=args.temperature)
    renderer = BrowserRenderer()
    client = get_client()
    conditions = ["single_shot", "reviewed"] if args.condition == "both" else [args.condition]
    out = Path(f"results/hard_benchmarks/{args.out_prefix}_run{args.run}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        row = {"task_id": task["task_id"],
               "category": task.get("category") or task.get("name", ""), "run": args.run}
        for cond in conditions:
            r = run_task(task, cond, args.max_iters, proposer, renderer, client,
                         system_prompt=system_prompt, include_alignment=args.include_alignment,
                         web=args.web, anim=args.animation)
            pre = "ss" if cond == "single_shot" else "rv"
            for k in ("n_defects", "text_overlap", "clipped", "overflow", "scrollbar",
                      "iterations", "total_tokens", "final_html", "screenshot_b64"):
                row[f"{pre}_{k}"] = r[k]
            logger.info("run%d %s [%s]: defects=%s (ovl=%s clip=%s ovf=%s scr=%s) it=%d",
                        args.run, task["task_id"], cond,
                        "FAILED" if r["failed"] else r["n_defects"], r["text_overlap"],
                        r["clipped"], r["overflow"], r["scrollbar"], r["iterations"])
        records.append(row)
        out.write_text(json.dumps(records, indent=2))
    logger.info("wrote %d records to %s", len(records), out)


if __name__ == "__main__":
    main()
