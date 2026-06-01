"""Diagram / SVG-figure modality (Type 3b), scored rubric-first.

This is the modality motivated by the most self-demonstrating failure of LLM
figure generation: architecture diagrams and flowcharts emitted as SVG come
out with overlapping boxes, labels spilling out of nodes, arrows crossing
through unrelated nodes, and uneven splits. A render-and-look loop (Type 3b)
is exactly the grounded feedback that fixes them.

Unlike the original visual modalities, scoring here is NOT a subjective
holistic 0-1 VLM rating. Each task ships ~8 concrete, objectively-checkable
requirements; the rubric judge (src/evaluation/checklist_judge) evaluates each
INDEPENDENTLY as satisfied/violated and returns structured JSON, and external
code averages n_met/n_total. That same rubric is the reviewer signal: the
proposer is told exactly which requirements are violated (with the judge's
per-requirement evidence) and revises.

Conditions:
  single_shot : 1 proposer call, rendered, rubric-scored, no revision.
  reviewed    : propose -> render -> rubric -> revise, up to max_iters,
                keep the highest rubric-scoring iteration.

Usage:
  python -m scripts.run_diagram_benchmark --run 1 --condition both --max-iters 3
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path

from src.config import ExperimentConfig, ModelConfig
from src.evaluation.checklist_judge import checklist_score
from src.rendering.browser import BrowserRenderer
from src.utils.code_utils import extract_code
from src.utils.llm_client import get_client
from src.experiments.hard_benchmark_runner import DESIGN_PRINCIPLES

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("diagram")
logger.setLevel(logging.INFO)

TASKS_PATH = "data/hard_benchmarks/diagrams/diagram_tasks.json"

DIAGRAM_SYSTEM_PROMPT = """\
You are an expert at producing publication-quality technical diagrams as \
self-contained HTML files using inline CSS and inline SVG.

Rules:
- Output a COMPLETE, self-contained HTML file (<!DOCTYPE html> through </html>)
- Draw the diagram with inline SVG (<svg> ... </svg>); use CSS only for page setup
- No external assets, fonts, scripts, or CDN links
- The figure must render at 1920x1080 pixels with NO scrolling needed
- Every label must be fully inside its shape; nothing may overflow or clip
- Boxes/nodes must NOT overlap each other; arrows must not pass through nodes
  that are not their endpoints; sibling elements must be evenly spaced

%s

- Wrap your complete HTML in a ```html code block""" % DESIGN_PRINCIPLES

REVISION_PROMPT = """\
Your previous diagram (shown above) was rendered and inspected. A rubric check \
found these specific requirements VIOLATED:

{violations}

Fix ONLY these violations. Apply the smallest edit that satisfies them; \
preserve everything that already passed. Re-emit the COMPLETE updated HTML in a \
```html code block."""


def _format_violations(verdicts: list[dict]) -> str:
    lines = []
    for v in verdicts:
        if not v["satisfied"]:
            ev = (v.get("evidence") or "").strip()
            lines.append(f"- {v['requirement']}" + (f"  (observed: {ev})" if ev else ""))
    return "\n".join(lines)


def run_task(task: dict, condition: str, max_iters: int,
             proposer_model: ModelConfig, judge_model: ModelConfig,
             renderer: BrowserRenderer, client) -> dict:
    client.reset_counters()
    start = time.time()

    description = task["description"]
    reqs = task["requirements"]
    prompt = description  # requirements are the hidden rubric; not given to proposer verbatim

    messages = [{"role": "user", "content": prompt}]
    resp = client.generate(config=proposer_model, system=DIAGRAM_SYSTEM_PROMPT,
                           messages=messages)
    html = extract_code(resp.content, "html")
    if not html.strip().startswith("<!") and not html.strip().startswith("<html"):
        html = resp.content

    max_i = 1 if condition == "single_shot" else max_iters
    best = None  # (score, html, verdicts, b64)
    trace = [{"step": "generate", "full_output": resp.content}]
    iterations = 1

    for i in range(max_i):
        try:
            png = renderer.render_html(html)
            b64 = base64.b64encode(png).decode()
        except Exception as e:  # noqa: BLE001
            logger.warning("render failed %s iter %d: %s", task["task_id"], i, e)
            break

        res = checklist_score(b64, reqs, model_config=judge_model, client=client)
        score = res["score"]
        if best is None or score > best[0]:
            best = (score, html, res["verdicts"], b64)

        if condition == "single_shot" or score >= 1.0:
            break
        if i < max_i - 1:
            violations = _format_violations(res["verdicts"])
            rev_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```html\n{html}\n```"},
                {"role": "user", "content": REVISION_PROMPT.format(violations=violations)},
            ]
            resp = client.generate(config=proposer_model,
                                   system=DIAGRAM_SYSTEM_PROMPT, messages=rev_messages)
            trace.append({"step": "revise", "violations": violations,
                          "full_output": resp.content})
            new_html = extract_code(resp.content, "html")
            if new_html.strip():
                html = new_html
            iterations = i + 2

    failed = best is None  # render never succeeded; exclude from aggregation
    score, html, verdicts, b64 = best if best else (None, html, [], "")
    usage = client.get_usage_summary()
    return {
        "task_id": task["task_id"],
        "category": task.get("category") or task.get("name", ""),
        "condition": condition,
        "rubric_score": score,
        "failed": failed,
        "n_met": sum(v["satisfied"] for v in verdicts),
        "n_total": len(reqs),
        "violations": [v["index"] for v in verdicts if not v["satisfied"]],
        "verdicts": verdicts,
        "iterations": iterations,
        "total_tokens": usage["total_tokens"],
        "wall_time_seconds": time.time() - start,
        "final_html": html,
        "screenshot_b64": b64,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--condition", choices=["single_shot", "reviewed", "both"],
                    default="both")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks (0=all)")
    ap.add_argument("--tasks", default=TASKS_PATH, help="task JSON path")
    ap.add_argument("--out-prefix", default="diagrams", help="output file prefix")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    if args.limit:
        tasks = tasks[:args.limit]

    proposer = ModelConfig(provider=ModelConfig.claude_sonnet().provider,
                           model_id="claude-sonnet-4-20250514",
                           max_tokens=8192, temperature=args.temperature)
    judge = ModelConfig.claude_sonnet()
    renderer = BrowserRenderer()
    client = get_client()

    conditions = ["single_shot", "reviewed"] if args.condition == "both" else [args.condition]
    out_path = Path(f"results/hard_benchmarks/{args.out_prefix}_run{args.run}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        row = {"task_id": task["task_id"],
               "category": task.get("category") or task.get("name", ""),
               "run": args.run}
        for cond in conditions:
            r = run_task(task, cond, args.max_iters, proposer, judge, renderer, client)
            pre = "ss" if cond == "single_shot" else "rv"
            row[f"{pre}_score"] = r["rubric_score"]
            row[f"{pre}_n_met"] = r["n_met"]
            row[f"{pre}_violations"] = r["violations"]
            row[f"{pre}_iters"] = r["iterations"]
            row[f"{pre}_tokens"] = r["total_tokens"]
            row[f"{pre}_html"] = r["final_html"]
            row[f"{pre}_screenshot_b64"] = r["screenshot_b64"]
            row[f"{pre}_trace"] = r["trace"]
            logger.info("run%d %s [%s]: score=%s (%d/%d) iters=%d viol=%s%s",
                        args.run, task["task_id"], cond,
                        "FAILED" if r["failed"] else f"{r['rubric_score']:.2f}",
                        r["n_met"], r["n_total"], r["iterations"], r["violations"],
                        " [FAILED]" if r["failed"] else "")
        records.append(row)
        out_path.write_text(json.dumps(records, indent=2))  # checkpoint
    logger.info("wrote %d task records to %s", len(records), out_path)


if __name__ == "__main__":
    main()
