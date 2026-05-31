"""Re-score saved on-policy runs of a modality with the per-requirement rubric.

Reuses the already-generated single-shot / reviewed artifacts (no regeneration)
and replaces the holistic 0-1 VLM/LLM score with the structured-JSON binary
rubric, externally averaged. Supports:

  web        : render ss/rv HTML -> screenshot -> checklist_score
  animations : render ss/rv HTML -> frame sequence -> checklist_score_frames
  research   : ss/rv report text -> checklist_score_text

Usage:
  python -m scripts.rescore_modality_rubric --modality web
  python -m scripts.rescore_modality_rubric --modality animations
  python -m scripts.rescore_modality_rubric --modality research
"""

import argparse
import base64
import json
import logging
from pathlib import Path

from src.evaluation.checklist_judge import (
    checklist_score,
    checklist_score_frames,
    checklist_score_text,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rescore")
logger.setLevel(logging.INFO)

CFG = {
    "web": {
        "tasks": "data/hard_benchmarks/webpages/webpage_tasks.json",
        "runs": [f"results/hard_benchmarks/webpages_onpolicy_run{i}.json" for i in (1, 2, 3)],
        "out": "results/hard_benchmarks/web_rubric_rescore.json",
        "kind": "screenshot",
    },
    "animations": {
        "tasks": "data/hard_benchmarks/animations/animation_tasks.json",
        "runs": [f"results/hard_benchmarks/animations_onpolicy_run{i}.json" for i in (1, 2, 3)],
        "out": "results/hard_benchmarks/animations_rubric_rescore.json",
        "kind": "frames",
    },
    "research": {
        "tasks": "data/hard_benchmarks/research/research_tasks.json",
        "runs": [f"results/hard_benchmarks/research_onpolicy_run{i}.json" for i in (1, 2, 3)],
        "out": "results/hard_benchmarks/research_rubric_rescore.json",
        "kind": "text",
    },
}


def load_tasks(path):
    d = json.load(open(path))
    d = d if isinstance(d, list) else d.get("tasks", list(d.values())[0])
    return {t["task_id"]: t for t in d}


def score_artifact(kind, code, task, renderer):
    reqs = task["requirements"]
    if kind == "text":
        return checklist_score_text(code, reqs)
    if kind == "screenshot":
        png = renderer.render_html(code)
        return checklist_score(base64.b64encode(png).decode(), reqs)
    if kind == "frames":
        ftimes = task.get("frame_times_ms", [0, 1000, 2000, 3000])
        frames = renderer.render_animation_frames(code, ftimes)
        b64s = [base64.b64encode(f).decode() for f in frames]
        return checklist_score_frames(b64s, ftimes, reqs)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(CFG))
    args = ap.parse_args()
    cfg = CFG[args.modality]
    tasks = load_tasks(cfg["tasks"])

    renderer = None
    if cfg["kind"] != "text":
        from src.rendering.browser import BrowserRenderer
        renderer = BrowserRenderer()

    out = Path(cfg["out"])
    records = []
    for run_path in cfg["runs"]:
        if not Path(run_path).exists():
            logger.warning("missing run %s", run_path)
            continue
        run_id = Path(run_path).stem
        for rec in json.load(open(run_path)):
            tid = rec["task_id"]
            if tid not in tasks:
                continue
            row = {"task_id": tid, "run": run_id,
                   "n_reqs": len(tasks[tid]["requirements"])}
            for cond, code_key, hol_key in (
                ("ss", "ss_final_code", "ss_quality"),
                ("rv", "rv_final_code", "rv_quality"),
            ):
                row[f"{cond}_holistic"] = rec.get(hol_key)
                try:
                    res = score_artifact(cfg["kind"], rec.get(code_key, ""),
                                         tasks[tid], renderer)
                    row[f"{cond}_rubric"] = res["score"]
                    row[f"{cond}_n_met"] = res["n_met"]
                    row[f"{cond}_violations"] = [
                        v["index"] for v in res["verdicts"] if not v["satisfied"]]
                except Exception as e:  # noqa: BLE001
                    logger.warning("fail %s %s %s: %s", tid, run_id, cond, e)
                    row[f"{cond}_rubric"] = None
            logger.info("%s %s: SS hol=%s rub=%s | RV hol=%s rub=%s", run_id, tid,
                        row["ss_holistic"], row.get("ss_rubric"),
                        row["rv_holistic"], row.get("rv_rubric"))
            records.append(row)
            out.write_text(json.dumps(records, indent=2))
    logger.info("wrote %d records to %s", len(records), out)


if __name__ == "__main__":
    main()
