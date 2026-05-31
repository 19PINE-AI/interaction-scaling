"""Score the hardened-slides on-policy runs with BOTH grounded instruments.

The original slide suite was saturated under the binary rubric (single-shot
0.948), so it had no headroom to measure interaction scaling. `slide_tasks_hard`
(20 tasks, 10 binary requirements each) is the hardened replacement. This scores
its single-shot vs reviewed artifacts two ways:

  rubric    -- binary per-requirement checklist (hires quadrant tiling),
               externally averaged  (`checklist_score`)
  geometric -- deterministic DOM geometry defect count, lower is better
               (`geometric_defects`); the reliable fixed-canvas instrument

Requires the artifacts produced by:
  python3 run_onpolicy_augmentation.py --categories slides_hard --runs 3

Usage: python -m scripts.rescore_slides_hard
"""

import base64
import json
import logging
from pathlib import Path

from src.evaluation.checklist_judge import checklist_score
from src.evaluation.geometric_checker import geometric_defects
from src.rendering.browser import BrowserRenderer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rescore_slides_hard")
logger.setLevel(logging.INFO)

TASKS = {
    t["task_id"]: t
    for t in json.load(open("data/hard_benchmarks/slides/slide_tasks_hard.json"))
}
RUNS = [f"results/hard_benchmarks/slides_hard_onpolicy_run{i}.json" for i in (1, 2, 3)]
OUT = Path("results/hard_benchmarks/slides_hard_rubric_rescore.json")


def main():
    renderer = BrowserRenderer()
    records = []
    for run_path in RUNS:
        if not Path(run_path).exists():
            logger.warning("missing run %s", run_path)
            continue
        run_id = Path(run_path).stem
        for rec in json.load(open(run_path)):
            tid = rec["task_id"]
            if tid not in TASKS:
                continue
            reqs = TASKS[tid]["requirements"]
            row = {"task_id": tid, "run": run_id, "n_reqs": len(reqs)}
            for cond, code_key, hol_key in (
                ("ss", "ss_final_code", "ss_quality"),
                ("rv", "rv_final_code", "rv_quality"),
            ):
                code = rec.get(code_key, "") or ""
                row[f"{cond}_holistic"] = rec.get(hol_key)
                try:
                    png = renderer.render_html(code)
                    b64 = base64.b64encode(png).decode()
                    res = checklist_score(b64, reqs)
                    row[f"{cond}_rubric"] = res["score"]
                    row[f"{cond}_n_met"] = res["n_met"]
                    row[f"{cond}_violations"] = [
                        v["index"] for v in res["verdicts"] if not v["satisfied"]]
                except Exception as e:  # noqa: BLE001
                    logger.warning("render/rubric fail %s %s %s: %s",
                                   tid, run_id, cond, e)
                    row[f"{cond}_rubric"] = None
                try:
                    g = geometric_defects(code, renderer=renderer)
                    row[f"{cond}_defects"] = g["n_defects"]
                    row[f"{cond}_defect_breakdown"] = {
                        k: g[k] for k in
                        ("text_overlap", "out_of_bounds", "overflow", "scrollbar")
                        if k in g}
                except Exception as e:  # noqa: BLE001
                    logger.warning("geom fail %s %s %s: %s", tid, run_id, cond, e)
                    row[f"{cond}_defects"] = None
            logger.info("%s %s: SS rub=%s def=%s | RV rub=%s def=%s",
                        run_id, tid, row.get("ss_rubric"), row.get("ss_defects"),
                        row.get("rv_rubric"), row.get("rv_defects"))
            records.append(row)
            OUT.write_text(json.dumps(records, indent=2))  # checkpoint each task
    logger.info("wrote %d records to %s", len(records), OUT)


if __name__ == "__main__":
    main()
