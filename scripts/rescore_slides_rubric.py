"""Re-score the existing slide on-policy runs with the per-requirement rubric.

For every (task, seed, condition) it re-renders the saved HTML and scores it
with the binary checklist judge, alongside the original holistic score. The
output supports the "the holistic metric was the masker" analysis:

* holistic noise magnitude (|holistic - checklist| and same-artifact variance)
* saturation under the rubric vs under the holistic score
* rubric-based single-shot vs reviewed lift + paired sign test

Usage: python -m scripts.rescore_slides_rubric
"""

import base64
import json
import logging
from pathlib import Path

from src.evaluation.checklist_judge import checklist_score
from src.rendering.browser import BrowserRenderer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rescore")
logger.setLevel(logging.INFO)

TASKS = {
    t["task_id"]: t
    for t in json.load(open("data/hard_benchmarks/slides/slide_tasks.json"))
}
RUNS = [
    f"results/hard_benchmarks/slides_onpolicy_run{i}.json" for i in (1, 2, 3)
]
OUT = Path("results/hard_benchmarks/slides_rubric_rescore.json")


def main():
    renderer = BrowserRenderer()
    records = []
    for run_path in RUNS:
        run_id = Path(run_path).stem
        data = json.load(open(run_path))
        for rec in data:
            tid = rec["task_id"]
            reqs = TASKS[tid]["requirements"]
            row = {"task_id": tid, "run": run_id, "n_reqs": len(reqs)}
            for cond, code_key, hol_key in (
                ("ss", "ss_final_code", "ss_quality"),
                ("rv", "rv_final_code", "rv_quality"),
            ):
                code = rec.get(code_key, "")
                row[f"{cond}_holistic"] = rec.get(hol_key)
                try:
                    png = renderer.render_html(code)
                    b64 = base64.b64encode(png).decode()
                    res = checklist_score(b64, reqs)
                    row[f"{cond}_rubric"] = res["score"]
                    row[f"{cond}_n_met"] = res["n_met"]
                    row[f"{cond}_violations"] = [
                        v["index"] for v in res["verdicts"] if not v["satisfied"]
                    ]
                except Exception as e:  # noqa: BLE001
                    logger.warning("render/judge failed %s %s %s: %s",
                                   tid, run_id, cond, e)
                    row[f"{cond}_rubric"] = None
            logger.info(
                "%s %s: SS hol=%.2f rub=%s | RV hol=%.2f rub=%s",
                run_id, tid, row["ss_holistic"] or 0, row.get("ss_rubric"),
                row["rv_holistic"] or 0, row.get("rv_rubric"),
            )
            records.append(row)
            OUT.write_text(json.dumps(records, indent=2))  # checkpoint each task

    logger.info("Wrote %d records to %s", len(records), OUT)


if __name__ == "__main__":
    main()
