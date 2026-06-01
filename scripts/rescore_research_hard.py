"""Score the hard-research harness reports against each task's factual rubric.

The hard research tasks ship a `requirements` list of exact ground-truth facts
(laureates, dates, figures, ...) plus `trap_claims` (common errors). We score the
single-shot and reviewed (Type-3d search-grounded) reports with the binary
checklist judge over those requirement-facts: the requirement states the correct
fact, the judge marks whether the report matches it. Externally averaged.

Usage: python -m scripts.rescore_research_hard
"""
import json, logging
from pathlib import Path
from src.evaluation.checklist_judge import checklist_score_text

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("rescore_research_hard"); log.setLevel(logging.INFO)

TASKS = {t["task_id"]: t for t in json.load(open("data/hard_benchmarks/research/research_tasks_hard.json"))}
HARNESS = json.load(open("results/hard_benchmarks/research_hard_harness.json"))
OUT = Path("results/hard_benchmarks/research_hard_rubric_rescore.json")

def main():
    records = []
    for rec in HARNESS:
        tid = rec["task_id"]
        if tid not in TASKS: continue
        reqs = TASKS[tid]["requirements"]
        row = {"task_id": tid, "n_reqs": len(reqs)}
        for cond, key in (("ss", "ss_report"), ("rv", "rv_report")):
            try:
                res = checklist_score_text(rec.get(key, "") or "", reqs)
                row[f"{cond}_rubric"] = res["score"]
                row[f"{cond}_n_met"] = res["n_met"]
            except Exception as e:  # noqa: BLE001
                log.warning("fail %s %s: %s", tid, cond, e); row[f"{cond}_rubric"] = None
        log.info("%s: SS=%.3f RV=%.3f", tid, row.get("ss_rubric") or 0, row.get("rv_rubric") or 0)
        records.append(row)
        OUT.write_text(json.dumps(records, indent=2))
    log.info("wrote %d records", len(records))

if __name__ == "__main__":
    main()
