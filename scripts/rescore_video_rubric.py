"""Re-score the saved video on-policy runs with the per-requirement rubric.

Video was never brought into the rubric program (only a holistic 0-1 score).
This reuses the already-generated single-shot / reviewed artifacts (the saved
editing scripts), RE-EXECUTES each script to regenerate /tmp/output.mp4, and
scores the WHOLE rendered video with Gemini 3.1 Pro's native video
understanding (`checklist_score_video`) — judging each requirement against the
full clip end-to-end rather than a handful of sampled stills. This lets the
judge verify temporal/whole-file properties (exact duration, motion,
transitions, codec validity) that keyframe sampling cannot see.

For each (task, run, condition) it records:
  *_holistic   -- the original holistic VLM/LLM score (for the noise comparison)
  *_rubric     -- fraction of requirements satisfied under the binary rubric
  *_n_met / *_violations
  *_exec_ok    -- whether the saved script actually produced a valid mp4

Usage: python -m scripts.rescore_video_rubric
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from src.evaluation.gemini_video_judge import checklist_score_video

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rescore_video")
logger.setLevel(logging.INFO)

OUTPUT_MP4 = Path("/tmp/output.mp4")


def _run_script(code: str, timeout: int = 180) -> bool:
    """Execute a saved editing script; return True iff it produced /tmp/output.mp4."""
    if not code.strip():
        return False
    OUTPUT_MP4.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name
    try:
        subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        logger.warning("script timed out after %ss", timeout)
    finally:
        Path(script_path).unlink(missing_ok=True)
    return OUTPUT_MP4.exists() and OUTPUT_MP4.stat().st_size > 0


def _score_condition(code: str, task: dict) -> dict:
    """Execute the editing script, then rubric-score the whole rendered video."""
    reqs = task["requirements"]
    if not _run_script(code):
        return {"rubric": 0.0, "n_met": 0, "violations": list(range(len(reqs))),
                "exec_ok": False}
    res = checklist_score_video(str(OUTPUT_MP4), reqs)
    return {
        "rubric": res["score"], "n_met": res["n_met"],
        "violations": [v["index"] for v in res["verdicts"] if not v["satisfied"]],
        "exec_ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/hard_benchmarks/video/video_tasks.json")
    ap.add_argument("--run-prefix", default="video_onpolicy_run",
                    help="basename prefix of the on-policy run files to rescore")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results/hard_benchmarks/video_rubric_rescore.json")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in json.load(open(args.tasks))}
    run_paths = [f"results/hard_benchmarks/{args.run_prefix}{i}.json" for i in args.runs]
    out = Path(args.out)

    records = []
    for run_path in run_paths:
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
                    s = _score_condition(rec.get(code_key, "") or "", tasks[tid])
                    row[f"{cond}_rubric"] = s["rubric"]
                    row[f"{cond}_n_met"] = s["n_met"]
                    row[f"{cond}_violations"] = s["violations"]
                    row[f"{cond}_exec_ok"] = s["exec_ok"]
                except Exception as e:  # noqa: BLE001
                    logger.warning("fail %s %s %s: %s", tid, run_id, cond, e)
                    row[f"{cond}_rubric"] = None
            logger.info("%s %s: SS hol=%s rub=%s ok=%s | RV hol=%s rub=%s ok=%s",
                        run_id, tid, row.get("ss_holistic"), row.get("ss_rubric"),
                        row.get("ss_exec_ok"), row.get("rv_holistic"),
                        row.get("rv_rubric"), row.get("rv_exec_ok"))
            records.append(row)
            out.write_text(json.dumps(records, indent=2))  # checkpoint each task
    logger.info("wrote %d records to %s", len(records), out)


if __name__ == "__main__":
    main()
