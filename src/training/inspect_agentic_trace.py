"""Generate compact trace inspection reports for manual quality review.

For each requested trace, writes:
- {out_dir}/{cat}_{task_id}_compact.json — compact step-by-step summary
- {out_dir}/{cat}_{task_id}/step{N}_{NNN}.png — copies of saved screenshots

Use this to spot-check trace quality before SFT.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def step_summary(step: dict, content_chars: int = 400, arg_chars: int = 200) -> dict:
    out = {
        "step": step.get("step"),
        "gen_s": step.get("gen_s"),
        "assistant_content_head": (step.get("assistant_content") or "")[:content_chars],
    }
    tcs = []
    for tc in step.get("tool_calls", []) or []:
        fn = tc.get("function", {}) if "function" in tc else tc
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
        except Exception:
            args = {}
        d = {"name": name}
        if name == "write_file":
            d["path"] = args.get("path")
            content = args.get("content", "")
            d["content_bytes"] = len(content)
            d["content_head"] = content[:arg_chars]
        elif name == "bash":
            d["command"] = args.get("command", "")[:arg_chars]
        elif name == "read_file":
            d["path"] = args.get("path")
        else:
            d["arguments_head"] = str(args)[:arg_chars]
        tcs.append(d)
    out["tool_calls"] = tcs
    out["tool_results"] = step.get("tool_results", [])
    return out


def inspect_one(trace: dict, out_dir: Path) -> dict:
    cat = trace.get("category", "?")
    tid = trace.get("task_id", "?")
    stem = f"{cat}_{tid}"
    summary = {
        "task_id": tid,
        "category": cat,
        "status": trace.get("status"),
        "final_passed": trace.get("final_passed"),
        "n_steps": trace.get("n_steps"),
        "elapsed_s": trace.get("elapsed_s"),
        "saved_images_count": len(trace.get("saved_images", [])),
        "steps": [step_summary(s) for s in trace.get("steps", [])],
    }
    if "judge" in trace:
        summary["judge"] = trace["judge"]

    # Copy screenshots
    img_subdir = out_dir / stem
    img_subdir.mkdir(parents=True, exist_ok=True)
    for sp in trace.get("saved_images", []):
        sp_path = Path(sp)
        if sp_path.exists():
            shutil.copy(sp_path, img_subdir / sp_path.name)

    (out_dir / f"{stem}_compact.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/training/agentic_traces.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-ids", nargs="*", default=None,
                    help="Specific task_ids to inspect, e.g. 'code_001 web_005'")
    ap.add_argument("--category", default=None,
                    help="Restrict to one category (code/webpages/slides)")
    ap.add_argument("--status", default=None,
                    help="Restrict to status (final/max_steps/api_error)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max traces to inspect (after filtering)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = json.loads(Path(args.input).read_text())
    if args.task_ids:
        ids = set(args.task_ids)
        traces = [t for t in traces if t.get("task_id") in ids]
    if args.category:
        traces = [t for t in traces if t.get("category") == args.category]
    if args.status:
        traces = [t for t in traces if t.get("status") == args.status]
    if args.limit:
        traces = traces[:args.limit]
    print(f"Inspecting {len(traces)} traces -> {out_dir}")
    for t in traces:
        s = inspect_one(t, out_dir)
        print(f"  {s['category']}/{s['task_id']}: status={s['status']} "
              f"steps={s['n_steps']} imgs={s['saved_images_count']} "
              f"final_passed={s['final_passed']}")


if __name__ == "__main__":
    main()
