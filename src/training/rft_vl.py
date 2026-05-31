"""Rejection-sampling Fine-Tuning (RFT) for VL student.

Pragmatic RL alternative to full GRPO for multi-turn multi-modal review loops.

Pipeline:
  1. Roll out N trajectories per training task with the SFT'd student
     (temperature > 0 for diversity).
  2. Judge each trajectory via Gemini multimodal judge (same as eval).
  3. Keep only judge-passing trajectories. Drop the rest.
  4. SFT the student on the kept trajectories (continuing from current policy
     with low LR, using same masking and reasoning-cap as Phase 5 V3).

Why RFT over GRPO:
  - No need for KL-to-reference, PPO clipping, or per-token advantage.
  - Reward is binary (judge keep/reject), works naturally with rejection
    sampling.
  - Expert iteration: each round bootstraps the next policy from its own
    best samples.

Usage:
    PYTHONPATH=. python -m src.training.rft_vl \\
        --adapter models/qwen3-vl-8b-vl-sft-v3 \\
        --tasks data/training/code_tasks_v2_scrubbed.json \\
        --rollouts-per-task 4 --workers 1 \\
        --output models/qwen3-vl-8b-vl-rft-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from src.evaluation.code_eval import CodeEvaluator
from src.training.run_vl_student import load_model, run_task

logger = logging.getLogger(__name__)


def collect_trajectories(processor, model, evaluator: CodeEvaluator,
                         tasks: list[dict], rollouts_per_task: int,
                         max_turns: int, temperature: float,
                         force_close_think_after: int,
                         out_path: Path) -> list[dict]:
    """Roll out N trajectories per task; save incrementally."""
    trajs: list[dict] = []
    if out_path.exists():
        trajs = json.loads(out_path.read_text())
        done_keys = {(t["category"], t["task_id"], t["seed"]) for t in trajs}
        logger.info("Resuming with %d trajectories already collected", len(trajs))
    else:
        done_keys = set()

    for task in tasks:
        for r in range(rollouts_per_task):
            seed = 1000 + r
            cat = task["category"]
            key = (cat, task["task_id"], seed)
            if key in done_keys:
                continue
            torch.manual_seed(seed)
            t0 = time.time()
            trace = run_task(processor, model, task, cat, max_turns, evaluator,
                             temperature=temperature,
                             repetition_penalty=1.1,
                             force_close_think_after=force_close_think_after)
            trace["seed"] = seed
            trace["elapsed_s"] = round(time.time() - t0, 2)
            trajs.append(trace)
            out_path.write_text(json.dumps(trajs, indent=2, default=str))
            logger.info(
                "  [%s/%s seed=%d] status=%s elapsed=%.1fs (n=%d)",
                cat, task["task_id"], seed,
                trace["status"], trace["elapsed_s"], len(trajs),
            )
    return trajs


def judge_and_filter(trajectories_path: str, output_path: str) -> int:
    """Run gemini judge over collected trajectories; return n_kept."""
    from src.training.judge_vl_traces import (
        load_task_by_id, render_final_images, call_judge, format_trace_for_judge,
    )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY required for judge")
    trajs = json.loads(Path(trajectories_path).read_text())
    judged = []
    kept = 0
    for trace in trajs:
        task = load_task_by_id(trace["category"], trace["task_id"])
        if task is None:
            continue
        prompt = format_trace_for_judge(trace, task)
        images = render_final_images(trace, task)
        verdict = call_judge(prompt, images)
        if verdict is None:
            verdict = {"keep": False, "error": "judge_failed"}
        trace["judge"] = verdict
        judged.append(trace)
        if verdict.get("keep"):
            kept += 1
        logger.info("  judge %s/%s seed=%d keep=%s",
                    trace["category"], trace["task_id"],
                    trace.get("seed"), verdict.get("keep"))
    Path(output_path).write_text(json.dumps(judged, indent=2, default=str))
    return kept


def trace_to_sft_example(trace: dict, task: dict) -> dict | None:
    """Convert a kept trajectory to a Qwen3-VL SFT example (re-rendered images
    on disk, messages list)."""
    from src.training.prepare_vl_sft import build_sft_example
    from pathlib import Path as P
    image_dir = P("data/training/vl_rft_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    return build_sft_example(trace, task, image_dir)


def write_rft_jsonl(judged_path: str, out_jsonl: str) -> int:
    """Build SFT JSONL from judge-kept trajectories."""
    from src.training.judge_vl_traces import load_task_by_id
    from src.training.prepare_vl_sft import build_sft_example, REASONING_CAP
    judged = json.loads(Path(judged_path).read_text())
    image_dir = Path("data/training/vl_rft_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_jsonl, "w") as f:
        for trace in judged:
            if not (trace.get("judge") or {}).get("keep"):
                continue
            task = load_task_by_id(trace["category"], trace["task_id"])
            if task is None:
                continue
            # student trajectories don't carry teacher 'reasoning' field —
            # the assistant content already includes <think>...</think> from
            # the student's own generation, so build_sft_example will use
            # turn['assistant'] as-is.
            ex = build_sft_example(trace, task, image_dir)
            if ex is None:
                continue
            f.write(json.dumps(ex) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/qwen3-vl-8b-vl-sft-v3")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--tasks", required=True,
                    help="JSON list of training tasks; should NOT include held-out")
    ap.add_argument("--rollouts-per-task", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--force-close-think-after", type=int, default=4000)
    ap.add_argument("--trajectories-out",
                    default="data/training/vl_rft_trajectories.json")
    ap.add_argument("--judged-out",
                    default="data/training/vl_rft_trajectories_judged.json")
    ap.add_argument("--rft-jsonl-out",
                    default="data/training/vl_rft_sft.jsonl")
    ap.add_argument("--skip-rollout", action="store_true",
                    help="Skip rollout step (use existing trajectories)")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip judge step (use existing judged file)")
    ap.add_argument("--log", default="logs/rft_vl.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    tasks = json.loads(Path(args.tasks).read_text())
    logger.info("RFT pipeline on %d tasks, %d rollouts each",
                len(tasks), args.rollouts_per_task)

    if not args.skip_rollout:
        logger.info("Step 1: rollouts")
        processor, model = load_model(args.model, args.adapter)
        evaluator = CodeEvaluator()
        Path(args.trajectories_out).parent.mkdir(parents=True, exist_ok=True)
        trajectories = collect_trajectories(
            processor, model, evaluator, tasks,
            args.rollouts_per_task, args.max_turns,
            args.temperature, args.force_close_think_after,
            Path(args.trajectories_out),
        )
        logger.info("  %d trajectories collected -> %s",
                    len(trajectories), args.trajectories_out)
        # Free model memory before judge+SFT
        del model
        del processor
        torch.cuda.empty_cache()

    if not args.skip_judge:
        logger.info("Step 2: judge")
        kept = judge_and_filter(args.trajectories_out, args.judged_out)
        logger.info("  %d trajectories judge-kept", kept)

    logger.info("Step 3: build RFT JSONL")
    n = write_rft_jsonl(args.judged_out, args.rft_jsonl_out)
    logger.info("  %d SFT examples -> %s", n, args.rft_jsonl_out)
    logger.info("Run train_vl_sft.py on %s next.", args.rft_jsonl_out)


if __name__ == "__main__":
    main()
