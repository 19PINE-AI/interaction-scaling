"""Convert filtered VL teacher traces into multimodal SFT examples for
Qwen3-VL-8B-Thinking.

Input: `data/training/vl_teacher_traces_filtered.json` (output of
`judge_vl_traces.py`, only `keep=true` traces).

Output: a JSONL of chat-format examples where each example's `messages`
list matches the Qwen3-VL chat template and image content is stored as
file paths to PNGs on disk.

Key decisions:
- We RETAIN teacher `<think>` reasoning in each assistant target, capped
  at REASONING_CAP chars per turn. Empirically (Phase 5 v1 eval) stripping
  reasoning caused the student to emit `</think>` immediately and collapse
  to identical retries on hard tasks. Keeping it gives the student scratch
  space to plan the edit between reading the screenshot and emitting code.
- Screenshots are re-rendered on disk; stored as file paths.
- Turn 0 assistant message: `<think>PLAN</think>\\n\\n```<lang>...```
- Turn N>=1: `<think>PLAN</think>\\n\\nCRITIQUE...\\n\\n```<lang>...```
  (or ...<final>OK</final>)
- We mask user/tool turns (labels=-100); only assistant turns contribute.

Usage:
    python -m src.training.prepare_vl_sft \\
        --input data/training/vl_teacher_traces_filtered.json \\
        --image-dir data/training/vl_sft_images/ \\
        --output data/training/vl_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.training.collect_vl_traces import (
    SYSTEM_PROMPTS, build_user_turn, _render_one,
    extract_artifact,
)

logger = logging.getLogger(__name__)

import os
REASONING_CAP = int(os.environ.get("REASONING_CAP", "1500"))  # per-turn char cap on teacher <think> content


def render_artifact_for_sft(html: str, category: str, viewports: list[int],
                            image_dir: Path, stem: str,
                            max_dim: int = 1280) -> list[Path]:
    """Re-render each turn's artifact to disk; return image paths.

    For SFT we use only the largest desktop viewport (avoid 3 viewports per
    turn) and cap the longest image dimension to `max_dim` to keep vision
    token counts trainable. For webpages we still capture full_page so the
    student sees the whole layout, but it's downscaled.
    """
    from PIL import Image
    import io
    paths: list[Path] = []
    if category == "code":
        return []
    # Single viewport, single screen-height (no full-page capture). Trades
    # off scroll-below visibility for tractable sequence length (~5K vs 36K).
    w = 1920
    h = 1080
    png = _render_one(html, w, h, full_page=False)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (max(1, int(img.size[0] * scale)),
                    max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    p = image_dir / f"{stem}_{w}x{h}_resized.png"
    img.save(p, "PNG", optimize=True)
    paths.append(p)
    return paths


def build_sft_example(trace: dict, task: dict, image_dir: Path) -> dict | None:
    """Return a chat-format example with image references, or None if the
    trace cannot be converted (e.g. no artifact on turn 0)."""
    category = trace["category"]
    task_id = trace["task_id"]
    system = SYSTEM_PROMPTS[category]
    user0 = build_user_turn(task, category)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user0}]},
    ]

    turns = trace.get("turns", [])
    if not turns:
        return None

    for i, turn in enumerate(turns):
        assistant_content = turn.get("assistant") or ""
        reasoning = turn.get("reasoning") or ""
        if not assistant_content.strip():
            return None

        # Prepend teacher reasoning as the <think> block so the student
        # learns to plan-before-emit. Cap per turn to keep seq lengths
        # trainable (teacher reasoning median 3K, max 80K).
        if reasoning:
            r = reasoning[:REASONING_CAP].rstrip()
            full_target = f"<think>\n{r}\n</think>\n\n{assistant_content}"
        else:
            full_target = assistant_content
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_target}],
        })

        # After assistant turn i, if i < len(turns)-1, a feedback user turn
        # followed during collection -- reconstruct it for the SFT target.
        is_last = (i == len(turns) - 1)
        if is_last:
            continue

        # Extract the artifact from this turn to render/execute for feedback
        art = extract_artifact(assistant_content, category)
        if art is None:
            return None

        if category == "code":
            # Use stored text feedback from the trace if present
            next_fb = turns[i + 1].get("feedback") if i + 1 < len(turns) else None
            fb_text = (next_fb or {}).get("text_payload") or ""
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": fb_text}],
            })
        else:
            viewports = task.get("viewport_sizes", [1920]) if category == "webpages" else [1920]
            stem = f"{category}_{task_id}_turn{i}"
            try:
                img_paths = render_artifact_for_sft(art, category, viewports,
                                                   image_dir, stem)
            except Exception as e:
                logger.warning("render failed for %s turn %d: %s", task_id, i, e)
                return None
            parts = [{"type": "text", "text": "rendered screenshot(s):"}]
            for p in img_paths:
                parts.append({"type": "image", "image": str(p)})
            messages.append({"role": "user", "content": parts})

    return {
        "task_id": task_id,
        "category": category,
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--image-dir", default="data/training/vl_sft_images")
    ap.add_argument("--output", default="data/training/vl_sft.jsonl")
    ap.add_argument("--log", default="logs/prepare_vl_sft.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    traces = json.loads(Path(args.input).read_text())
    # Load all task specs once for lookup (orig + generated)
    task_by_id: dict[str, dict] = {}
    sources = [
        ("code", "data/hard_benchmarks/code/code_tasks.json"),
        ("webpages", "data/hard_benchmarks/webpages/webpage_tasks.json"),
        ("webpages", "data/training/vl_webpage_tasks_gen.json"),
        ("slides", "data/hard_benchmarks/slides/slide_tasks.json"),
        ("slides", "data/training/vl_slide_tasks_gen.json"),
    ]
    for cat, p in sources:
        if Path(p).exists():
            for t in json.loads(Path(p).read_text()):
                task_by_id[(cat, t["task_id"])] = t

    require_keep = os.environ.get("REQUIRE_JUDGE_KEEP", "1") == "1"
    kept = 0
    with open(args.output, "w") as f:
        for trace in traces:
            if require_keep and not trace.get("judge", {}).get("keep"):
                continue
            # Always require at least review_specific=True to filter out
            # failed-mid-trace teacher runs (e.g. errored, no-artifact).
            j = trace.get("judge") or {}
            if not j.get("review_specific"):
                continue
            task = task_by_id.get((trace["category"], trace["task_id"]))
            if task is None:
                logger.warning("no task spec for %s/%s",
                               trace["category"], trace["task_id"])
                continue
            ex = build_sft_example(trace, task, image_dir)
            if ex is None:
                continue
            f.write(json.dumps(ex) + "\n")
            kept += 1

    logger.info("Wrote %d SFT examples -> %s  (images in %s) require_keep=%s",
                kept, args.output, image_dir, require_keep)


if __name__ == "__main__":
    main()
