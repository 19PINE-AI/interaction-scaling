"""Convert agentic teacher trajectories into multimodal SFT JSONL.

Input format (output of `collect_agentic_traces.py`):
- `messages`: full OpenAI-style chat history with assistant.tool_calls and
  tool/user roles. For image tool returns, the trace stores a `tool` text
  response (with hint "see: next user message") followed by a `user` message
  carrying the image as base64 `image_url`. This pattern is forced on us by
  the OpenRouter API which doesn't reliably accept images in tool role.
- `saved_images`: file paths to the actual PNGs in step order.

For SFT we collapse the (tool-text + user-image) pair into a single Qwen3-VL
`tool` role message whose content is `[{type:text, ...}, {type:image, image:<path>}]`.
The Qwen3-VL chat template renders this correctly:
    <|im_start|>user
    <tool_response>
    <summary text><|vision_start|><|image_pad|><|vision_end|>
    </tool_response><|im_end|>

We replace the heavy agentic system prompt used at trace collection time
with a stripped student-target prompt. Tool schemas auto-inject under the
chat template's `# Tools` section.

Usage:
    python -m src.training.prepare_agentic_sft \\
        --input data/training/agentic_traces_judged.json \\
        --output data/training/agentic_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


STUDENT_SYSTEM_PROMPT = (
    "You are an autonomous engineer agent. Use the provided tools "
    "(write_file, read_file, bash) to solve the user's task. "
    "For visual tasks, render the artifact to a PNG via bash and "
    "use read_file to view it before revising. "
    "When the task is complete, emit <final>OK</final> in your final "
    "assistant message and stop calling tools."
)


# Loaded from agent_workspace.TOOLS_SCHEMA — re-export for convenience to
# downstream scripts that need the tools list at training time.
from src.training.agent_workspace import TOOLS_SCHEMA  # noqa: E402


def _is_image_tool_text(content) -> bool:
    """Heuristic: a tool message that says image will arrive next-user."""
    if not isinstance(content, str):
        return False
    return '"kind": "image"' in content and '"see"' in content


def _user_msg_image_paths(msg, saved_images_iter) -> list[str] | None:
    """If msg is a user message carrying image_url parts, return matched
    saved-image file paths (popping from the iter). Returns None otherwise."""
    content = msg.get("content")
    if not isinstance(content, list):
        return None
    n_imgs = sum(1 for p in content if p.get("type") == "image_url")
    if n_imgs == 0:
        return None
    paths = []
    for _ in range(n_imgs):
        try:
            paths.append(next(saved_images_iter))
        except StopIteration:
            return None
    return paths


def collapse_messages(trace: dict) -> list[dict] | None:
    """Walk the saved messages and emit the SFT message list:
    - keep system/user/assistant (with tool_calls) as-is
    - for tool messages with image hint + following user(image_url): collapse
      into a single tool message with image content (file path)
    - replace the system prompt with the stripped student prompt
    """
    saved_images = trace.get("saved_images", [])
    img_iter = iter(saved_images)
    src = trace.get("messages")
    if not src:
        return None
    out: list[dict] = []
    i = 0
    while i < len(src):
        m = src[i]
        role = m.get("role")
        content = m.get("content")

        if role == "system":
            # Replace with stripped student prompt
            out.append({"role": "system", "content": STUDENT_SYSTEM_PROMPT})
            i += 1
            continue

        if role == "user":
            # First user is the task; stays as text content
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                i += 1
                continue
            # Other user messages are image-follow-ups for prior tool — should
            # have been consumed when we hit the tool message. Skip if seen
            # alone (defensive).
            if isinstance(content, list) and any(p.get("type") == "image_url" for p in content):
                # orphan image follow-up; consume + drop matching saved_image
                _ = _user_msg_image_paths(m, img_iter)
                i += 1
                continue
            # Fallback
            out.append({"role": "user", "content": content})
            i += 1
            continue

        if role == "assistant":
            am = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                # Strip the chatcmpl-tool-... ids; let the chat template emit
                # bare <tool_call> XML without ids.
                am["tool_calls"] = [
                    {"type": "function", "function": tc.get("function", {})}
                    for tc in m["tool_calls"]
                ]
            out.append(am)
            i += 1
            continue

        if role == "tool":
            if _is_image_tool_text(content) and i + 1 < len(src):
                nxt = src[i + 1]
                paths = _user_msg_image_paths(nxt, img_iter)
                if paths is not None:
                    # Build a Qwen3-VL tool message with image content directly
                    parts = [{"type": "text", "text": "screenshot:"}]
                    for p in paths:
                        parts.append({"type": "image", "image": p})
                    out.append({"role": "tool", "content": parts})
                    i += 2
                    continue
            # Plain text tool response
            out.append({"role": "tool", "content": content})
            i += 1
            continue

        # Unknown role — pass through
        out.append(m)
        i += 1

    return out


def build_example(trace: dict) -> dict | None:
    msgs = collapse_messages(trace)
    if not msgs:
        return None
    # Sanity: must have at least 1 assistant message after the user task
    if not any(m.get("role") == "assistant" for m in msgs):
        return None
    # Verify all referenced image files actually exist on disk
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image":
                    if not Path(p["image"]).exists():
                        logger.warning("missing image %s for %s/%s",
                                       p["image"], trace.get("category"),
                                       trace.get("task_id"))
                        return None
    return {
        "task_id": trace.get("task_id"),
        "category": trace.get("category"),
        "messages": msgs,
        "tools": TOOLS_SCHEMA,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="agentic traces JSON (raw or judged)")
    ap.add_argument("--output", default="data/training/agentic_sft.jsonl")
    ap.add_argument("--require-final-passed", action="store_true",
                    help="Only keep traces where final_passed=True (code) or status=final (visual)")
    ap.add_argument("--require-judge-keep", action="store_true",
                    help="Only keep traces where judge.keep=True (after running judge)")
    ap.add_argument("--max-steps-cap", type=int, default=20,
                    help="Drop traces with more steps than this (training seq length sanity)")
    ap.add_argument("--max-chars", type=int, default=80000,
                    help="Drop examples whose rendered chat-template text exceeds this length")
    ap.add_argument("--log", default="logs/prepare_agentic_sft.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler(sys.stdout)],
    )

    traces = json.loads(Path(args.input).read_text())
    logger.info("Loaded %d traces from %s", len(traces), args.input)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Lazy-load processor only if char filter is requested (it requires
    # downloading the tokenizer, slow). We use a cheap proxy: char count of
    # serialized messages.
    n_kept = 0
    n_skipped = {"steps": 0, "no_msgs": 0, "judge": 0, "final_passed": 0,
                 "missing_image": 0, "too_long": 0}
    with open(args.output, "w") as f:
        for t in traces:
            if t.get("n_steps", 0) > args.max_steps_cap:
                n_skipped["steps"] += 1
                continue
            if args.require_judge_keep:
                if not (t.get("judge") or {}).get("keep"):
                    n_skipped["judge"] += 1
                    continue
            if args.require_final_passed:
                cat = t.get("category")
                if cat == "code":
                    if not t.get("final_passed"):
                        n_skipped["final_passed"] += 1
                        continue
                else:
                    if t.get("status") != "final":
                        n_skipped["final_passed"] += 1
                        continue
            ex = build_example(t)
            if ex is None:
                n_skipped["missing_image"] += 1
                continue
            # Cheap char-budget filter: serialized messages length.
            # Empirically chat_template adds ~20% to JSON-dumped content,
            # so applying cap on JSON-dumped messages is a fair proxy.
            ser_len = len(json.dumps(ex["messages"]))
            if ser_len > args.max_chars:
                n_skipped["too_long"] += 1
                continue
            f.write(json.dumps(ex) + "\n")
            n_kept += 1

    logger.info("Wrote %d examples -> %s; skipped: %s",
                n_kept, args.output, n_skipped)


if __name__ == "__main__":
    main()
