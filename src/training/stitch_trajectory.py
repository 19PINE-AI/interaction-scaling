"""Stage D: stitch raw traces into single-assistant-turn chat-template examples.

Takes `data/training/review_traces_v2.json` (from collect_review_traces_v2) and
produces `data/training/sft_review_v1.json`, a list of chat-template examples
ready for SFT.

Format per example:

    {
      "task_id": "...",
      "stratum": "no_revise" | "one_revise",
      "tools": [<tool_schema_json>],
      "messages": [
        {"role": "user", "content": "<task description>"},
        {"role": "assistant", "content": "<preamble + code + before_exec>",
         "tool_calls": [{"type":"function", "function":{"name":"execute_code",
                          "arguments": "<json>"}}]},
        {"role": "tool", "content": "<tool result JSON>"},
        ... (one_revise adds diagnosis/revision/second tool call/response)
        {"role": "assistant", "content": "<confirm>"}
      ]
    }

The Qwen3 tokenizer's `apply_chat_template` understands this directly.

Usage:
    python -m src.training.stitch_trajectory \
        --traces data/training/review_traces_v2.json \
        --output data/training/sft_review_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute Python code combined with a test harness. "
            "Returns a JSON object {passed, stdout, stderr, error} summarizing the run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python source to execute."},
                "test_code": {"type": "string", "description": "Assertions to run after the code."},
            },
            "required": ["code", "test_code"],
        },
    },
}

# ~30 connective variants woven between preamble / before_exec to avoid
# teaching one canonical phrase.
CONNECTIVES_VERIFY = [
    "Before I submit, I'd like to check this actually works.",
    "Let me run the tests to be sure.",
    "I'll verify this by executing the tests now.",
    "Quick sanity check — running the tests against this implementation.",
    "Worth verifying before I call this done.",
    "Let me confirm the behavior by running the tests.",
    "Running the tests to make sure I haven't missed a case.",
    "One quick execution to validate this.",
    "Let me prove it works by running the tests.",
    "Now I'll run it against the provided tests.",
    "Executing the tests against this draft.",
    "Verifying correctness before submission.",
    "Let me actually run it — I want to see green before submitting.",
    "A quick test run to double-check.",
    "I'd rather verify now than ship a regression.",
    "Let me run this through the tests.",
    "Running the test harness on this attempt.",
    "Let me see it pass before submitting.",
    "Kicking off the tests to confirm.",
    "Before finalizing, I'll run the test suite.",
]

CONNECTIVES_FAIL_TO_FIX = [
    "That's a real failure — let me correct it.",
    "The test is right, my code is wrong. Fixing.",
    "OK, that confirms the bug. Applying the correction.",
    "I need to fix this before submitting.",
    "Clear bug — let me patch it.",
    "Reproduced the issue. Now the fix.",
    "I can see the flaw now. Correcting it.",
    "The failure points directly at the bug. Let me address it.",
    "That gives me the exact spot to fix.",
    "The error trace is unambiguous — time to correct the implementation.",
]

CONNECTIVES_VERIFY_FIX = [
    "Re-running the tests to confirm the fix holds.",
    "Let me verify the correction actually solves it.",
    "Running the tests against the fixed version.",
    "Checking that the revised code passes.",
    "Let me confirm the patch works by re-executing.",
    "Re-executing to make sure the fix sticks.",
    "Running the tests one more time on the corrected implementation.",
    "Second pass — verifying the fix.",
    "Validating the correction with the same tests.",
    "Let me confirm we're green now.",
]

CONFIRM_PASS = [
    "All green — submitting.",
    "Tests pass. This is ready.",
    "Everything checks out. Submitting the solution.",
    "Perfect — tests all pass. Done.",
    "Confirmed working. Final answer.",
    "That's it — submitting.",
]


def tool_result_str(exec_result: dict) -> str:
    """Compact JSON of the execution result for the tool response."""
    return json.dumps({
        "passed": exec_result["passed"],
        "error": exec_result.get("error"),
        "stdout": (exec_result.get("stdout") or "")[-800:],
        "stderr": (exec_result.get("stderr") or "")[-800:],
    })


def code_block(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def stitch(trace: dict, rng: random.Random) -> dict:
    voice = trace["voice"]
    stratum = trace["stratum"]
    description = trace["description"]
    test_code = trace["test_code"]

    tool_args_draft = json.dumps({"code": trace["draft_code"], "test_code": test_code})
    draft_result_str = tool_result_str(trace["draft_exec"])

    # First assistant content: preamble + code + connective.
    verify_lead = rng.choice(CONNECTIVES_VERIFY)
    # prefer model-provided before_exec if it's short and specific; otherwise fall back to canned
    before_exec = (voice.get("before_exec") or "").strip()
    if not before_exec or len(before_exec) > 300:
        before_exec = verify_lead

    first_text = (
        f"{voice['preamble'].strip()}\n\n"
        f"{code_block(trace['draft_code'])}\n\n"
        f"{before_exec}"
    )

    messages = [
        {"role": "user", "content": description},
        {
            "role": "assistant",
            "content": first_text,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": tool_args_draft,
                },
            }],
        },
        {"role": "tool", "content": draft_result_str},
    ]

    if stratum == "no_revise":
        confirm = (voice.get("confirm") or "").strip() or rng.choice(CONFIRM_PASS)
        messages.append({"role": "assistant", "content": confirm})
    else:  # one_revise
        diagnosis = (voice.get("diagnosis") or "").strip()
        fix_lead = rng.choice(CONNECTIVES_FAIL_TO_FIX)
        revision_pre = (voice.get("revision_preamble") or "").strip() or "Here is the fix:"

        second_text = (
            f"{diagnosis}\n\n{fix_lead}\n\n"
            f"{revision_pre}\n\n"
            f"{code_block(trace['fixed_code'])}\n\n"
            f"{rng.choice(CONNECTIVES_VERIFY_FIX)}"
        )

        tool_args_fixed = json.dumps({"code": trace["fixed_code"], "test_code": test_code})
        fixed_result_str = tool_result_str(trace["fixed_exec"])
        messages.append({
            "role": "assistant",
            "content": second_text,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": tool_args_fixed,
                },
            }],
        })
        messages.append({"role": "tool", "content": fixed_result_str})
        confirm = (voice.get("confirm") or "").strip() or rng.choice(CONFIRM_PASS)
        messages.append({"role": "assistant", "content": confirm})

    return {
        "task_id": trace["task_id"],
        "stratum": stratum,
        "bug_class": trace.get("bug_class", "?"),
        "tools": [TOOL_SCHEMA],
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/training/review_traces_v2.json")
    ap.add_argument("--output", default="data/training/sft_review_v1.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    traces = json.loads(Path(args.traces).read_text())
    logger.info("Loaded %d raw traces", len(traces))

    rng = random.Random(args.seed)
    out = []
    strata_counts = {"no_revise": 0, "one_revise": 0}
    for trace in traces:
        ex = stitch(trace, rng)
        out.append(ex)
        strata_counts[ex["stratum"]] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("Stitched %d examples (%d no_revise / %d one_revise) -> %s",
                len(out), strata_counts["no_revise"], strata_counts["one_revise"],
                args.output)


if __name__ == "__main__":
    main()
