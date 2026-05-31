"""Stitch v3 variable-length traces into chat-template SFT examples.

Handles all strata produced by collect_review_traces_v3:
  - no_revise      (1 tool_call:  draft → pass → confirm)
  - one_revise     (2 tool_calls: draft → fail → fix1 → pass → confirm)
  - two_revise     (3 tool_calls: draft → fail → fix1 → fail → fix2 → pass)
  - three_revise   (4 tool_calls: draft → fail → fix1 → fail → fix2 → fail → fix3 → pass)

Each attempt in trace["attempts"] provides its own diagnosis + revision_preamble
(Sonnet-written, derived from the actual stderr it saw — no answer-key leak).

Usage:
    python -m src.training.stitch_trajectory_v3 \\
        --traces data/training/review_traces_v3.json \\
        --output data/training/sft_review_v3.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

CONNECTIVES_VERIFY = [
    "Before I submit, I'd like to check this actually works.",
    "Let me run the tests to be sure.",
    "I'll verify this by executing the tests now.",
    "Quick sanity check — running the tests against this implementation.",
    "Worth verifying before I call this done.",
    "Let me confirm the behavior by running the tests.",
    "Running the tests to make sure I haven't missed a case.",
    "Let me prove it works by running the tests.",
    "Now I'll run it against the provided tests.",
    "Executing the tests against this draft.",
    "Verifying correctness before submission.",
    "Let me actually run it — I want to see green before submitting.",
    "A quick test run to double-check.",
    "Let me run this through the tests.",
    "Running the test harness on this attempt.",
    "Kicking off the tests to confirm.",
    "Before finalizing, I'll run the test suite.",
]

CONNECTIVES_VERIFY_FIX = [
    "Re-running the tests to confirm the fix holds.",
    "Let me verify the correction actually solves it.",
    "Running the tests against the fixed version.",
    "Checking that the revised code passes.",
    "Let me confirm the patch works by re-executing.",
    "Re-executing to make sure the fix sticks.",
    "Running the tests one more time on the corrected implementation.",
    "Validating the correction with the same tests.",
    "Let me confirm we're green now.",
]

CONNECTIVES_RETRY_AFTER_FIX_FAILED = [
    "Still failing — my fix didn't address the full picture. Let me reconsider.",
    "That didn't resolve it. I missed something.",
    "The correction wasn't enough; the test is still unhappy. Let me look again.",
    "Still red. I need to rethink this.",
    "The patch didn't take. Let me dig deeper into what the traceback is telling me.",
    "Not yet — the same test is still failing. Let me look more carefully.",
]

CONFIRM_PASS_FALLBACK = [
    "All green — submitting.",
    "Tests pass. This is ready.",
    "Everything checks out. Submitting the solution.",
    "Confirmed working. Final answer.",
    "That's it — submitting.",
]


def tool_result_str(exec_result: dict) -> str:
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

    # Draft turn — always present.
    verify_lead = rng.choice(CONNECTIVES_VERIFY)
    before_exec = (voice.get("before_exec") or "").strip()
    if not before_exec or len(before_exec) > 300:
        before_exec = verify_lead

    first_text = (
        f"{voice['preamble'].strip()}\n\n"
        f"{code_block(trace['draft_code'])}\n\n"
        f"{before_exec}"
    )

    tool_args_draft = json.dumps({"code": trace["draft_code"], "test_code": test_code})
    draft_result_str = tool_result_str(trace["draft_exec"])

    messages = [
        {"role": "user", "content": description},
        {
            "role": "assistant",
            "content": first_text,
            "tool_calls": [{
                "type": "function",
                "function": {"name": "execute_code", "arguments": tool_args_draft},
            }],
        },
        {"role": "tool", "content": draft_result_str},
    ]

    if stratum == "no_revise":
        confirm = (voice.get("confirm") or "").strip() or rng.choice(CONFIRM_PASS_FALLBACK)
        messages.append({"role": "assistant", "content": confirm})
        return _wrap(trace, messages)

    # Revise path — one or more attempts.
    attempts = trace["attempts"]
    assert attempts, f"{trace['task_id']}: revise stratum with no attempts"

    for i, attempt in enumerate(attempts):
        diagnosis = (attempt.get("diagnosis") or "").strip()
        revision_pre = (attempt.get("revision_preamble") or "").strip() or "Here is the fix:"

        if i == 0:
            # First revision after draft failure
            lead = ""  # diagnosis already reacts to the failure
        else:
            # Retry after a failed fix attempt
            lead = rng.choice(CONNECTIVES_RETRY_AFTER_FIX_FAILED) + "\n\n"

        attempt_text = (
            f"{lead}{diagnosis}\n\n"
            f"{revision_pre}\n\n"
            f"{code_block(attempt['code'])}\n\n"
            f"{rng.choice(CONNECTIVES_VERIFY_FIX)}"
        )

        tool_args = json.dumps({"code": attempt["code"], "test_code": test_code})
        result_str = tool_result_str(attempt["exec"])

        messages.append({
            "role": "assistant",
            "content": attempt_text,
            "tool_calls": [{
                "type": "function",
                "function": {"name": "execute_code", "arguments": tool_args},
            }],
        })
        messages.append({"role": "tool", "content": result_str})

    confirm = (voice.get("confirm") or "").strip() or rng.choice(CONFIRM_PASS_FALLBACK)
    messages.append({"role": "assistant", "content": confirm})
    return _wrap(trace, messages)


def _wrap(trace: dict, messages: list) -> dict:
    return {
        "task_id": trace["task_id"],
        "stratum": trace["stratum"],
        "bug_class": trace.get("bug_class", "?"),
        "tools": [TOOL_SCHEMA],
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/training/review_traces_v3.json")
    ap.add_argument("--output", default="data/training/sft_review_v3.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    traces = json.loads(Path(args.traces).read_text())
    logger.info("Loaded %d traces", len(traces))

    rng = random.Random(args.seed)
    out = []
    from collections import Counter
    counts = Counter()
    for trace in traces:
        ex = stitch(trace, rng)
        out.append(ex)
        counts[ex["stratum"]] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("Stitched %d examples. Strata: %s -> %s",
                len(out), dict(counts), args.output)


if __name__ == "__main__":
    main()
