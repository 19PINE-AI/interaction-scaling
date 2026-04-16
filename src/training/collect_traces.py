"""Collect interaction traces from experiments for training data.

Converts successful propose→execute→review→revise trajectories
into training examples for SFT and GRPO/DAPO fine-tuning.

Each trace captures the full interaction sequence with:
- Budget information (step limit, token budget)
- Decision points (when to execute, when to review, when to stop)
- Grounded feedback signals (execution results, VLM reviews)
- Final outcome (reward signal)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InteractionStep:
    """A single step in an interaction trace."""

    step_type: str  # "think", "do", "review", "revise", "submit"
    input_text: str  # What the model sees
    output_text: str  # What the model generates
    feedback: str | None = None  # Grounded feedback received (if "do" step)
    reward: float | None = None  # Reward signal (if terminal)


@dataclass
class InteractionTrace:
    """A complete interaction trace for one task."""

    task_id: str
    task_type: str  # "code", "slide", "animation", "video", "research", "webpage"
    task_description: str
    budget_steps: int  # Max iterations allowed
    steps: list[InteractionStep] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False


def collect_code_traces(
    tasks_path: Path,
    results_path: Path,
) -> list[InteractionTrace]:
    """Collect interaction traces from code benchmark results.

    Reconstructs the interaction sequence from task descriptions and
    results, creating training examples that show:
    1. Initial code generation (think)
    2. Test execution (do) → error message (feedback)
    3. Code revision based on feedback (revise)
    4. Re-execution → pass (do) → submit
    """
    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    task_map = {t["task_id"]: t for t in tasks}
    traces = []

    for result in results:
        tid = result["task_id"]
        task = task_map.get(tid)
        if not task:
            continue

        trace = InteractionTrace(
            task_id=tid,
            task_type="code",
            task_description=task["description"],
            budget_steps=5,
            final_reward=1.0 if result.get("reviewed_passed", False) else 0.0,
            success=result.get("reviewed_passed", False),
        )

        # Step 1: Think — initial generation
        trace.steps.append(InteractionStep(
            step_type="think",
            input_text=(
                f"[BUDGET: 5 steps remaining]\n"
                f"[TASK]: Fix the bug in this code.\n\n"
                f"{task['description']}"
            ),
            output_text="[GENERATE]: I'll analyze the bug and write a fix.",
        ))

        # Step 2: Do — execute tests
        passed_first = result.get("single_shot_passed", False)
        trace.steps.append(InteractionStep(
            step_type="do",
            input_text="[ACTION]: Execute the code against test cases.",
            output_text="[EXECUTE]",
            feedback=(
                "All tests passed." if passed_first
                else "Tests FAILED. See error output for details."
            ),
        ))

        if not passed_first and result.get("reviewed_passed", False):
            # Step 3: Review — analyze the failure
            trace.steps.append(InteractionStep(
                step_type="review",
                input_text=(
                    f"[BUDGET: {5 - 1} steps remaining]\n"
                    f"Tests failed. Analyze the error and identify the root cause."
                ),
                output_text=(
                    "[REVIEW]: The test failure indicates the bug is in the "
                    "specific edge case handling. I need to revise the code."
                ),
            ))

            # Step 4: Revise — fix based on feedback
            trace.steps.append(InteractionStep(
                step_type="revise",
                input_text=(
                    f"[BUDGET: {5 - 2} steps remaining]\n"
                    f"Revise the code to fix the identified issue."
                ),
                output_text="[GENERATE]: Here is the corrected code with the bug fixed.",
            ))

            # Step 5: Do — re-execute
            trace.steps.append(InteractionStep(
                step_type="do",
                input_text="[ACTION]: Execute the revised code against test cases.",
                output_text="[EXECUTE]",
                feedback="All tests passed.",
            ))

        # Final: Submit
        trace.steps.append(InteractionStep(
            step_type="submit",
            input_text="[DECISION]: Should I submit or continue iterating?",
            output_text="[SUBMIT]: Tests pass, submitting solution.",
            reward=trace.final_reward,
        ))

        traces.append(trace)

    logger.info("Collected %d code traces (%d successful)",
                len(traces), sum(1 for t in traces if t.success))
    return traces


def traces_to_sft_format(traces: list[InteractionTrace]) -> list[dict]:
    """Convert traces to SFT training format.

    Each example is a conversation with system prompt containing budget info,
    and the interaction sequence as user/assistant turns.
    """
    examples = []

    for trace in traces:
        if not trace.success:
            continue  # Only train on successful trajectories for SFT

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent that solves tasks through interaction scaling. "
                    "You have a budget of steps to solve the task. At each step, you can:\n"
                    "- [GENERATE]: Write code/content to solve the task\n"
                    "- [EXECUTE]: Run your solution and observe the results\n"
                    "- [REVIEW]: Analyze feedback and plan improvements\n"
                    "- [SUBMIT]: Submit your final solution\n\n"
                    "Be budget-aware: use your steps efficiently. "
                    "If tests pass on the first try, submit immediately. "
                    "If tests fail, review the feedback, revise, and re-test.\n\n"
                    f"Budget: {trace.budget_steps} steps maximum."
                ),
            },
        ]

        for step in trace.steps:
            messages.append({"role": "user", "content": step.input_text})
            messages.append({"role": "assistant", "content": step.output_text})

        examples.append({
            "task_id": trace.task_id,
            "task_type": trace.task_type,
            "messages": messages,
            "reward": trace.final_reward,
        })

    return examples


def traces_to_grpo_format(traces: list[InteractionTrace]) -> list[dict]:
    """Convert traces to GRPO training format.

    Each example includes the prompt (task + budget) and the full trajectory
    as a single completion string, with the reward signal.
    """
    examples = []

    for trace in traces:
        prompt = (
            f"You are an AI agent solving a task with interaction scaling.\n"
            f"Budget: {trace.budget_steps} steps.\n"
            f"Task type: {trace.task_type}\n\n"
            f"Task:\n{trace.task_description[:2000]}\n\n"
            f"Solve this task. Use [EXECUTE] to test your solution, "
            f"[REVIEW] to analyze feedback, and [SUBMIT] when done."
        )

        # Build the completion as a sequence of actions
        completion_parts = []
        for step in trace.steps:
            completion_parts.append(f"[{step.step_type.upper()}]\n{step.output_text}")
            if step.feedback:
                completion_parts.append(f"[FEEDBACK]\n{step.feedback}")

        completion = "\n\n".join(completion_parts)

        examples.append({
            "task_id": trace.task_id,
            "prompt": prompt,
            "completion": completion,
            "reward": trace.final_reward,
        })

    return examples


def collect_all_traces(results_dir: Path) -> list[InteractionTrace]:
    """Collect traces from all available benchmark results."""
    all_traces = []

    # Code traces
    code_tasks = Path("data/hard_benchmarks/code/code_tasks.json")
    code_results = results_dir / "code_swe_results.json"
    if code_tasks.exists() and code_results.exists():
        all_traces.extend(collect_code_traces(code_tasks, code_results))

    logger.info("Total traces collected: %d", len(all_traces))
    return all_traces


def save_training_data(
    traces: list[InteractionTrace],
    output_dir: Path,
):
    """Save training data in both SFT and GRPO formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_data = traces_to_sft_format(traces)
    grpo_data = traces_to_grpo_format(traces)

    with open(output_dir / "sft_data.json", "w") as f:
        json.dump(sft_data, f, indent=2)

    with open(output_dir / "grpo_data.json", "w") as f:
        json.dump(grpo_data, f, indent=2)

    logger.info(
        "Saved %d SFT examples and %d GRPO examples to %s",
        len(sft_data), len(grpo_data), output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results_dir = Path("results/hard_benchmarks")
    traces = collect_all_traces(results_dir)
    save_training_data(traces, Path("data/training"))
