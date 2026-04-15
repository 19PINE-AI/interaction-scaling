"""Experiment 5: Verification Gap Analysis.

Measures the gap between generating correct solutions (pass@K)
and selecting them (choose@K) with different selection methods:
- Self-selection (Type 0)
- Cross-model selection (Type 1)
- Grounded reviewer selection (Type 3)
"""

import json
import logging
from pathlib import Path

from src.agents.proposer import ProposerAgent
from src.agents.reviewer import ReviewerAgent
from src.benchmarks.humaneval import HumanEvalBenchmark
from src.benchmarks.mbpp import MBPPBenchmark
from src.config import ExperimentConfig, ModelConfig, RESULTS_DIR
from src.evaluation.code_eval import CodeEvaluator
from src.feedback.type0_self import SelfReviewFeedback
from src.feedback.type1_cross import CrossModelFeedback
from src.feedback.type3a_execution import ExecutionFeedback
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)


def run_exp5(
    benchmark_name: str = "humaneval",
    num_problems: int | None = None,
    num_candidates: int = 5,
    output_dir: Path | None = None,
) -> dict:
    """Run Experiment 5: Verification Gap Analysis."""
    output_dir = output_dir or RESULTS_DIR / "exp5_verification_gap"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    if benchmark_name == "humaneval":
        benchmark = HumanEvalBenchmark()
    elif benchmark_name == "mbpp":
        benchmark = MBPPBenchmark()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    problems = benchmark.load()
    if num_problems:
        problems = problems[:num_problems]

    logger.info(
        "Exp5: Verification gap on %d %s problems with K=%d candidates",
        len(problems),
        benchmark_name,
        num_candidates,
    )

    # Create agents
    sampling_model = ModelConfig(
        provider=ModelConfig.claude_sonnet().provider,
        model_id=ModelConfig.claude_sonnet().model_id,
        max_tokens=ModelConfig.claude_sonnet().max_tokens,
        temperature=0.8,
    )
    proposer = ProposerAgent(sampling_model)
    evaluator = CodeEvaluator()
    client = get_client()

    # Feedback providers for selection
    self_reviewer = SelfReviewFeedback()
    cross_reviewer = CrossModelFeedback()
    exec_feedback = ExecutionFeedback()

    results = []

    for i, problem in enumerate(problems):
        logger.info("[%d/%d] %s", i + 1, len(problems), problem.task_id)

        # Generate K candidates
        candidates = []
        for k in range(num_candidates):
            response = proposer.generate(problem.prompt)
            eval_result = evaluator.evaluate(response.code, problem.test_code)
            candidates.append({
                "code": response.code,
                "passed": eval_result.passed,
                "error": eval_result.error_message,
            })

        num_correct = sum(1 for c in candidates if c["passed"])
        pass_at_k = 1.0 if num_correct > 0 else 0.0

        problem_dict = {
            "test_code": problem.test_code,
            "entry_point": problem.entry_point,
            "prompt": problem.prompt,
        }

        # Selection method 1: Self-review (Type 0)
        self_scores = []
        for c in candidates:
            fb = self_reviewer.get_feedback(c["code"], problem_dict)
            # Use the review content length as a rough score proxy
            # (more issues = lower score)
            score = 1.0 if "no issues" in fb.content.lower() else 0.5
            self_scores.append(score)
        self_best_idx = max(range(len(candidates)), key=lambda i: self_scores[i])
        self_choose = candidates[self_best_idx]["passed"]

        # Selection method 2: Cross-model (Type 1)
        cross_scores = []
        for c in candidates:
            fb = cross_reviewer.get_feedback(c["code"], problem_dict)
            score = 1.0 if "no issues" in fb.content.lower() else 0.5
            cross_scores.append(score)
        cross_best_idx = max(range(len(candidates)), key=lambda i: cross_scores[i])
        cross_choose = candidates[cross_best_idx]["passed"]

        # Selection method 3: Grounded execution (Type 3)
        exec_scores = []
        for c in candidates:
            fb = exec_feedback.get_feedback(c["code"], problem_dict)
            score = 1.0 if fb.structured_data.get("passed", False) else 0.0
            exec_scores.append(score)
        exec_best_idx = max(range(len(candidates)), key=lambda i: exec_scores[i])
        exec_choose = candidates[exec_best_idx]["passed"]

        record = {
            "problem_id": problem.task_id,
            "num_candidates": num_candidates,
            "num_correct": num_correct,
            "pass_at_k": pass_at_k,
            "choose_self": self_choose,
            "choose_cross": cross_choose,
            "choose_exec": exec_choose,
        }
        results.append(record)

        logger.info(
            "  pass@K=%s, choose(self)=%s, choose(cross)=%s, choose(exec)=%s",
            pass_at_k,
            self_choose,
            cross_choose,
            exec_choose,
        )

    # Compute aggregate metrics
    n = len(results)
    summary = {
        "num_problems": n,
        "num_candidates": num_candidates,
        "pass_at_k": sum(r["pass_at_k"] for r in results) / n if n else 0,
        "choose_at_k_self": sum(r["choose_self"] for r in results) / n if n else 0,
        "choose_at_k_cross": sum(r["choose_cross"] for r in results) / n if n else 0,
        "choose_at_k_exec": sum(r["choose_exec"] for r in results) / n if n else 0,
    }
    summary["verification_gap_self"] = summary["pass_at_k"] - summary["choose_at_k_self"]
    summary["verification_gap_cross"] = summary["pass_at_k"] - summary["choose_at_k_cross"]
    summary["verification_gap_exec"] = summary["pass_at_k"] - summary["choose_at_k_exec"]

    output = {"results": results, "summary": summary}
    output_path = output_dir / f"exp5_{benchmark_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Exp5 summary: %s", json.dumps(summary, indent=2))
    logger.info("Results saved to %s", output_path)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_exp5(num_problems=10, num_candidates=3)
