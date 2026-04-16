"""Experiment runner for hard benchmarks (slides, animations, code).

Unlike the HumanEval runner, this handles:
- HTML generation and browser rendering
- Multi-frame animation capture
- VLM-based visual review
- Structured quality scoring
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.proposer import ProposerAgent
from src.agents.reviewer import ReviewerAgent
from src.config import ExperimentConfig, ModelConfig, RESULTS_DIR
from src.utils.code_utils import extract_code
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SLIDE_SYSTEM_PROMPT = """\
You are an expert web developer creating presentation slides as single-page HTML files.

Rules:
- Output a COMPLETE, self-contained HTML file (<!DOCTYPE html> through </html>)
- Use inline CSS (no external stylesheets)
- The slide must render at 1920×1080 pixels with NO scrolling needed
- ALL text must be fully visible — no overflow, no truncation, no clipping
- Text elements must NOT overlap each other
- Maintain proper alignment and visual hierarchy
- Use appropriate font sizes (title: 36-48px, body: 16-24px, footnotes: 10-14px)
- Wrap your complete HTML in a ```html code block"""

ANIMATION_SYSTEM_PROMPT = """\
You are an expert web developer creating animations as single-page HTML files.

Rules:
- Output a COMPLETE, self-contained HTML file with embedded CSS and JavaScript
- No external dependencies (no CDN links, no imports)
- The animation should run in a 1920×1080 viewport
- Use requestAnimationFrame for smooth JavaScript animations
- CSS animations should use proper keyframes
- All visual elements must stay within the viewport bounds
- Wrap your complete HTML in a ```html code block"""

CODE_SYSTEM_PROMPT = """\
You are an expert Python programmer. Write correct, efficient solutions.

Rules:
- Output ONLY the function definition with any needed imports
- Handle ALL edge cases carefully
- Pay attention to boundary conditions, empty inputs, and special values
- Wrap your code in a ```python code block"""

VISUAL_REVIEW_PROMPT = """\
You are reviewing a rendered screenshot of a presentation slide or animation.
Carefully examine the image for these specific issues:

1. **Text overflow**: Any text that is cut off, extends beyond its container, or requires scrolling
2. **Text overlap**: Any text or elements that overlap each other, making content unreadable
3. **Alignment issues**: Elements that should be aligned but aren't (e.g., columns not lined up, text not centered when it should be)
4. **Readability**: Text too small to read, poor contrast, or unclear visual hierarchy
5. **Layout problems**: Missing elements, broken layouts, elements outside the visible area
6. **Content completeness**: All required content is present and visible

The original requirements were:
{requirements}

Respond with ONLY a JSON object:
{{
  "issues": [
    {{"description": "...", "severity": "critical|major|minor", "location": "top-left|center|etc"}}
  ],
  "quality_score": 0.0-1.0,
  "meets_requirements": true/false,
  "suggestions": ["specific fix 1", "specific fix 2"]
}}"""

REVISION_PROMPT = """\
Your previous HTML had visual problems when rendered. Here is the feedback:

## Previous HTML
```html
{previous_html}
```

## Visual Review Feedback
{feedback}

Fix ALL the identified issues. Pay special attention to:
- Text overflow and truncation
- Element overlap
- Alignment and spacing
- Content completeness

Output the COMPLETE fixed HTML in a ```html code block."""


@dataclass
class HardBenchmarkResult:
    """Result from a single hard benchmark problem."""

    task_id: str
    task_type: str  # "slide", "animation", "code"
    baseline: str
    iterations: int
    final_code: str
    quality_score: float  # 0-1, from VLM review
    issues_found: list[dict] = field(default_factory=list)
    meets_requirements: bool = False
    total_tokens: int = 0
    wall_time_seconds: float = 0
    screenshots: list[str] = field(default_factory=list)  # base64 encoded


class HardBenchmarkRunner:
    """Run hard benchmark experiments with visual/execution feedback."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = get_client()
        self._renderer = None

    @property
    def renderer(self):
        if self._renderer is None:
            from src.rendering.browser import BrowserRenderer
            self._renderer = BrowserRenderer()
        return self._renderer

    def run_slide_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
        reviewer_model: ModelConfig | None = None,
    ) -> HardBenchmarkResult:
        """Run a single slide generation task."""
        start_time = time.time()
        self.client.reset_counters()

        proposer = ProposerAgent(self.config.proposer_model)
        description = task["description"]
        requirements = "\n".join(f"- {r}" for r in task.get("requirements", []))

        # Initial generation
        prompt = f"{description}\n\nSpecific requirements:\n{requirements}"
        response = proposer.generate(prompt)
        current_html = extract_code(response.reasoning, "html")
        if not current_html.strip().startswith("<!") and not current_html.strip().startswith("<html"):
            current_html = response.reasoning  # fallback

        iteration = 1
        quality_score = 0.0
        issues = []
        meets_reqs = False
        screenshots_b64 = []

        for i in range(max_iterations):
            # Render
            try:
                screenshot_bytes = self.renderer.render_html(current_html)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                screenshots_b64.append(screenshot_b64)
            except Exception as e:
                logger.warning("Render failed for %s iter %d: %s", task["task_id"], i, e)
                break

            if not use_review:
                # Single-shot: just render, no review
                # Do a basic quality assessment
                quality_score, issues, meets_reqs = self._vlm_review(
                    screenshot_b64, requirements, reviewer_model
                )
                break

            # VLM Review
            quality_score, issues, meets_reqs = self._vlm_review(
                screenshot_b64, requirements, reviewer_model
            )

            if meets_reqs and quality_score >= 0.9:
                logger.info(
                    "Task %s: quality %.2f after %d iterations, stopping",
                    task["task_id"], quality_score, i + 1,
                )
                break

            if i < max_iterations - 1:
                # Revise
                feedback = self._format_review_feedback(issues, quality_score, meets_reqs)
                context = {
                    "previous_code": current_html,
                    "feedback": feedback,
                    "iteration": i + 1,
                }
                response = proposer.generate(prompt, context)
                new_html = extract_code(response.reasoning, "html")
                if new_html.strip():
                    current_html = new_html
                iteration = i + 2

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        result = HardBenchmarkResult(
            task_id=task["task_id"],
            task_type="slide",
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_html,
            quality_score=quality_score,
            issues_found=issues,
            meets_requirements=meets_reqs,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
            screenshots=screenshots_b64[-1:],  # Keep only final screenshot
        )

        logger.info(
            "Task %s [%s]: quality=%.2f, meets_reqs=%s, iters=%d, tokens=%d, time=%.1fs",
            task["task_id"],
            result.baseline,
            quality_score,
            meets_reqs,
            iteration,
            usage["total_tokens"],
            elapsed,
        )
        return result

    def run_animation_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
        reviewer_model: ModelConfig | None = None,
    ) -> HardBenchmarkResult:
        """Run a single animation task with multi-frame capture."""
        start_time = time.time()
        self.client.reset_counters()

        proposer = ProposerAgent(self.config.proposer_model)
        prompt = task["description"]
        requirements = "\n".join(f"- {r}" for r in task.get("requirements", []))
        frame_times = task.get("frame_times_ms", [0, 1000, 2000, 3000])

        # Use animation system prompt
        response = self.client.generate(
            config=self.config.proposer_model,
            system=ANIMATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{prompt}\n\nRequirements:\n{requirements}"}],
        )
        current_html = extract_code(response.content, "html")

        iteration = 1
        quality_score = 0.0
        issues = []
        meets_reqs = False
        screenshots_b64 = []

        for i in range(max_iterations):
            # Render multiple frames
            try:
                frames = self.renderer.render_animation_frames(
                    current_html, frame_times
                )
                frame_b64s = [base64.b64encode(f).decode() for f in frames]
                screenshots_b64 = frame_b64s
            except Exception as e:
                logger.warning("Animation render failed: %s", e)
                break

            # VLM review of frames
            quality_score, issues, meets_reqs = self._vlm_review_animation(
                frame_b64s, requirements, frame_times, reviewer_model
            )

            if not use_review:
                break

            if meets_reqs and quality_score >= 0.9:
                break

            if i < max_iterations - 1:
                feedback = self._format_review_feedback(issues, quality_score, meets_reqs)
                revision_response = self.client.generate(
                    config=self.config.proposer_model,
                    system=ANIMATION_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"{prompt}\n\nRequirements:\n{requirements}"},
                        {"role": "assistant", "content": f"```html\n{current_html}\n```"},
                        {"role": "user", "content": REVISION_PROMPT.format(
                            previous_html=current_html, feedback=feedback
                        )},
                    ],
                )
                new_html = extract_code(revision_response.content, "html")
                if new_html.strip():
                    current_html = new_html
                iteration = i + 2

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        return HardBenchmarkResult(
            task_id=task["task_id"],
            task_type="animation",
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_html,
            quality_score=quality_score,
            issues_found=issues,
            meets_requirements=meets_reqs,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
            screenshots=screenshots_b64[-1:],
        )

    def run_code_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
    ) -> HardBenchmarkResult:
        """Run a single hard coding task with execution feedback."""
        from src.evaluation.code_eval import CodeEvaluator
        from src.feedback.type3a_execution import ExecutionFeedback

        start_time = time.time()
        self.client.reset_counters()

        proposer = ProposerAgent(self.config.proposer_model)
        evaluator = CodeEvaluator()
        exec_feedback = ExecutionFeedback()

        description = task["description"]
        test_code = task["test_code"]

        response = proposer.generate(description)
        current_code = response.code
        iteration = 1
        passed = False

        for i in range(max_iterations):
            # Execute
            eval_result = evaluator.evaluate(current_code, test_code)
            passed = eval_result.passed

            if passed:
                break

            if not use_review or i >= max_iterations - 1:
                break

            # Revise with execution feedback
            feedback = f"FAILED: {eval_result.error_message}\n"
            if eval_result.stderr:
                feedback += f"stderr:\n{eval_result.stderr[:2000]}\n"
            if eval_result.stdout:
                feedback += f"stdout:\n{eval_result.stdout[:1000]}\n"

            context = {
                "previous_code": current_code,
                "feedback": feedback,
                "iteration": i + 1,
            }
            response = proposer.generate(description, context)
            current_code = response.code
            iteration = i + 2

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        return HardBenchmarkResult(
            task_id=task["task_id"],
            task_type="code",
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_code,
            quality_score=1.0 if passed else 0.0,
            meets_requirements=passed,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
        )

    # -------------------------------------------------------------------
    # VLM Review helpers
    # -------------------------------------------------------------------

    def _vlm_review(
        self,
        screenshot_b64: str,
        requirements: str,
        model_config: ModelConfig | None = None,
    ) -> tuple[float, list[dict], bool]:
        """Use VLM to review a single screenshot."""
        config = model_config or ModelConfig.claude_sonnet()
        prompt = VISUAL_REVIEW_PROMPT.format(requirements=requirements)

        response = self.client.generate(
            config=config,
            system="You are a visual quality reviewer. Respond with ONLY valid JSON.",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        return self._parse_review_json(response.content)

    def _vlm_review_animation(
        self,
        frame_b64s: list[str],
        requirements: str,
        frame_times: list[int],
        model_config: ModelConfig | None = None,
    ) -> tuple[float, list[dict], bool]:
        """Use VLM to review animation frames."""
        config = model_config or ModelConfig.claude_sonnet()

        # Build multi-image message
        content = []
        for i, (frame_b64, t) in enumerate(zip(frame_b64s, frame_times)):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": frame_b64,
                },
            })
            content.append({
                "type": "text",
                "text": f"Frame at t={t}ms",
            })

        review_prompt = (
            f"These are frames captured from an animation at different timestamps.\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"Check for:\n"
            f"1. Visual glitches or rendering artifacts\n"
            f"2. Elements going out of bounds\n"
            f"3. Animation timing issues (too fast, too slow, jerky)\n"
            f"4. Missing or incorrect visual elements\n"
            f"5. Temporal coherence (smooth transitions between frames)\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"issues": [{{"description": "...", "severity": "critical|major|minor", '
            f'"frame": "t=Xms"}}], "quality_score": 0.0-1.0, '
            f'"meets_requirements": true/false, "suggestions": ["..."]}}'
        )
        content.append({"type": "text", "text": review_prompt})

        response = self.client.generate(
            config=config,
            system="You are a visual quality reviewer for animations. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_review_json(response.content)

    @staticmethod
    def _parse_review_json(
        raw: str,
    ) -> tuple[float, list[dict], bool]:
        """Parse VLM review JSON response."""
        raw = raw.strip()
        if raw.startswith("```"):
            first_nl = raw.index("\n")
            raw = raw[first_nl + 1:]
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    return 0.5, [], False
            else:
                return 0.5, [], False

        return (
            float(data.get("quality_score", 0.5)),
            data.get("issues", []),
            data.get("meets_requirements", False),
        )

    @staticmethod
    def _format_review_feedback(
        issues: list[dict], quality_score: float, meets_reqs: bool
    ) -> str:
        """Format review results as feedback for the proposer."""
        parts = [f"Quality score: {quality_score:.0%}"]
        if not meets_reqs:
            parts.append("DOES NOT MEET REQUIREMENTS")
        if issues:
            parts.append("\nIssues found:")
            for i, issue in enumerate(issues, 1):
                sev = issue.get("severity", "unknown").upper()
                desc = issue.get("description", "")
                loc = issue.get("location", issue.get("frame", ""))
                parts.append(f"  {i}. [{sev}] {desc}" + (f" (at {loc})" if loc else ""))
        suggestions = issues[0].get("suggestions", []) if issues else []
        if not suggestions:
            # Try to get from top-level parsed data
            pass
        return "\n".join(parts)


def run_hard_benchmarks(
    task_types: list[str] | None = None,
    num_tasks: int | None = None,
    max_iterations: int = 3,
    output_dir: Path | None = None,
) -> dict:
    """Run hard benchmark experiments."""
    task_types = task_types or ["slides", "animations", "code"]
    output_dir = output_dir or RESULTS_DIR / "hard_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        name="hard_benchmarks",
        benchmark="hard",
        budget_tokens=500_000,
        output_dir=output_dir,
    )
    runner = HardBenchmarkRunner(config)
    all_results = {}

    for task_type in task_types:
        task_file = Path(f"data/hard_benchmarks/{task_type}/{task_type.rstrip('s')}_tasks.json")
        if task_type == "code":
            task_file = Path("data/hard_benchmarks/code/code_tasks.json")

        if not task_file.exists():
            logger.warning("Task file not found: %s", task_file)
            continue

        with open(task_file) as f:
            tasks = json.load(f)

        if num_tasks:
            tasks = tasks[:num_tasks]

        logger.info("Running %d %s tasks", len(tasks), task_type)

        for task in tasks:
            task_id = task["task_id"]

            # Single-shot (no review)
            if task_type == "slides":
                r1 = runner.run_slide_task(task, use_review=False, max_iterations=1)
            elif task_type == "animations":
                r1 = runner.run_animation_task(task, use_review=False, max_iterations=1)
            elif task_type == "code":
                r1 = runner.run_code_task(task, use_review=False, max_iterations=1)
            else:
                continue

            # With review
            if task_type == "slides":
                r2 = runner.run_slide_task(task, use_review=True, max_iterations=max_iterations)
            elif task_type == "animations":
                r2 = runner.run_animation_task(task, use_review=True, max_iterations=max_iterations)
            elif task_type == "code":
                r2 = runner.run_code_task(task, use_review=True, max_iterations=max_iterations)
            else:
                continue

            all_results[f"{task_id}_single_shot"] = r1
            all_results[f"{task_id}_reviewed"] = r2

            logger.info(
                "  %s: single_shot=%.2f, reviewed=%.2f (delta=+%.2f)",
                task_id,
                r1.quality_score,
                r2.quality_score,
                r2.quality_score - r1.quality_score,
            )

    # Save results (without screenshots to keep file size reasonable)
    save_data = {}
    for key, r in all_results.items():
        save_data[key] = {
            "task_id": r.task_id,
            "task_type": r.task_type,
            "baseline": r.baseline,
            "iterations": r.iterations,
            "quality_score": r.quality_score,
            "issues_found": r.issues_found,
            "meets_requirements": r.meets_requirements,
            "total_tokens": r.total_tokens,
            "wall_time_seconds": r.wall_time_seconds,
        }

    with open(output_dir / "hard_benchmark_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Print summary
    _print_summary(all_results)

    return all_results


def _print_summary(results: dict):
    """Print a summary table of hard benchmark results."""
    from collections import defaultdict

    by_type_baseline = defaultdict(list)
    for r in results.values():
        by_type_baseline[(r.task_type, r.baseline)].append(r)

    print("\n" + "=" * 80)
    print("Hard Benchmark Results")
    print("=" * 80)
    print(f"{'Type':<12} {'Baseline':<20} {'N':>4} {'Avg Quality':>12} {'Meets Reqs':>12} {'Avg Tokens':>12}")
    print("-" * 80)

    for (task_type, baseline), items in sorted(by_type_baseline.items()):
        n = len(items)
        avg_q = sum(r.quality_score for r in items) / n
        meets = sum(1 for r in items if r.meets_requirements) / n
        avg_tok = sum(r.total_tokens for r in items) / n
        print(f"{task_type:<12} {baseline:<20} {n:>4} {avg_q:>12.2f} {meets:>11.0%} {avg_tok:>12.0f}")

    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_hard_benchmarks(num_tasks=3, max_iterations=3)
