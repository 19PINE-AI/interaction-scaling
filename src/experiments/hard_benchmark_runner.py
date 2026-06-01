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

# The four core graphic-design principles (Robin Williams, "The Non-Designer's
# Design Book"): Proximity, Alignment, Repetition, Contrast, plus deliberate
# color. Applied identically in the generation prompt, the reviewer feedback
# prompt, and the rubric judge so the artifact is produced, critiqued, and
# scored against the same design criteria.
DESIGN_PRINCIPLES = """\
Apply the four core design principles throughout:
- PROXIMITY: group related items into a single visual unit with tight
  internal spacing, and separate unrelated groups with generous whitespace.
  Related things belong together; unrelated things must be visibly apart.
- ALIGNMENT: every element shares a strong edge or centerline with
  another — nothing is placed arbitrarily. Side-by-side panels/pillars/cards
  must be equal width, share top and bottom edges, and have uniform gutters;
  columns and rows of boxes line up exactly.
- REPETITION: reuse a consistent visual system — the same heading style,
  body font, accent color, box style, corner radius, bullet glyph, and spacing
  unit everywhere. Repetition unifies the slide/page.
- CONTRAST: if two elements differ in role, make them strongly different
  (size, weight, color, structure) to build a clear hierarchy and guide the eye
  — avoid timid, near-identical styling. The title must dominate.
- COLOR: choose one deliberate, harmonious palette (complementary, triadic, or
  analogous) with one accent color used consistently, and keep strong
  text/background contrast for legibility."""

SLIDE_SYSTEM_PROMPT = """\
You are an expert presentation designer creating slides as single-page HTML files.

Rules:
- Output a COMPLETE, self-contained HTML file (<!DOCTYPE html> through </html>)
- Use inline CSS (no external stylesheets)
- The slide must render at 1920×1080 pixels with NO scrolling needed
- ALL text must be fully visible — no overflow, no truncation, no clipping
- Text elements must NOT overlap each other
- Use appropriate font sizes (title: 36-48px, body: 16-24px, footnotes: 10-14px)

%s

- Wrap your complete HTML in a ```html code block""" % DESIGN_PRINCIPLES

WEBPAGE_SYSTEM_PROMPT = """\
You are an expert web designer creating webpages as single-page HTML files.

Rules:
- Output a COMPLETE, self-contained HTML file (<!DOCTYPE html> through </html>)
- Use inline CSS (no external stylesheets)
- The page is rendered in a 1920px-wide viewport; vertical scrolling is permitted
- ALL text must be readable — no unintended overlap, clipping, or truncation
- Preserve the requested sections and content
- Use appropriate font sizes and spacing for a desktop landing page

%s

- Wrap your complete HTML in a ```html code block""" % DESIGN_PRINCIPLES

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
You are reviewing a rendered screenshot of a presentation slide or web page.
Carefully examine the image for these specific issues:

1. **Text overflow**: Any text that is cut off, extends beyond its container, or requires scrolling
2. **Text overlap**: Any text or elements that overlap each other, making content unreadable
3. **Alignment**: Elements that should share an edge or centerline but don't — side-by-side panels/pillars/cards of unequal width, misaligned top/bottom edges, uneven gutters, columns/rows that don't line up
4. **Proximity**: Related items not grouped (scattered, ambiguous spacing) or unrelated groups crowded together with no separating whitespace
5. **Repetition**: Inconsistent visual system — heading/body fonts, accent color, box style, corner radius, bullet glyph, or spacing that varies where it should repeat
6. **Contrast**: Weak hierarchy — title not dominant, elements of different roles styled too similarly, or poor text/background color contrast
7. **Readability & layout**: Text too small, missing elements, broken layout, elements outside the visible area
8. **Content completeness**: All required content is present and visible

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
Your previous HTML (shown in the prior turn) had issues when rendered.

## Visual Review Feedback
{feedback}

Apply the SMALLEST edit that fixes only the identified issues.
- Preserve working layout, colors, typography, and content that was NOT mentioned in the feedback
- Do not refactor, reorganize, or restyle unrelated code
- Change only the specific elements called out above

Output the complete updated HTML in a ```html code block."""


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
    full_output: str = ""  # Full LLM output including thinking tokens
    intermediate_outputs: list[str] = field(default_factory=list)  # All revision outputs
    interaction_trace: list[dict] = field(default_factory=list)  # Full input/output pairs per step


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
        task_type: str = "slide",
    ) -> HardBenchmarkResult:
        """Run a single slide/webpage generation task.

        ``task_type`` selects the system prompt and result labeling:
        - "slide":   fixed 1920×1080, no scroll (SLIDE_SYSTEM_PROMPT)
        - "webpage": 1920px wide, scroll permitted (WEBPAGE_SYSTEM_PROMPT)
        """
        start_time = time.time()
        self.client.reset_counters()

        system_prompt = (
            WEBPAGE_SYSTEM_PROMPT if task_type == "webpage" else SLIDE_SYSTEM_PROMPT
        )

        description = task["description"]
        requirements = "\n".join(f"- {r}" for r in task.get("requirements", []))
        prompt = f"{description}\n\nSpecific requirements:\n{requirements}"

        init_messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(
            config=self.config.proposer_model,
            system=system_prompt,
            messages=init_messages,
        )
        current_html = extract_code(response.content, "html")
        if not current_html.strip().startswith("<!") and not current_html.strip().startswith("<html"):
            current_html = response.content

        trace_steps = [{
            "step": "generate",
            "system": system_prompt,
            "messages": init_messages,
            "full_output": response.content,
        }]

        iteration = 1
        quality_score = 0.0
        issues = []
        meets_reqs = False
        screenshots_b64 = []
        best = None  # keep-best: (quality, html, issues, meets, screenshot_b64)
        prior_reviews: list[dict] = []

        for i in range(max_iterations):
            try:
                screenshot_bytes = self.renderer.render_html(current_html)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                screenshots_b64.append(screenshot_b64)
            except Exception as e:
                logger.warning("Render failed for %s iter %d: %s", task["task_id"], i, e)
                break

            if not use_review:
                quality_score, issues, meets_reqs, _suggestions = self._vlm_review(
                    screenshot_b64, requirements, reviewer_model
                )
                if best is None or quality_score > best[0]:
                    best = (quality_score, current_html, issues, meets_reqs, screenshot_b64)
                break

            quality_score, issues, meets_reqs, suggestions = self._vlm_review(
                screenshot_b64, requirements, reviewer_model,
                prior_reviews=prior_reviews or None,
            )
            prior_reviews.append({"quality_score": quality_score, "issues": issues})

            if best is None or quality_score > best[0]:
                best = (quality_score, current_html, issues, meets_reqs, screenshot_b64)

            if quality_score >= 0.9:
                logger.info(
                    "Task %s: quality %.2f after %d iterations, stopping",
                    task["task_id"], quality_score, i + 1,
                )
                break

            if i < max_iterations - 1:
                feedback = self._format_review_feedback(
                    issues, quality_score, meets_reqs, suggestions
                )
                revision_msg = REVISION_PROMPT.format(feedback=feedback)
                revision_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```html\n{current_html}\n```"},
                    {"role": "user", "content": revision_msg},
                ]
                response = self.client.generate(
                    config=self.config.proposer_model,
                    system=system_prompt,
                    messages=revision_messages,
                )
                trace_steps.append({
                    "step": "revise",
                    "system": system_prompt,
                    "feedback": feedback,
                    "messages_text": revision_messages,
                    "full_output": response.content,
                })
                new_html = extract_code(response.content, "html")
                if new_html.strip():
                    current_html = new_html
                iteration = i + 2

        # keep-best: return the highest-scoring iteration, not the last
        if best is not None:
            quality_score, current_html, issues, meets_reqs, final_screenshot = best
            if screenshots_b64 and final_screenshot != screenshots_b64[-1]:
                screenshots_b64.append(final_screenshot)

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        result = HardBenchmarkResult(
            task_id=task["task_id"],
            task_type=task_type,
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_html,
            quality_score=quality_score,
            issues_found=issues,
            meets_requirements=meets_reqs,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
            screenshots=screenshots_b64[-1:],
            full_output=trace_steps[0]["full_output"],
            intermediate_outputs=[s["full_output"] for s in trace_steps[1:]],
            interaction_trace=trace_steps,
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

    def run_webpage_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
        reviewer_model: ModelConfig | None = None,
    ) -> HardBenchmarkResult:
        """Webpage variant — uses WEBPAGE_SYSTEM_PROMPT (scrollable pages)."""
        return self.run_slide_task(
            task,
            use_review=use_review,
            max_iterations=max_iterations,
            reviewer_model=reviewer_model,
            task_type="webpage",
        )

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
        full_prompt = f"{prompt}\n\nRequirements:\n{requirements}"
        init_messages = [{"role": "user", "content": full_prompt}]
        response = self.client.generate(
            config=self.config.proposer_model,
            system=ANIMATION_SYSTEM_PROMPT,
            messages=init_messages,
        )
        current_html = extract_code(response.content, "html")

        trace_steps = [{
            "step": "generate",
            "system": ANIMATION_SYSTEM_PROMPT,
            "messages": init_messages,
            "full_output": response.content,
        }]

        iteration = 1
        quality_score = 0.0
        issues = []
        meets_reqs = False
        screenshots_b64: list[str] = []
        best = None  # (quality, html, issues, meets, frames)
        prior_reviews: list[dict] = []

        for i in range(max_iterations):
            try:
                frames = self.renderer.render_animation_frames(
                    current_html, frame_times
                )
                frame_b64s = [base64.b64encode(f).decode() for f in frames]
                screenshots_b64 = frame_b64s
            except Exception as e:
                logger.warning("Animation render failed: %s", e)
                break

            quality_score, issues, meets_reqs, suggestions = self._vlm_review_animation(
                frame_b64s, requirements, frame_times, reviewer_model,
                prior_reviews=prior_reviews or None,
            )
            prior_reviews.append({"quality_score": quality_score, "issues": issues})

            if best is None or quality_score > best[0]:
                best = (quality_score, current_html, issues, meets_reqs, frame_b64s)

            if not use_review:
                break

            if quality_score >= 0.9:
                break

            if i < max_iterations - 1:
                feedback = self._format_review_feedback(
                    issues, quality_score, meets_reqs, suggestions
                )
                revision_msg = REVISION_PROMPT.format(feedback=feedback)
                revision_messages = [
                    {"role": "user", "content": full_prompt},
                    {"role": "assistant", "content": f"```html\n{current_html}\n```"},
                    {"role": "user", "content": revision_msg},
                ]
                revision_response = self.client.generate(
                    config=self.config.proposer_model,
                    system=ANIMATION_SYSTEM_PROMPT,
                    messages=revision_messages,
                )
                trace_steps.append({
                    "step": "revise",
                    "system": ANIMATION_SYSTEM_PROMPT,
                    "feedback": feedback,
                    "messages_text": revision_messages,
                    "full_output": revision_response.content,
                })
                new_html = extract_code(revision_response.content, "html")
                if new_html.strip():
                    current_html = new_html
                iteration = i + 2

        if best is not None:
            quality_score, current_html, issues, meets_reqs, screenshots_b64 = best

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
            screenshots=screenshots_b64[-1:] if screenshots_b64 else [],
            full_output=trace_steps[0]["full_output"],
            intermediate_outputs=[s["full_output"] for s in trace_steps[1:]],
            interaction_trace=trace_steps,
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

        evaluator = CodeEvaluator()
        exec_feedback = ExecutionFeedback()

        description = task["description"]
        test_code = task["test_code"]

        # Use LLM client directly (instead of ProposerAgent) to capture full output
        init_messages = [{"role": "user", "content": description}]
        response = self.client.generate(
            config=self.config.proposer_model,
            system=CODE_SYSTEM_PROMPT,
            messages=init_messages,
        )
        current_code = extract_code(response.content, "python")

        trace_steps = [{
            "step": "generate",
            "system": CODE_SYSTEM_PROMPT,
            "messages": init_messages,
            "full_output": response.content,
        }]

        iteration = 1
        passed = False
        best_code = current_code
        best_passed = False
        prior_errors: list[str] = []

        for i in range(max_iterations):
            eval_result = evaluator.evaluate(current_code, test_code)
            passed = eval_result.passed

            if passed and not best_passed:
                best_code, best_passed = current_code, True

            if passed:
                break

            if not use_review or i >= max_iterations - 1:
                break

            feedback = f"FAILED: {eval_result.error_message}\n"
            if eval_result.stderr:
                feedback += f"stderr:\n{eval_result.stderr[:2000]}\n"
            if eval_result.stdout:
                feedback += f"stdout:\n{eval_result.stdout[:1000]}\n"

            prior_note = ""
            if prior_errors:
                prior_note = (
                    "\n\nPrior failing errors (if the same error reappears, change your approach "
                    "— the previous fix did not address the root cause):\n"
                    + "\n".join(f"- {e}" for e in prior_errors)
                )

            revision_messages = [
                {"role": "user", "content": description},
                {"role": "assistant", "content": f"```python\n{current_code}\n```"},
                {"role": "user", "content": (
                    f"Your code failed the tests:\n\n{feedback}{prior_note}\n\n"
                    "Apply the smallest edit that fixes the specific failure above. "
                    "Preserve working code paths that were not implicated in the failure."
                )},
            ]
            response = self.client.generate(
                config=self.config.proposer_model,
                system=CODE_SYSTEM_PROMPT,
                messages=revision_messages,
            )
            trace_steps.append({
                "step": "revise",
                "system": CODE_SYSTEM_PROMPT,
                "feedback": feedback,
                "messages_text": revision_messages,
                "full_output": response.content,
            })
            prior_errors.append((eval_result.error_message or "unknown")[:180])
            current_code = extract_code(response.content, "python")
            iteration = i + 2

        if best_passed and not passed:
            current_code, passed = best_code, True

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
            full_output=trace_steps[0]["full_output"],
            intermediate_outputs=[s["full_output"] for s in trace_steps[1:]],
            interaction_trace=trace_steps,
        )

    def run_video_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
        reviewer_model: ModelConfig | None = None,
    ) -> HardBenchmarkResult:
        """Run a single video editing task."""
        start_time = time.time()
        self.client.reset_counters()

        description = task["description"]
        requirements = "\n".join(f"- {r}" for r in task.get("requirements", []))
        source_video_name = task.get("source_video", "")
        frame_times = task.get("frame_check_times_s", [0, 1, 2, 3])

        # Resolve source video to absolute path
        video_data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "hard_benchmarks" / "video"
        if source_video_name:
            source_video = str(video_data_dir / source_video_name)
        else:
            source_video = ""

        video_system = (
            "You are an expert video editor writing Python code using moviepy or ffmpeg.\n"
            "Rules:\n"
            "- Output a complete Python script that performs the video editing\n"
            "- Use moviepy (from moviepy import *) or subprocess with ffmpeg\n"
            "- The script should read from the source video and write to '/tmp/output.mp4'\n"
            "- Handle all edge cases (duration, resolution, codec)\n"
            "- Wrap your code in a ```python code block"
        )

        prompt = f"{description}\n\nSource video: {source_video}\n\nRequirements:\n{requirements}"

        init_messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(
            config=self.config.proposer_model,
            system=video_system,
            messages=init_messages,
        )
        current_code = extract_code(response.content, "python")

        trace_steps = [{
            "step": "generate",
            "system": video_system,
            "messages": init_messages,
            "full_output": response.content,
        }]

        iteration = 1
        quality_score = 0.0
        issues = []
        meets_reqs = False
        best = None  # (quality, code, issues, meets)
        prior_summaries: list[str] = []

        from src.feedback.type3c_video import VideoFeedback
        video_fb = VideoFeedback(reviewer_model)

        for i in range(max_iterations):
            problem = {
                "source_video": source_video,
                "output_path": "/tmp/output.mp4",
                "frame_check_times_s": frame_times,
                "requirements": requirements,
            }
            fb = video_fb.get_feedback(current_code, problem)

            quality_score = fb.structured_data.get("quality_score", 0)
            issues = fb.structured_data.get("issues", [])
            meets_reqs = fb.structured_data.get("meets_requirements", False)

            if best is None or quality_score > best[0]:
                best = (quality_score, current_code, issues, meets_reqs)

            if not use_review:
                break

            if quality_score >= 0.9:
                break

            if i < max_iterations - 1:
                feedback_text = fb.content
                prior_note = ""
                if prior_summaries:
                    prior_note = (
                        "\n\nPrior attempt summaries (the same issue may be recurring — "
                        "try a different approach if you see the same feedback again):\n"
                        + "\n".join(f"- {s}" for s in prior_summaries)
                    )
                revision_msg = (
                    f"Your video editing code had problems:\n\n{feedback_text}{prior_note}\n\n"
                    "Apply the smallest possible edit that fixes the identified issues. "
                    "Preserve any parts of the script that were NOT flagged."
                )
                revision_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```python\n{current_code}\n```"},
                    {"role": "user", "content": revision_msg},
                ]
                response = self.client.generate(
                    config=self.config.proposer_model,
                    system=video_system,
                    messages=revision_messages,
                )
                trace_steps.append({
                    "step": "revise",
                    "system": video_system,
                    "feedback": feedback_text,
                    "messages_text": revision_messages,
                    "full_output": response.content,
                })
                prior_summaries.append(
                    f"iter{i+1} q={quality_score:.0%}: " + "; ".join(
                        (it.get("description", "") or "")[:100] for it in (issues or [])[:3]
                    )
                )
                new_code = extract_code(response.content, "python")
                if new_code.strip():
                    current_code = new_code
                iteration = i + 2

        if best is not None:
            quality_score, current_code, issues, meets_reqs = best

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        return HardBenchmarkResult(
            task_id=task["task_id"],
            task_type="video",
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_code,
            quality_score=quality_score,
            issues_found=issues,
            meets_requirements=meets_reqs,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
            full_output=trace_steps[0]["full_output"],
            intermediate_outputs=[s["full_output"] for s in trace_steps[1:]],
            interaction_trace=trace_steps,
        )

    def run_research_task(
        self,
        task: dict,
        use_review: bool = False,
        max_iterations: int = 1,
    ) -> HardBenchmarkResult:
        """Run a single deep research task with fact-checking feedback."""
        start_time = time.time()
        self.client.reset_counters()

        description = task["description"]
        requirements = task.get("requirements", [])
        req_text = "\n".join(f"- {r}" for r in requirements)

        research_system = (
            "You are a research analyst writing factual research reports.\n"
            "Rules:\n"
            "- Write a well-structured, factually accurate report\n"
            "- Include specific numbers, dates, names — be precise\n"
            "- Cite specific facts rather than generalities\n"
            "- If unsure about a fact, flag it rather than guessing\n"
            "- Output your report directly (no code blocks needed)"
        )

        prompt = f"{description}\n\nThe report must accurately cover:\n{req_text}"

        init_messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(
            config=self.config.proposer_model,
            system=research_system,
            messages=init_messages,
        )
        current_report = response.content

        trace_steps = [{
            "step": "generate",
            "system": research_system,
            "messages": init_messages,
            "full_output": response.content,
        }]

        iteration = 1
        accuracy = 0.0
        issues = []
        meets_reqs = False
        best = None  # (accuracy, report, issues, meets)
        prior_summaries: list[str] = []

        from src.feedback.type3d_factual import FactualVerificationFeedback
        fact_fb = FactualVerificationFeedback()

        for i in range(max_iterations):
            problem = {"requirements": requirements}
            fb = fact_fb.get_feedback(current_report, problem)

            accuracy = fb.structured_data.get("accuracy", 0)
            meets_reqs = fb.structured_data.get("passed", False)
            verifications = fb.structured_data.get("verifications", [])
            issues = [
                v for v in verifications if v.get("verdict") == "contradicted"
            ]

            if best is None or accuracy > best[0]:
                best = (accuracy, current_report, issues, meets_reqs)

            if not use_review:
                break

            if accuracy >= 0.9:
                break

            if i < max_iterations - 1:
                prior_note = ""
                if prior_summaries:
                    prior_note = (
                        "\n\nPrior attempts (avoid introducing NEW factual errors — "
                        "do not invent facts to fix unverifiable ones):\n"
                        + "\n".join(f"- {s}" for s in prior_summaries)
                    )
                revision_msg = (
                    f"Fact-checking found errors in your report:\n\n{fb.content}{prior_note}\n\n"
                    "Correct only the specific factual errors listed above. "
                    "Preserve everything that was NOT flagged. "
                    "If a fact cannot be verified, remove or hedge it — do not fabricate a replacement."
                )
                revision_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": current_report},
                    {"role": "user", "content": revision_msg},
                ]
                response = self.client.generate(
                    config=self.config.proposer_model,
                    system=research_system,
                    messages=revision_messages,
                )
                trace_steps.append({
                    "step": "revise",
                    "system": research_system,
                    "feedback": fb.content,
                    "messages_text": revision_messages,
                    "full_output": response.content,
                })
                prior_summaries.append(
                    f"iter{i+1} acc={accuracy:.0%}: " + "; ".join(
                        (v.get("claim", "") or "")[:80] for v in issues[:3]
                    )
                )
                current_report = response.content
                iteration = i + 2

        if best is not None:
            accuracy, current_report, issues, meets_reqs = best

        elapsed = time.time() - start_time
        usage = self.client.get_usage_summary()

        return HardBenchmarkResult(
            task_id=task["task_id"],
            task_type="research",
            baseline="single_shot" if not use_review else "proposer_reviewer",
            iterations=iteration,
            final_code=current_report,
            quality_score=accuracy,
            issues_found=issues,
            meets_requirements=meets_reqs,
            total_tokens=usage["total_tokens"],
            wall_time_seconds=elapsed,
            full_output=trace_steps[0]["full_output"],
            intermediate_outputs=[s["full_output"] for s in trace_steps[1:]],
            interaction_trace=trace_steps,
        )

    # -------------------------------------------------------------------
    # VLM Review helpers
    # -------------------------------------------------------------------

    def _vlm_review(
        self,
        screenshot_b64: str,
        requirements: str,
        model_config: ModelConfig | None = None,
        prior_reviews: list[dict] | None = None,
    ) -> tuple[float, list[dict], bool, list[str]]:
        """Use VLM to review a single screenshot.

        If ``prior_reviews`` is provided, the reviewer is told which issues
        were previously flagged so it can mark still-present ones and avoid
        verbatim repetition of fixed-in-this-iteration ones.
        """
        config = model_config or ModelConfig.claude_sonnet()
        prompt = VISUAL_REVIEW_PROMPT.format(requirements=requirements)
        if prior_reviews:
            prompt = self._augment_review_with_history(prompt, prior_reviews)

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
        prior_reviews: list[dict] | None = None,
    ) -> tuple[float, list[dict], bool, list[str]]:
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

        n_frames = len(frame_b64s)
        span_ms = frame_times[-1] - frame_times[0] if n_frames > 1 else 0
        review_prompt = (
            f"You are given {n_frames} frames captured from an animation, "
            f"spanning {span_ms}ms (t={frame_times[0]}ms to t={frame_times[-1]}ms). "
            f"Each frame is labeled with its timestamp. Evaluate the animation as "
            f"a SEQUENCE across all frames — not only the first frame.\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"Check for:\n"
            f"1. Visual glitches or rendering artifacts in any frame\n"
            f"2. Elements going out of bounds\n"
            f"3. Animation timing and motion: do elements change between frames "
            f"as the requirements describe? Is motion smooth and well-paced?\n"
            f"4. Missing or incorrect visual elements\n"
            f"5. Temporal coherence (smooth transitions between consecutive frames)\n\n"
            f"Score primarily on whether the animation satisfies the requirements. "
            f"If frames are visually similar, consider whether the requirements "
            f"actually call for motion at those timestamps before penalizing — "
            f"e.g., a brief static hold between transitions can be intentional. "
            f"Do not penalize minor per-frame imperfections when the overall "
            f"behavior matches the spec.\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"issues": [{{"description": "...", "severity": "critical|major|minor", '
            f'"frame": "t=Xms"}}], "quality_score": 0.0-1.0, '
            f'"meets_requirements": true/false, "suggestions": ["..."]}}'
        )
        if prior_reviews:
            review_prompt = self._augment_review_with_history(
                review_prompt, prior_reviews
            )
        content.append({"type": "text", "text": review_prompt})

        response = self.client.generate(
            config=config,
            system="You are a visual quality reviewer for animations. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_review_json(response.content)

    @staticmethod
    def _augment_review_with_history(
        prompt: str, prior_reviews: list[dict]
    ) -> str:
        """Prepend prior-iteration feedback so the reviewer can detect repeats."""
        lines = ["REVIEW HISTORY (earlier attempts at this artifact):"]
        for i, rev in enumerate(prior_reviews, start=1):
            q = rev.get("quality_score", 0.0)
            issues = rev.get("issues", []) or []
            short = "; ".join(
                (it.get("description", "") or "")[:120] for it in issues[:5]
            )
            lines.append(f"  Attempt {i}: quality={q:.0%} — {short}")
        lines.append(
            "\nFor each still-present issue, note that it RECURRED across attempts "
            "(the proposer's last fix did not resolve it). If the same issue appears "
            "again, mark it explicitly so the proposer tries a different strategy."
        )
        return prompt + "\n\n" + "\n".join(lines)

    @staticmethod
    def _parse_review_json(
        raw: str,
    ) -> tuple[float, list[dict], bool, list[str]]:
        """Parse VLM review JSON response.

        Returns (quality_score, issues, meets_requirements, suggestions).
        """
        raw = raw.strip()
        if raw.startswith("```"):
            first_nl = raw.find("\n")
            if first_nl == -1:
                raw = raw[3:]
            else:
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
                    return 0.5, [], False, []
            else:
                return 0.5, [], False, []

        return (
            float(data.get("quality_score", 0.5)),
            data.get("issues", []),
            data.get("meets_requirements", False),
            data.get("suggestions", []),
        )

    @staticmethod
    def _format_review_feedback(
        issues: list[dict],
        quality_score: float,
        meets_reqs: bool,
        suggestions: list[str] | None = None,
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
        if suggestions:
            parts.append("\nSuggested fixes:")
            for i, s in enumerate(suggestions, 1):
                parts.append(f"  {i}. {s}")
        return "\n".join(parts)


def run_hard_benchmarks(
    task_types: list[str] | None = None,
    num_tasks: int | None = None,
    max_iterations: int = 3,
    output_dir: Path | None = None,
) -> dict:
    """Run hard benchmark experiments."""
    task_types = task_types or ["slides", "animations", "code", "webpages", "video", "research"]
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
        # Map task type to file path
        file_map = {
            "slides": "slides/slide_tasks.json",
            "animations": "animations/animation_tasks.json",
            "code": "code/code_tasks.json",
            "webpages": "webpages/webpage_tasks.json",
            "video": "video/video_tasks.json",
            "research": "research/research_tasks.json",
        }
        task_file = Path(f"data/hard_benchmarks/{file_map.get(task_type, '')}")

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

            # Map task type to runner method
            run_fn = {
                "slides": runner.run_slide_task,
                "webpages": runner.run_slide_task,  # Same HTML rendering pipeline
                "animations": runner.run_animation_task,
                "code": runner.run_code_task,
                "video": runner.run_video_task,
                "research": runner.run_research_task,
            }.get(task_type)

            if run_fn is None:
                logger.warning("Unknown task type: %s", task_type)
                continue

            # Single-shot (no review)
            r1 = run_fn(task, use_review=False, max_iterations=1)

            # With review
            review_iters = 5 if task_type in ("code", "research") else max_iterations
            r2 = run_fn(task, use_review=True, max_iterations=review_iters)

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
