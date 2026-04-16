"""Type 3b feedback: visual analysis via VLM (visual grounding).

Renders HTML solutions to screenshots and sends them to a vision-language
model (Claude Sonnet) which inspects the rendered output for visual defects
such as text overflow, overlapping elements, broken layouts, poor contrast,
and animation glitches.

This complements Type 3a (execution) by catching issues that are visually
apparent but invisible to unit tests or static analysis.
"""

import base64
import json
import logging

from src.config import ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.rendering.browser import BrowserRenderer
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

VISUAL_REVIEW_SYSTEM = (
    "You are an expert UI/UX reviewer with deep knowledge of web rendering. "
    "You analyse screenshots of rendered HTML and identify visual defects. "
    "Always respond with valid JSON matching the schema described in the user "
    "prompt.  Do not include any text outside the JSON object."
)

VISUAL_REVIEW_PROMPT = """\
Carefully examine the provided screenshot(s) of a rendered HTML page and \
identify any visual issues.

Check for ALL of the following categories:
1. **Text overflow** -- text extending beyond its container boundaries.
2. **Text/element overlap** -- elements or text overlapping each other \
unintentionally.
3. **Alignment issues** -- misaligned elements that should share an edge or \
center.
4. **Readability problems** -- text that is too small to read, low contrast \
between text and background, or illegible fonts.
5. **Broken layout** -- elements that appear missing, incorrectly sized, or \
positioned outside the viewport.
6. **Animation issues** (if multiple frames are provided) -- temporal \
incoherence between frames, visual glitches, incorrect timing or \
sequencing.

Respond with a JSON object having exactly this structure:
{
    "issues": [
        {
            "category": "<one of: text_overflow, overlap, alignment, readability, broken_layout, animation>",
            "severity": "<one of: low, medium, high, critical>",
            "location": "<brief description of where on the page>",
            "description": "<what the problem is>"
        }
    ],
    "overall_quality": <float 0.0 to 1.0 where 1.0 is perfect>,
    "suggestions": ["<improvement suggestion>", "..."]
}

If the page looks correct with no issues, return an empty issues list and \
a quality score near 1.0.
"""

# Severity string to numeric weight for scoring validation
_SEVERITY_WEIGHTS = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.30,
    "critical": 0.50,
}


class VisualFeedback(FeedbackProvider):
    """Type 3b -- VLM-based visual analysis of rendered HTML.

    Screenshots can be supplied directly in the *problem* dict (key
    ``'screenshots'`` containing a list of raw PNG byte-strings), or the
    provider can render them on-the-fly from the ``'html'`` key.

    Args:
        model_config: VLM model to use for analysis.  Defaults to Claude
            Sonnet.
        renderer: Optional :class:`BrowserRenderer` for on-the-fly rendering.
            One is created lazily if needed and not supplied.
        render_width: Default viewport width when rendering HTML.
        render_height: Default viewport height when rendering HTML.
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        renderer: BrowserRenderer | None = None,
        render_width: int = 1920,
        render_height: int = 1080,
    ) -> None:
        self.model_config = model_config or ModelConfig.claude_sonnet()
        self._renderer = renderer
        self.render_width = render_width
        self.render_height = render_height

    @property
    def renderer(self) -> BrowserRenderer:
        if self._renderer is None:
            self._renderer = BrowserRenderer(
                default_width=self.render_width,
                default_height=self.render_height,
            )
        return self._renderer

    # ------------------------------------------------------------------
    # FeedbackProvider interface
    # ------------------------------------------------------------------

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Analyse rendered output of *code* / *problem* for visual issues.

        The method resolves screenshots in this order:
        1. ``problem['screenshots']`` -- pre-rendered PNG byte-strings.
        2. ``problem['html']`` -- HTML to render on-the-fly.
        3. Fall back to wrapping *code* in a minimal HTML document.

        Returns:
            A :class:`FeedbackResult` with ``feedback_type=VISUAL``.
        """
        screenshots = self._resolve_screenshots(code, problem)
        if not screenshots:
            logger.warning(
                "Type 3b visual: no screenshots could be produced"
            )
            return FeedbackResult(
                feedback_type=FeedbackType.VISUAL,
                content="SKIP: unable to produce screenshots for visual review.",
                structured_data={"skipped": True, "reason": "no_screenshots"},
                tokens_used=0,
            )

        # Encode to base64
        b64_images = [base64.b64encode(img).decode("ascii") for img in screenshots]

        logger.info(
            "Type 3b visual: sending %d screenshot(s) to %s",
            len(b64_images),
            self.model_config.model_id,
        )

        # Build multimodal message
        content_blocks: list[dict] = []
        for b64_data in b64_images:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_data,
                    },
                }
            )
        content_blocks.append({"type": "text", "text": VISUAL_REVIEW_PROMPT})

        client = get_client()
        response = client.anthropic_client.messages.create(
            model=self.model_config.model_id,
            max_tokens=2048,
            messages=[{"role": "user", "content": content_blocks}],
            system=VISUAL_REVIEW_SYSTEM,
        )

        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text += block.text

        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Track tokens on the shared client
        client.total_input_tokens += response.usage.input_tokens
        client.total_output_tokens += response.usage.output_tokens
        client.call_count += 1

        logger.debug("Type 3b visual: %d tokens used", tokens_used)

        parsed = self._parse_response(raw_text)

        return FeedbackResult(
            feedback_type=FeedbackType.VISUAL,
            content=self._format_human_readable(parsed),
            structured_data=parsed,
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # Screenshot resolution
    # ------------------------------------------------------------------

    def _resolve_screenshots(
        self, code: str, problem: dict
    ) -> list[bytes]:
        """Obtain screenshot PNG bytes from *problem* or by rendering."""
        # 1. Pre-supplied screenshots
        screenshots = problem.get("screenshots")
        if screenshots and isinstance(screenshots, list):
            return screenshots

        # 2. HTML in problem dict
        html = problem.get("html")
        if html:
            return [self.renderer.render_html(html, self.render_width, self.render_height)]

        # 3. Wrap code in minimal HTML
        if code.strip():
            wrapped = self._wrap_code_as_html(code)
            return [self.renderer.render_html(wrapped, self.render_width, self.render_height)]

        return []

    @staticmethod
    def _wrap_code_as_html(code: str) -> str:
        """Wrap raw code in a minimal HTML document for rendering."""
        return (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset='utf-8'></head>\n"
            "<body>\n"
            f"{code}\n"
            "</body></html>"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse the VLM JSON response, with fallback on parse failure."""
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Type 3b visual: failed to parse VLM response as JSON"
            )
            return {
                "issues": [],
                "overall_quality": 0.5,
                "suggestions": [],
                "raw_response": raw,
                "parse_error": True,
            }

        # Normalise fields
        data.setdefault("issues", [])
        data.setdefault("overall_quality", 0.5)
        data.setdefault("suggestions", [])

        # Clamp quality to [0, 1]
        try:
            data["overall_quality"] = max(0.0, min(1.0, float(data["overall_quality"])))
        except (TypeError, ValueError):
            data["overall_quality"] = 0.5

        return data

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_human_readable(parsed: dict) -> str:
        """Build a human-readable summary from the parsed VLM output."""
        parts: list[str] = []
        issues = parsed.get("issues", [])

        if not issues:
            parts.append("VISUAL REVIEW PASSED: no issues detected.")
        else:
            parts.append(f"VISUAL REVIEW: {len(issues)} issue(s) found.\n")
            for i, issue in enumerate(issues, 1):
                severity = issue.get("severity", "unknown").upper()
                category = issue.get("category", "unknown")
                location = issue.get("location", "unspecified")
                description = issue.get("description", "")
                parts.append(
                    f"  {i}. [{severity}] {category} @ {location}: {description}"
                )

        quality = parsed.get("overall_quality", "N/A")
        parts.append(f"\nOverall visual quality: {quality}")

        suggestions = parsed.get("suggestions", [])
        if suggestions:
            parts.append("\nSuggestions:")
            for s in suggestions:
                parts.append(f"  - {s}")

        return "\n".join(parts)
