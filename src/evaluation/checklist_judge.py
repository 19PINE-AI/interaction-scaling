"""Per-requirement binary checklist judge for visual artifacts.

Motivation
----------
The six-modality headline uses a single holistic VLM ``quality_score`` in
``[0, 1]`` as both the reviewer signal and the final judge. On the visual
modalities (slides, web, animations) this metric is noisy and saturating:

* ~23% of single-shot slide renders are scored a perfect 1.0 (zero headroom),
  so the harness can only stay flat or regress on them.
* The stochastic gestalt rating manufactures spurious negative deltas
  (a same-or-better revision re-scored lower), which kill the paired sign
  test even though the harness clearly fixes the genuinely-broken slides.

This module replaces the gestalt score with a *per-requirement* checklist.
Each task ships a list of concrete, near-deterministic requirements
("no scrollbar", "8 boxes visible", "arrows must not overlap text labels",
"boxes evenly spaced"). The judge evaluates each requirement *independently*
as satisfied / violated and returns the fraction satisfied. This is the
visual analogue of code's binary ``pytest`` pass-rate: low-noise, it cannot
award 1.0 to a slide with a visible overlap, and it localizes the defect.
"""

import base64
import json
import logging

from src.config import ModelConfig
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)

CHECKLIST_SYSTEM = (
    "You are a meticulous visual QA inspector. You are given a rendered "
    "screenshot and a numbered list of concrete, objectively-checkable "
    "requirements. Evaluate EACH requirement independently by looking at the "
    "image. Do not give partial credit and do not let a good overall "
    "impression excuse a specific violation: if a requirement says elements "
    "must not overlap and ANY two overlap, that requirement is violated even "
    "if the slide looks nice overall. Respond with ONLY valid JSON."
)

CHECKLIST_PROMPT = """\
Inspect the screenshot and judge each requirement below independently.

Requirements:
{numbered_requirements}

For each requirement, decide: is it SATISFIED (true) or VIOLATED (false) in \
the rendered image? Be strict about overlap, overflow, clipping, scrollbars, \
even spacing, and alignment — these are exactly the failures you must catch.

Respond with ONLY a JSON object:
{{
  "verdicts": [
    {{"index": 1, "satisfied": true|false, "evidence": "what you see in the image"}},
    ...
  ]
}}
Return exactly one verdict per requirement, in order."""


CHECKLIST_TEXT_SYSTEM = (
    "You are a meticulous fact and requirement checker. You are given a "
    "document and a numbered list of concrete, objectively-checkable "
    "requirements (often including exact facts/figures). Evaluate EACH "
    "requirement independently against the document. A requirement is "
    "satisfied only if the document actually states/does what it specifies; "
    "an omitted or contradicted fact is a violation. Respond with ONLY valid "
    "JSON."
)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        raw = raw[3:] if first_nl == -1 else raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return raw


def _quadrant_tiles_b64(screenshot_b64: str, overlap_frac: float = 0.08) -> list[str]:
    """Split a screenshot into 4 overlapping quadrant crops at native pixels.

    VLMs downscale a 1920x1080 image (long edge -> ~1568px), which hides small
    box/label overlaps in dense regions. Sending native-resolution quadrant
    crops alongside the full image roughly doubles the effective detail the
    judge sees, so it can catch fine-grained overlap/overflow/clipping.
    """
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(screenshot_b64))).convert("RGB")
    w, h = im.size
    ox, oy = int(w * overlap_frac), int(h * overlap_frac)
    boxes = [
        (0, 0, w // 2 + ox, h // 2 + oy),
        (w // 2 - ox, 0, w, h // 2 + oy),
        (0, h // 2 - oy, w // 2 + ox, h),
        (w // 2 - ox, h // 2 - oy, w, h),
    ]
    tiles = []
    for b in boxes:
        buf = io.BytesIO()
        im.crop(b).save(buf, format="PNG")
        tiles.append(base64.b64encode(buf.getvalue()).decode())
    return tiles


def checklist_score(
    screenshot_b64: str,
    requirements: list[str],
    model_config: ModelConfig | None = None,
    client=None,
    hires: bool = True,
) -> dict:
    """Score a screenshot by the fraction of requirements independently met.

    When ``hires`` is set (default), the full image is accompanied by four
    native-resolution overlapping quadrant crops so the judge can detect small
    overlaps/overflows that downscaling would otherwise hide.

    Returns a dict with:
        score          -- fraction of requirements satisfied in [0, 1]
        n_met / n_total
        verdicts       -- list of {index, requirement, satisfied, evidence}
    """
    if not requirements:
        return {"score": 0.0, "n_met": 0, "n_total": 0, "verdicts": []}

    config = model_config or ModelConfig.claude_sonnet()
    client = client or get_client()

    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
    prompt = CHECKLIST_PROMPT.format(numbered_requirements=numbered)

    content = [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": screenshot_b64},
    }, {"type": "text", "text": "Full rendered image above."}]
    if hires:
        try:
            tiles = _quadrant_tiles_b64(screenshot_b64)
            labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
            for tb64, lab in zip(tiles, labels):
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": tb64}})
                content.append({"type": "text", "text": (
                    f"Native-resolution {lab} quadrant (zoom in to check for "
                    "overlap/overflow/clipping you cannot see in the full image).")})
        except Exception:  # noqa: BLE001 -- fall back to full image only
            pass
    content.append({"type": "text", "text": prompt})

    response = client.generate(
        config=config,
        system=CHECKLIST_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return _verdicts_from_response(response.content, requirements)


def _verdicts_from_response(content: str, requirements: list[str]) -> dict:
    """Shared parse + index-align logic for the structured-JSON contract."""
    raw = _strip_fences(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        try:
            data = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        except json.JSONDecodeError:
            logger.warning("Checklist judge returned unparseable JSON")
            data = {}
    raw_verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    by_index = {v["index"]: v for v in raw_verdicts
                if isinstance(v, dict) and isinstance(v.get("index"), int)}
    verdicts, n_met = [], 0
    for i, req in enumerate(requirements, 1):
        v = by_index.get(i, {})
        satisfied = bool(v.get("satisfied", False))
        n_met += satisfied
        verdicts.append({"index": i, "requirement": req,
                         "satisfied": satisfied, "evidence": v.get("evidence", "")})
    n_total = len(requirements)
    return {"score": n_met / n_total if n_total else 0.0,
            "n_met": n_met, "n_total": n_total, "verdicts": verdicts}


def checklist_score_frames(
    frame_b64s: list[str],
    frame_times_ms: list[int],
    requirements: list[str],
    model_config: ModelConfig | None = None,
    client=None,
) -> dict:
    """Rubric score over an animation frame SEQUENCE.

    The requirements span per-frame axes (in-bounds, no glitch) and temporal
    axes (motion occurs, smoothness); each is judged independently across the
    whole sequence and externally averaged.
    """
    if not requirements:
        return {"score": 0.0, "n_met": 0, "n_total": 0, "verdicts": []}
    config = model_config or ModelConfig.claude_sonnet()
    client = client or get_client()

    content = []
    for b64, t in zip(frame_b64s, frame_times_ms):
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": b64}})
        content.append({"type": "text", "text": f"Frame at t={t}ms"})
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
    content.append({"type": "text", "text": (
        f"You are given {len(frame_b64s)} frames sampled from one animation, in "
        f"chronological order. Judge each requirement below independently across "
        f"the SEQUENCE (not just one frame).\n\nRequirements:\n{numbered}\n\n"
        "Respond with ONLY a JSON object: "
        '{"verdicts":[{"index":1,"satisfied":true|false,"evidence":"..."}, ...]} '
        "with exactly one verdict per requirement, in order.")})

    resp = client.generate(
        config=config,
        system=("You are a meticulous animation QA inspector judging a frame "
                "sequence against objective binary requirements. Respond with "
                "ONLY valid JSON."),
        messages=[{"role": "user", "content": content}],
    )
    return _verdicts_from_response(resp.content, requirements)


def checklist_score_text(
    document: str,
    requirements: list[str],
    model_config: ModelConfig | None = None,
    client=None,
) -> dict:
    """Rubric score for a text document (e.g. a research report) against
    objective binary requirements, including exact-fact requirements."""
    if not requirements:
        return {"score": 0.0, "n_met": 0, "n_total": 0, "verdicts": []}
    config = model_config or ModelConfig.claude_sonnet()
    client = client or get_client()
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
    prompt = (f"DOCUMENT:\n{document}\n\n----\nJudge each requirement "
              f"independently against the document above.\n\nRequirements:\n"
              f"{numbered}\n\nRespond with ONLY a JSON object: "
              '{"verdicts":[{"index":1,"satisfied":true|false,"evidence":"..."}, '
              "...]} with exactly one verdict per requirement, in order.")
    resp = client.generate(config=config, system=CHECKLIST_TEXT_SYSTEM,
                           messages=[{"role": "user", "content": prompt}])
    return _verdicts_from_response(resp.content, requirements)
