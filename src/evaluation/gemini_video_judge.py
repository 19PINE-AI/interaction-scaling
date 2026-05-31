"""Native full-video rubric judge using Gemini 3.1 Pro.

Instead of extracting keyframes and judging stills (which cannot see motion,
transitions, exact duration, or smoothness), this uploads the WHOLE rendered
video to Gemini via the Files API and asks it to judge each requirement
independently against the full clip it watches end-to-end. Gemini's native video
understanding is the appropriate instrument for a video-editing modality.

Returns the same structured dict as `checklist_judge` (score / n_met / n_total /
verdicts) so it is a drop-in replacement for `checklist_score_frames`.
"""

import logging
import os
import time

from src.evaluation.checklist_judge import _verdicts_from_response

logger = logging.getLogger("gemini_video_judge")

DEFAULT_MODEL = "gemini-3.1-pro-preview"

_SYSTEM = (
    "You are a meticulous video QA inspector. You are given ONE video and a list "
    "of objective binary requirements. Watch the entire video and judge each "
    "requirement INDEPENDENTLY as satisfied (true) or not (false), using what you "
    "actually observe across the whole clip (content, motion, transitions, "
    "duration, color). Respond with ONLY a JSON object: "
    '{"verdicts":[{"index":1,"satisfied":true|false,"evidence":"..."}, ...]} '
    "with exactly one verdict per requirement, in order."
)


def _client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _wait_active(client, file, timeout: float = 180.0):
    """Block until an uploaded file finishes processing (state ACTIVE)."""
    t0 = time.time()
    while getattr(file.state, "name", str(file.state)) == "PROCESSING":
        if time.time() - t0 > timeout:
            raise TimeoutError("Gemini file processing timed out")
        time.sleep(2)
        file = client.files.get(name=file.name)
    state = getattr(file.state, "name", str(file.state))
    if state != "ACTIVE":
        raise RuntimeError(f"Gemini file not ACTIVE (state={state})")
    return file


def checklist_score_video(
    video_path: str,
    requirements: list[str],
    model: str = DEFAULT_MODEL,
    client=None,
) -> dict:
    """Rubric-score a video file by full-clip understanding (Gemini native)."""
    if not requirements:
        return {"score": 0.0, "n_met": 0, "n_total": 0, "verdicts": []}

    client = client or _client()
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
    prompt = (
        "Judge the attached video against the following requirements. Judge each "
        "one independently across the WHOLE video.\n\nRequirements:\n"
        f"{numbered}"
    )

    uploaded = client.files.upload(file=video_path)
    try:
        uploaded = _wait_active(client, uploaded)
        from google.genai import types
        resp = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return _verdicts_from_response(resp.text or "", requirements)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:  # noqa: BLE001
            pass
