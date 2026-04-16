"""Type 3c feedback: video keyframe extraction and VLM temporal analysis.

Executes the model's video editing code, extracts keyframes at specified
timestamps, and has a VLM analyze the frames for correctness.
"""

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from src.config import ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)


class VideoFeedback(FeedbackProvider):
    """Type 3c — execute video editing code and review keyframes via VLM.

    The problem dict should contain:
    - ``code``: Python code that produces a video file
    - ``output_path``: expected output video path
    - ``frame_check_times_s``: list of timestamps to extract frames
    - ``requirements``: string describing what the video should look like
    - ``source_video``: path to source video (if needed)
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.claude_sonnet()
        self.client = get_client()

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Execute video code, extract frames, review via VLM."""
        frame_times = problem.get("frame_check_times_s", [0, 1, 2, 3])
        requirements = problem.get("requirements", "")

        # Step 1: Execute the video editing code
        exec_result = self._execute_code(code, problem)
        if not exec_result["success"]:
            return FeedbackResult(
                feedback_type=FeedbackType.EXECUTION,
                content=f"EXECUTION FAILED:\n{exec_result['error']}",
                structured_data={"passed": False, "error": exec_result["error"]},
                tokens_used=0,
            )

        output_path = exec_result["output_path"]
        if not Path(output_path).exists():
            return FeedbackResult(
                feedback_type=FeedbackType.EXECUTION,
                content=f"FAILED: Output video not found at {output_path}",
                structured_data={"passed": False, "error": "output_not_found"},
                tokens_used=0,
            )

        # Step 2: Extract keyframes
        frames = self._extract_frames(output_path, frame_times)
        if not frames:
            return FeedbackResult(
                feedback_type=FeedbackType.EXECUTION,
                content="FAILED: Could not extract any frames from output video",
                structured_data={"passed": False, "error": "frame_extraction_failed"},
                tokens_used=0,
            )

        # Step 3: VLM review of frames
        review = self._vlm_review_frames(frames, frame_times, requirements)

        return FeedbackResult(
            feedback_type=FeedbackType.VISUAL,
            content=review["content"],
            structured_data=review["structured"],
            tokens_used=review["tokens"],
        )

    def _execute_code(self, code: str, problem: dict) -> dict:
        """Execute the video editing Python code."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            source_video = problem.get("source_video", "")
            if source_video and Path(source_video).parent.is_dir():
                cwd = str(Path(source_video).parent)
            else:
                cwd = "/tmp"
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=cwd,
            )
            if result.returncode != 0:
                return {"success": False, "error": result.stderr[:3000]}
            return {
                "success": True,
                "output_path": problem.get("output_path", "/tmp/output.mp4"),
                "stdout": result.stdout[:2000],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out after 60s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _extract_frames(
        self, video_path: str, timestamps: list[float]
    ) -> list[bytes]:
        """Extract frames at specified timestamps using ffmpeg."""
        frames = []
        for t in timestamps:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                frame_path = f.name

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(t),
                        "-i", video_path,
                        "-vframes", "1",
                        "-f", "image2",
                        frame_path,
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if Path(frame_path).exists() and Path(frame_path).stat().st_size > 0:
                    frames.append(Path(frame_path).read_bytes())
                else:
                    logger.warning("Frame extraction failed at t=%.1fs", t)
            except Exception as e:
                logger.warning("Frame extraction error at t=%.1fs: %s", t, e)
            finally:
                Path(frame_path).unlink(missing_ok=True)

        return frames

    def _vlm_review_frames(
        self,
        frames: list[bytes],
        timestamps: list[float],
        requirements: str,
    ) -> dict:
        """Have VLM review extracted keyframes."""
        content = []
        for i, (frame, t) in enumerate(zip(frames, timestamps)):
            b64 = base64.b64encode(frame).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
            content.append({"type": "text", "text": f"Frame at t={t:.1f}s"})

        prompt = (
            f"These are keyframes extracted from an edited video.\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"Check for:\n"
            f"1. Does the video content match the editing requirements?\n"
            f"2. Are transitions smooth (no black frames, glitches)?\n"
            f"3. Are effects applied correctly (watermarks, fades, PiP)?\n"
            f"4. Is timing correct (durations, speeds)?\n"
            f"5. Is visual quality maintained (no artifacts, correct resolution)?\n\n"
            f'Respond with ONLY JSON: {{"issues": [{{"description": "...", '
            f'"severity": "critical|major|minor", "frame": "t=Xs"}}], '
            f'"quality_score": 0.0-1.0, "meets_requirements": true/false, '
            f'"suggestions": ["..."]}}'
        )
        content.append({"type": "text", "text": prompt})

        response = self.client.generate(
            config=self.model_config,
            system="You are a video quality reviewer. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": content}],
        )

        # Parse response
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw[raw.index("\n") + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {"issues": [], "quality_score": 0.5, "meets_requirements": False}

        formatted = (
            f"Quality: {data.get('quality_score', 0):.0%}\n"
            f"Meets requirements: {data.get('meets_requirements', False)}\n"
        )
        if data.get("issues"):
            formatted += "Issues:\n"
            for issue in data["issues"]:
                formatted += f"  - [{issue.get('severity', '?')}] {issue.get('description', '')}\n"
        if data.get("suggestions"):
            formatted += "Suggestions:\n"
            for s in data["suggestions"]:
                formatted += f"  - {s}\n"

        return {
            "content": formatted,
            "structured": data,
            "tokens": response.input_tokens + response.output_tokens,
        }
