"""Per-trajectory sandbox + 3 general tools (write_file, read_file, bash).

Each trajectory gets a fresh workspace tmpdir. All file paths are forced
inside that root so the model can't escape with `..`. Bash runs with a
timeout and captures stdout/stderr/exit_code.

read_file(path) returns:
  - text content for *.py/*.html/*.css/*.json/*.md/*.txt/*.csv/...
  - PIL.Image for *.png/*.jpg/*.jpeg/*.webp
This is the one place we need a special tool (vs going through bash) because
bash returns text, and the model needs proper image content for visual tasks.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

TEXT_EXTS = {".py", ".html", ".css", ".js", ".json", ".md", ".txt", ".csv",
             ".yaml", ".yml", ".toml", ".sh", ".log", ".xml", ".svg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class Workspace:
    """One workspace per trajectory. Provides write_file / read_file / bash."""

    def __init__(self, root: Path | None = None):
        if root is None:
            root = Path(tempfile.mkdtemp(prefix=f"agent_ws_{uuid.uuid4().hex[:8]}_"))
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Add a .bin/ shim so `python` resolves to whatever python3 is.
        # This avoids agent failures when models default to `python` on
        # systems where only `python3` is on PATH.
        self.bin_dir = self.root / ".bin"
        self.bin_dir.mkdir(exist_ok=True)
        py3 = shutil.which("python3") or "/usr/bin/python3"
        py_shim = self.bin_dir / "python"
        py_shim.write_text(f"#!/bin/sh\nexec {py3} \"$@\"\n")
        py_shim.chmod(0o755)
        self._env = os.environ.copy()
        self._env["PATH"] = f"{self.bin_dir}:{self._env.get('PATH', '')}"

    def cleanup(self):
        try:
            shutil.rmtree(self.root, ignore_errors=True)
        except Exception:
            pass

    # --- path safety ----------------------------------------------------

    def _resolve_safe(self, path: str) -> Path:
        """Resolve path inside workspace; raise on traversal."""
        if not path:
            raise ValueError("path cannot be empty")
        p = (self.root / path).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"path escapes workspace: {path!r}")
        return p

    # --- tools ----------------------------------------------------------

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            p = self._resolve_safe(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"ok": True, "path": str(p.relative_to(self.root)),
                    "bytes": len(content.encode("utf-8"))}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def read_file(self, path: str, max_text_chars: int = 20000) -> dict[str, Any]:
        try:
            p = self._resolve_safe(path)
            if not p.exists():
                return {"ok": False, "error": f"file not found: {path!r}"}
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                img = Image.open(p).convert("RGB")
                # Resize huge images to keep token count reasonable for the VLM
                if max(img.size) > 1280:
                    s = 1280 / max(img.size)
                    img = img.resize((max(1, int(img.size[0]*s)),
                                      max(1, int(img.size[1]*s))), Image.LANCZOS)
                return {"ok": True, "kind": "image", "path": str(p.relative_to(self.root)),
                        "size": img.size, "image": img}
            elif ext in TEXT_EXTS or ext == "":
                text = p.read_text(encoding="utf-8", errors="replace")
                truncated = False
                if len(text) > max_text_chars:
                    text = text[:max_text_chars] + f"\n... [truncated {len(text) - max_text_chars} more chars]"
                    truncated = True
                return {"ok": True, "kind": "text", "path": str(p.relative_to(self.root)),
                        "content": text, "truncated": truncated}
            else:
                # Unknown extension — return as text best-effort
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    if len(text) > max_text_chars:
                        text = text[:max_text_chars] + f"\n... [truncated]"
                    return {"ok": True, "kind": "text", "path": str(p.relative_to(self.root)),
                            "content": text}
                except Exception:
                    return {"ok": False, "error": f"unknown file type, cannot read as text or image: {ext}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def bash(self, command: str, timeout: int = 60) -> dict[str, Any]:
        try:
            timeout = min(max(int(timeout), 1), 300)
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            # Truncate output to avoid blowing up context
            if len(stdout) > 8000:
                stdout = stdout[:8000] + f"\n... [truncated {len(stdout)-8000} chars of stdout]"
            if len(stderr) > 4000:
                stderr = stderr[:4000] + f"\n... [truncated {len(stderr)-4000} chars of stderr]"
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s",
                    "exit_code": -1, "stdout": "", "stderr": "(killed)"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "exit_code": -1, "stdout": "", "stderr": ""}


def dispatch_tool_call(ws: Workspace, name: str, arguments: dict) -> dict:
    """Dispatch a single parsed tool_call. Returns dict suitable for tool
    response message content (caller decides how to embed images)."""
    if name == "write_file":
        return ws.write_file(arguments.get("path", ""), arguments.get("content", ""))
    if name == "read_file":
        return ws.read_file(arguments.get("path", ""))
    if name == "bash":
        return ws.bash(arguments.get("command", ""), arguments.get("timeout", 60))
    return {"ok": False, "error": f"unknown tool: {name}"}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace, creating parent dirs if needed. Overwrites if exists. Use for creating source files, HTML, configs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root, e.g. 'solution.py' or 'site/index.html'"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace. Text files (.py/.html/.css/.json/.md/.txt etc.) return as text. Image files (.png/.jpg/.jpeg/.webp) return as image content viewable by the model. Use this (NOT bash cat) when you need to see an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the workspace directory. Returns stdout, stderr, exit_code. Use for any shell operation: ls, rm, mkdir, mv, cat (text), find, chmod, running scripts, executing tests (pytest/python), rendering HTML to PNG (playwright/chromium --headless), etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command, e.g. 'ls -la', 'python -m pytest test.py', 'chromium --headless --screenshot=out.png --window-size=1920,1080 file:///workspace/index.html'"},
                    "timeout": {"type": "integer", "description": "Seconds before kill (max 300)", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
]
