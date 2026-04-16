"""Browser-based HTML renderer using Playwright.

Renders HTML content to PNG screenshots via a headless Chromium browser.
Used for visual feedback (Type 3b) -- the rendered screenshots are sent to a
VLM for quality analysis.
"""

import logging
from pathlib import Path

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

logger = logging.getLogger(__name__)


class BrowserRenderer:
    """Render HTML strings to PNG screenshots using headless Chromium.

    The browser instance is lazily initialised on first use and reused for
    subsequent renders.  Call :pymethod:`close` (or rely on ``__del__``) to
    release resources.

    Args:
        default_width: Default viewport width in pixels.
        default_height: Default viewport height in pixels.
    """

    def __init__(
        self,
        default_width: int = 1920,
        default_height: int = 1080,
    ) -> None:
        self.default_width = default_width
        self.default_height = default_height
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    # -- lazy init ---------------------------------------------------------

    def _ensure_browser(self) -> Browser:
        """Start Playwright and launch Chromium if not already running."""
        if self._browser is None:
            logger.info("BrowserRenderer: launching headless Chromium")
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        return self._browser

    def _new_page(self, width: int, height: int) -> Page:
        """Create a new page with the given viewport size."""
        browser = self._ensure_browser()
        return browser.new_page(viewport={"width": width, "height": height})

    # -- public API --------------------------------------------------------

    def render_html(
        self,
        html: str,
        width: int = 1920,
        height: int = 1080,
    ) -> bytes:
        """Render an HTML string and return the screenshot as PNG bytes.

        Args:
            html: Complete HTML document (or fragment) to render.
            width: Viewport width in pixels.
            height: Viewport height in pixels.

        Returns:
            Raw PNG image bytes of the rendered page.
        """
        page = self._new_page(width, height)
        try:
            page.set_content(html, wait_until="networkidle")
            screenshot = page.screenshot(type="png", full_page=False)
            logger.debug(
                "BrowserRenderer: captured screenshot (%d bytes, %dx%d)",
                len(screenshot),
                width,
                height,
            )
            return screenshot
        finally:
            page.close()

    def render_html_to_file(
        self,
        html: str,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Render an HTML string and save the screenshot to *output_path*.

        Args:
            html: Complete HTML document (or fragment) to render.
            output_path: Filesystem path for the output PNG.
            width: Viewport width in pixels.
            height: Viewport height in pixels.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = self.render_html(html, width, height)
        output_path.write_bytes(png_bytes)
        logger.info("BrowserRenderer: saved screenshot to %s", output_path)

    def render_animation_frames(
        self,
        html: str,
        frame_times_ms: list[int],
        width: int = 1920,
        height: int = 1080,
    ) -> list[bytes]:
        """Render multiple frames of an animated HTML page.

        The page is loaded once, then screenshots are captured at each
        timestamp by waiting the appropriate delta between consecutive
        captures.

        Args:
            html: HTML document containing CSS/JS animations.
            frame_times_ms: Monotonically increasing list of timestamps (in
                milliseconds from page load) at which to capture frames.
            width: Viewport width in pixels.
            height: Viewport height in pixels.

        Returns:
            List of PNG byte-strings, one per requested frame time.
        """
        if not frame_times_ms:
            return []

        page = self._new_page(width, height)
        try:
            page.set_content(html, wait_until="networkidle")

            frames: list[bytes] = []
            elapsed_ms = 0

            for target_ms in frame_times_ms:
                wait = max(0, target_ms - elapsed_ms)
                if wait > 0:
                    page.wait_for_timeout(wait)
                elapsed_ms = target_ms
                frame = page.screenshot(type="png", full_page=False)
                frames.append(frame)

            logger.debug(
                "BrowserRenderer: captured %d animation frames at %s ms",
                len(frames),
                frame_times_ms,
            )
            return frames
        finally:
            page.close()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Shut down the browser and Playwright server."""
        if self._browser is not None:
            logger.info("BrowserRenderer: closing browser")
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
