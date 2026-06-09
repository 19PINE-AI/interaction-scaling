#!/usr/bin/env python3
"""Build the human-preference study kit for the grounded-geometry result.

Purpose
-------
The paper claims grounded geometric feedback removes *real, human-visible*
layout defects. The deterministic DOM instrument both drives the revision and
scores it, so the reported defect reductions are partly an optimization of the
metric itself (see the circularity caveat in sec_grounded_eval.tex). The
definitive check is whether *humans* prefer the reviewed renders. This script
builds the stimuli + blind manifest so that check can actually be run on real
raters; it does NOT produce any human judgements itself.

What it does
------------
For each (task, seed) in the four visual modalities (academic figures, dense
slides, web pages, animations) it:
  1. loads the saved single-shot (``ss_*``) and reviewed (``rv_*``) artifacts
     from the geometry result JSONs (which store the final HTML + defect counts);
  2. *re-renders* each artifact FULL-PAGE so that off-canvas overflow -- the very
     defect a fixed-viewport screenshot crops away -- is visible to the rater
     (figures/slides at 1920x1080 full-page; web at desktop 1920 + mobile 375;
     animations as a horizontal strip of the scored frame times);
  3. randomly assigns the two renders to sides A/B (blind), and shuffles pair
     order, using a fixed RNG seed for reproducibility;
  4. writes ``study/manifest.js`` (window.MANIFEST -- NO ground truth, safe to
     load in the browser) and ``study/key.json`` (the hidden condition/defect
     key, consumed only by the offline analysis script).

Stimuli faithfulness: artifacts are re-rendered from the exact final HTML that
was scored, at the same 1920x1080 design target the instrument used. The only
deliberate difference from the scored capture is full_page=True, which *reveals*
rather than hides overflow -- a conservative choice that makes the human task
fair rather than rigged toward "no difference".
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import struct
from pathlib import Path

from playwright.sync_api import sync_playwright

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "hard_benchmarks"
OUT = ROOT / "study"

# Modality -> the geometry result run files that back Table 2 (tab:geometric).
# web is two 10-task sets (plain + "_b") that together form the 20-task suite.
MODALITIES = {
    "figures": {
        "kind": "static",
        "files": ["paperfig_geom_hard_run1", "paperfig_geom_hard_run2", "paperfig_geom_hard_run3"],
        "label": "Academic figure",
        "instruction": "Which figure has the cleaner layout (no overlapping or cut-off labels, aligned boxes)?",
    },
    "slides": {
        "kind": "static",
        "files": ["slides_hard2_geom_run1", "slides_hard2_geom_run2", "slides_hard2_geom_run3"],
        "label": "Slide",
        "instruction": "Which slide has the cleaner layout (no overlapping or overflowing text, aligned elements)?",
    },
    "web": {
        "kind": "web",
        "files": ["web_hard_geom_run1", "web_hard_geom_b_run1",
                  "web_hard_geom_run2", "web_hard_geom_b_run2",
                  "web_hard_geom_run3", "web_hard_geom_b_run3"],
        "label": "Web page",
        "instruction": "Which web page lays out better across desktop AND mobile (no horizontal overflow or overlap)?",
    },
    "animations": {
        "kind": "anim",
        "files": ["anim_clean_geom_run1", "anim_clean_geom_run2", "anim_clean_geom_run3"],
        "label": "Animation",
        "instruction": "Across the frames (left=earliest), which animation keeps every element on-screen and non-overlapping?",
    },
}

DESKTOP = (1920, 1080)
MOBILE = (375, 812)
DEFAULT_FRAME_TIMES = [0, 1000, 2000, 3000, 4000]


def png_size(b: bytes) -> tuple[int, int]:
    w, h = struct.unpack(">II", b[16:24])
    return w, h


def seed_tag(fname: str) -> str:
    # e.g. paperfig_geom_hard_run2 -> run2 ; web_hard_geom_b_run3 -> b_run3
    stem = fname.replace("_geom", "").replace("paperfig_hard", "fig").replace("slides_hard2", "slides")
    stem = stem.replace("anim_clean", "anim").replace("web_hard", "web")
    return stem


class Renderer:
    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)

    def _shot(self, html: str, w: int, h: int, full_page: bool) -> bytes:
        page = self._browser.new_page(viewport={"width": w, "height": h})
        try:
            try:
                page.set_content(html, wait_until="networkidle", timeout=15000)
            except Exception:
                page.set_content(html, wait_until="load", timeout=15000)
            page.wait_for_timeout(150)
            return page.screenshot(type="png", full_page=full_page)
        finally:
            page.close()

    def static(self, html: str) -> bytes:
        return self._shot(html, *DESKTOP, full_page=True)

    def web(self, html: str) -> tuple[bytes, bytes]:
        return self._shot(html, *DESKTOP, full_page=True), self._shot(html, *MOBILE, full_page=True)

    def frames(self, html: str, times: list[int]) -> bytes:
        page = self._browser.new_page(viewport={"width": DESKTOP[0], "height": DESKTOP[1]})
        try:
            try:
                page.set_content(html, wait_until="networkidle", timeout=15000)
            except Exception:
                page.set_content(html, wait_until="load", timeout=15000)
            page.wait_for_timeout(50)
            shots, elapsed = [], 0
            for t in times:
                wait = max(0, t - elapsed)
                if wait:
                    page.wait_for_timeout(wait)
                elapsed = t
                shots.append(page.screenshot(type="png", full_page=False))
            return _contact_sheet(shots)
        finally:
            page.close()

    def close(self):
        self._browser.close()
        self._pw.stop()


def _contact_sheet(shots: list[bytes]) -> bytes:
    """Compose frames into one horizontal strip (downscaled) with thin separators."""
    if Image is None:
        return shots[0]
    imgs = [Image.open(_BytesIO(s)).convert("RGB") for s in shots]
    th = 360
    scaled = [im.resize((max(1, int(im.width * th / im.height)), th)) for im in imgs]
    gap = 6
    W = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    sheet = Image.new("RGB", (W, th), (220, 220, 220))
    x = 0
    for im in scaled:
        sheet.paste(im, (x, 0))
        x += im.width + gap
    buf = _BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


from io import BytesIO as _BytesIO  # noqa: E402  (kept local to the renderer helpers)


def build(modalities: list[str], rng_seed: int = 20260609):
    rng = random.Random(rng_seed)
    OUT.mkdir(exist_ok=True)
    manifest: list[dict] = []
    key: dict[str, dict] = {}
    r = Renderer()
    counter = 0
    try:
        for mod in modalities:
            cfg = MODALITIES[mod]
            stim_dir = OUT / "stimuli" / mod
            stim_dir.mkdir(parents=True, exist_ok=True)
            n_rendered = 0
            for fname in cfg["files"]:
                fpath = RESULTS / f"{fname}.json"
                if not fpath.exists():
                    print(f"  [skip] missing {fpath.name}")
                    continue
                stag = seed_tag(fname)
                records = json.load(open(fpath))
                for rec in records:
                    counter += 1
                    pid = f"p{counter:04d}"
                    task_id = rec.get("task_id", f"t{counter}")
                    base = f"{task_id}__{stag}".replace("/", "_")
                    ss_html, rv_html = rec.get("ss_final_html"), rec.get("rv_final_html")
                    if not ss_html or not rv_html:
                        counter -= 1
                        continue
                    decisive = rec["ss_n_defects"] != rec["rv_n_defects"]

                    # blind side assignment
                    a_is = rng.choice(["ss", "rv"])
                    sides = {"A": a_is, "B": ("rv" if a_is == "ss" else "ss")}
                    html_for = {"ss": ss_html, "rv": rv_html}

                    entry = {"pair_id": pid, "modality": mod, "label": cfg["label"],
                             "instruction": cfg["instruction"], "decisive": decisive}
                    try:
                        if cfg["kind"] == "static":
                            entry["views"] = ["full"]
                            for side, cond in sides.items():
                                p = stim_dir / f"{base}_{side}.png"
                                p.write_bytes(r.static(html_for[cond]))
                                entry.setdefault(side, {})["full"] = f"stimuli/{mod}/{p.name}"
                        elif cfg["kind"] == "web":
                            entry["views"] = ["desktop", "mobile"]
                            for side, cond in sides.items():
                                d, m = r.web(html_for[cond])
                                pd = stim_dir / f"{base}_{side}_desktop.png"
                                pm = stim_dir / f"{base}_{side}_mobile.png"
                                pd.write_bytes(d)
                                pm.write_bytes(m)
                                entry.setdefault(side, {})
                                entry[side]["desktop"] = f"stimuli/{mod}/{pd.name}"
                                entry[side]["mobile"] = f"stimuli/{mod}/{pm.name}"
                        else:  # anim
                            entry["views"] = ["frames"]
                            times = rec.get("frame_times_ms") or DEFAULT_FRAME_TIMES
                            for side, cond in sides.items():
                                p = stim_dir / f"{base}_{side}_frames.png"
                                p.write_bytes(r.frames(html_for[cond], times))
                                entry.setdefault(side, {})["frames"] = f"stimuli/{mod}/{p.name}"
                    except Exception as e:  # pragma: no cover
                        print(f"  [render-fail] {mod}/{base}: {e}")
                        counter -= 1
                        continue

                    manifest.append(entry)
                    key[pid] = {
                        "modality": mod, "task_id": task_id, "seed": stag,
                        "A_condition": sides["A"], "B_condition": sides["B"],
                        "ss_defects": rec["ss_n_defects"], "rv_defects": rec["rv_n_defects"],
                        "decisive": decisive,
                    }
                    n_rendered += 1
            print(f"[{mod}] {n_rendered} pairs rendered")
    finally:
        r.close()

    rng.shuffle(manifest)
    (OUT / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    n_dec = sum(1 for e in manifest if e["decisive"])
    print(f"\nTotal: {len(manifest)} pairs ({n_dec} decisive, {len(manifest) - n_dec} tie controls)")
    print(f"Wrote {OUT/'manifest.js'} and {OUT/'key.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                    choices=list(MODALITIES))
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()
    build(args.modalities, args.seed)
