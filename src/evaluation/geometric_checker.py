"""Deterministic geometry checker for rendered HTML/SVG artifacts.

VLM-based visual judging is unreliable for the exact defects that matter most
in generated figures/slides: small text-on-text collisions, labels overflowing
their box, elements clipped at the viewport edge. Even at high resolution a VLM
misses blatant overlaps (observed: an Evoformer figure with two superimposed
section titles scored 1.00 from the VLM rubric).

Because these artifacts are HTML+SVG, we can measure the real on-screen
bounding box of every element in the browser and compute the geometric defects
EXACTLY -- no model in the loop. This module returns objective, reproducible
counts that serve as the geometric axes of the rubric; a VLM is only needed for
semantic axes (are the right components present / wired correctly).

Detected defects:
  * text_overlap   -- pairs of non-nested text elements whose boxes intersect
  * out_of_bounds  -- visible elements extending beyond the viewport
  * overflow       -- text wider/taller than its nearest block container
  * scrollbar      -- document larger than the viewport (content didn't fit)
"""

import json
import logging

logger = logging.getLogger(__name__)

# JS executed in the page: collect geometry and compute defects deterministically.
_PROBE_JS = r"""
() => {
  const W = window.innerWidth, H = window.innerHeight;
  const EPS = 4;            // ignore sub-pixel / hairline touches (px)
  const MINOVL = 6;         // min intersection side to count as a real overlap

  // --- collect leaf text elements (HTML leaves with text, and SVG <text>) ---
  const texts = [];
  const all = Array.from(document.querySelectorAll('*'));
  for (const el of all) {
    const tag = el.tagName.toLowerCase();
    if (tag === 'script' || tag === 'style' || tag === 'defs') continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) === 0) continue;
    let isText = false;
    if (tag === 'text' || tag === 'tspan') {
      isText = (el.textContent || '').trim().length > 0;
    } else {
      // HTML element whose DIRECT children include non-empty text
      let direct = '';
      for (const n of el.childNodes) if (n.nodeType === 3) direct += n.textContent;
      isText = direct.trim().length > 0;
    }
    if (!isText) continue;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    texts.push({tag, t: (el.textContent||'').trim().slice(0,40),
                x: r.left, y: r.top, r: r.right, b: r.bottom, el});
  }

  const inter = (a, b) => {
    const ix = Math.min(a.r, b.r) - Math.max(a.x, b.x);
    const iy = Math.min(a.b, b.b) - Math.max(a.y, b.y);
    return Math.min(ix, iy);
  };
  const nested = (a, b) => a.el.contains(b.el) || b.el.contains(a.el);

  // --- text-on-text overlaps ---
  const overlaps = [];
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      if (nested(a, b)) continue;
      if (inter(a, b) >= MINOVL) overlaps.push([a.t, b.t]);
    }
  }

  // --- CONTENT leaves clipped at the viewport edge ---
  // Count only painted LEAF content (text + SVG primitives), never structural
  // containers (html/body/div/svg/g) -- those double-count one oversized canvas
  // as many defects. The "content didn't fit" failure is captured once by the
  // scrollbar flag below; this counts genuinely clipped individual labels/shapes.
  const LEAF = new Set(['rect','circle','ellipse','polygon','line','path','text','tspan','img']);
  let oob = 0; const oobEx = [];
  const seen = new Set();
  // An element counts as CLIPPED only if it is partially in view and crosses an
  // edge. Elements entirely off-screen (e.g. position:absolute;left:-9999px
  // a11y/skip-link text, or off-canvas decorations) are hidden on purpose, not
  // clipped, so they must not register as defects.
  const partlyInView = (x, y, r, b) => r > 0 && b > 0 && x < W && y < H;
  const crossesEdge = (x, y, r, b) => r > W + EPS || b > H + EPS || x < -EPS || y < -EPS;
  for (const t of texts) {  // clipped text labels
    if (partlyInView(t.x, t.y, t.r, t.b) && crossesEdge(t.x, t.y, t.r, t.b)) {
      oob++; if (oobEx.length < 8) oobEx.push(t.t); seen.add(t.el);
    }
  }
  for (const el of all) {   // clipped SVG primitive shapes
    if (!LEAF.has(el.tagName.toLowerCase())) continue;
    if (seen.has(el)) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 && r.height < 2) continue;
    if (partlyInView(r.left, r.top, r.right, r.bottom) && crossesEdge(r.left, r.top, r.right, r.bottom)) {
      oob++; if (oobEx.length < 8) oobEx.push(el.tagName + ':' + (el.textContent||'').trim().slice(0,20));
    }
  }

  // --- text overflowing its nearest block container (HTML only) ---
  let overflow = 0; const ovEx = [];
  for (const t of texts) {
    if (t.tag === 'text' || t.tag === 'tspan') continue;
    let p = t.el.parentElement, box = null;
    while (p) {
      const pcs = getComputedStyle(p);
      if (['block','flex','grid','inline-block'].includes(pcs.display) &&
          (pcs.borderStyle !== 'none' || pcs.backgroundColor !== 'rgba(0, 0, 0, 0)' || pcs.overflow !== 'visible')) {
        box = p; break;
      }
      p = p.parentElement;
    }
    if (!box) continue;
    const br = box.getBoundingClientRect();
    if (t.r > br.right + EPS || t.b > br.bottom + EPS || t.x < br.left - EPS || t.y < br.top - EPS) {
      overflow++; if (ovEx.length < 8) ovEx.push(t.t);
    }
  }

  // Overflow beyond the viewport. Threshold at 16px so trivial default-margin
  // (e.g. an 8px body margin) is not counted; a real "doesn't fit / scrollbar"
  // failure (tens of px of content/canvas past the edge) is.
  const OVTHRESH = 16;
  const overW = document.documentElement.scrollWidth - W;
  const overH = document.documentElement.scrollHeight - H;
  const scrollbar = (overW > OVTHRESH) || (overH > OVTHRESH);

  return {
    n_text: texts.length,
    text_overlap: overlaps.length, overlap_examples: overlaps.slice(0, 12),
    out_of_bounds: oob, oob_examples: oobEx,
    overflow: overflow, overflow_examples: ovEx,
    scrollbar: scrollbar, overflow_px: [overW, overH],
    viewport: [W, H],
  };
}
"""


def geometric_defects(html: str, width: int = 1920, height: int = 1080,
                      renderer=None) -> dict:
    """Render *html* and return its deterministic geometric defects.

    Returns a dict with raw counts plus ``clean`` (bool: zero geometric defects)
    and ``n_defects`` (total). Reuses a BrowserRenderer's browser if given.
    """
    from src.rendering.browser import BrowserRenderer
    own = renderer is None
    renderer = renderer or BrowserRenderer(default_width=width, default_height=height)
    page = renderer._new_page(width, height)
    try:
        page.set_content(html, wait_until="networkidle")
        page.wait_for_timeout(80)
        data = page.evaluate(_PROBE_JS)
    except Exception as e:  # noqa: BLE001
        logger.warning("geometric probe failed: %s", e)
        data = {"error": str(e)}
    finally:
        page.close()
        if own:
            renderer.close()
    if "error" not in data:
        n = (data["text_overlap"] + data["out_of_bounds"] + data["overflow"]
             + (1 if data["scrollbar"] else 0))
        data["n_defects"] = n
        data["clean"] = (n == 0)
    return data


if __name__ == "__main__":
    import sys
    print(json.dumps(geometric_defects(open(sys.argv[1]).read()), indent=2))
