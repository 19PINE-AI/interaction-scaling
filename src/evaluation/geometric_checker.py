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
(cfg) => {
  const WEB = !!(cfg && cfg.web);   // scrollable page: only horizontal overflow is a defect
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
  // On a scrollable web page, content extending BELOW the fold (b > H) is normal,
  // not a defect; only HORIZONTAL clipping (past the right edge or before the
  // left edge) is a real layout failure. Fixed-canvas artifacts use all edges.
  const partlyInView = (x, y, r, b) => WEB ? (r > 0 && x < W) : (r > 0 && b > 0 && x < W && y < H);
  const crossesEdge = (x, y, r, b) => WEB ? (r > W + EPS || x < -EPS)
                                          : (r > W + EPS || b > H + EPS || x < -EPS || y < -EPS);
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
  // Vertical scroll is expected on a web page; only a HORIZONTAL scrollbar is a defect there.
  const scrollbar = WEB ? (overW > OVTHRESH) : ((overW > OVTHRESH) || (overH > OVTHRESH));

  // --- alignment of card/pillar BOX groups (HTML) ---
  // A frequent slide defect the overlap/overflow checks miss: a row of "stage"
  // boxes or side-by-side panels that should be equal-width with aligned edges
  // and uniform gutters, but are not. We collect bordered/filled HTML boxes,
  // keep the outermost ones, cluster them into rows (shared top) and columns
  // (shared left), and within each group of >=3 flag unequal size, misaligned
  // far edges, and uneven gaps. Tolerances are loose so only clear, visible
  // violations count (this is a reliability-first instrument).
  const SKIP_AL = new Set(['table','thead','tbody','tfoot','tr','td','th','svg','g',
                           'text','tspan','path','rect','circle','line','polygon',
                           'ellipse','html','body','ul','ol','li','br','hr']);
  const cand = [];
  for (const el of all) {
    const tag = el.tagName.toLowerCase();
    if (SKIP_AL.has(tag)) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) === 0) continue;
    let nb = 0;
    for (const s of ['Top','Right','Bottom','Left'])
      if (cs['border'+s+'Style'] !== 'none' && parseFloat(cs['border'+s+'Width']) > 0) nb++;
    const bg = cs.backgroundColor;
    const hasBg = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
    if (nb < 2 && !hasBg) continue;            // must look like a card/panel
    const r = el.getBoundingClientRect();
    if (r.width < 80 || r.height < 30) continue;
    if (r.width > W * 0.72 || r.height > H * 0.88) continue;  // skip full-canvas bg
    cand.push({el, x: r.left, y: r.top, r: r.right, b: r.bottom, w: r.width, h: r.height});
  }
  // keep only outermost cards (drop any card nested inside another card)
  const cards = cand.filter(c => !cand.some(o => o !== c && o.el.contains(c.el)));

  const spread = a => a.length ? Math.max(...a) - Math.min(...a) : 0;
  const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
  let misalign = 0; const malEx = [];
  const clusterBy = (key) => {        // greedy 1-D clustering on an edge coord
    const TOL = 16, groups = [];
    for (const c of [...cards].sort((p, q) => p[key] - q[key])) {
      const gg = groups.find(g => Math.abs(g[0][key] - c[key]) <= TOL);
      if (gg) gg.push(c); else groups.push([c]);
    }
    return groups.filter(g => g.length >= 3);
  };

  // rows: cards sharing a top edge -> should share bottom, equal width, even h-gaps
  for (const g of clusterBy('y')) {
    if (spread(g.map(c => c.w)) > Math.max(12, 0.06 * avg(g.map(c => c.w)))) {
      misalign++; if (malEx.length < 8) malEx.push('row: unequal box widths (x' + g.length + ')'); }
    if (spread(g.map(c => c.b)) > 14) {
      misalign++; if (malEx.length < 8) malEx.push('row: misaligned bottom edges'); }
    const s = [...g].sort((a, b) => a.x - b.x), gaps = [];
    for (let i = 1; i < s.length; i++) gaps.push(s[i].x - s[i - 1].r);
    if (gaps.length >= 2 && spread(gaps) > Math.max(12, 0.35 * Math.abs(avg(gaps)))) {
      misalign++; if (malEx.length < 8) malEx.push('row: uneven gutters'); }
  }
  // columns: cards sharing a left edge -> should share right, equal height, even v-gaps
  for (const g of clusterBy('x')) {
    if (spread(g.map(c => c.h)) > Math.max(12, 0.06 * avg(g.map(c => c.h)))) {
      misalign++; if (malEx.length < 8) malEx.push('col: unequal box heights (x' + g.length + ')'); }
    if (spread(g.map(c => c.r)) > 14) {
      misalign++; if (malEx.length < 8) malEx.push('col: misaligned right edges'); }
    const s = [...g].sort((a, b) => a.y - b.y), gaps = [];
    for (let i = 1; i < s.length; i++) gaps.push(s[i].y - s[i - 1].b);
    if (gaps.length >= 2 && spread(gaps) > Math.max(12, 0.35 * Math.abs(avg(gaps)))) {
      misalign++; if (malEx.length < 8) malEx.push('col: uneven vertical gaps'); }
  }

  return {
    n_text: texts.length,
    text_overlap: overlaps.length, overlap_examples: overlaps.slice(0, 12),
    out_of_bounds: oob, oob_examples: oobEx,
    overflow: overflow, overflow_examples: ovEx,
    scrollbar: scrollbar, overflow_px: [overW, overH],
    misalignment: misalign, misalignment_examples: malEx, n_cards: cards.length,
    viewport: [W, H],
  };
}
"""


def geometric_defects(html: str, width: int = 1920, height: int = 1080,
                      renderer=None, include_alignment: bool = False,
                      web: bool = False) -> dict:
    """Render *html* and return its deterministic geometric defects.

    Returns a dict with raw counts plus ``clean`` (bool: zero geometric defects)
    and ``n_defects`` (total). Reuses a BrowserRenderer's browser if given.

    ``misalignment`` (unequal-width / misaligned-edge / uneven-gutter box groups)
    is always reported but only folded into ``n_defects`` when
    ``include_alignment`` is set, so existing callers (figures, original slides)
    keep their published overlap/overflow-only counts unchanged.

    ``web=True`` scores a scrollable page: vertical document overflow and
    below-the-fold clipping are NOT defects (the page scrolls); only horizontal
    overflow, text-on-text overlap, container overflow, and misalignment count.
    """
    from src.rendering.browser import BrowserRenderer
    own = renderer is None
    renderer = renderer or BrowserRenderer(default_width=width, default_height=height)
    page = renderer._new_page(width, height)
    try:
        page.set_content(html, wait_until="networkidle")
        page.wait_for_timeout(80)
        data = page.evaluate(_PROBE_JS, {"web": web})
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
        if include_alignment:
            n += data.get("misalignment", 0)
        data["n_defects"] = n
        data["clean"] = (n == 0)
    return data


def animation_geometric_defects(html: str, frame_times_ms, renderer=None,
                                width: int = 1920, height: int = 1080,
                                include_alignment: bool = True) -> dict:
    """Deterministic geometry of an SVG/DOM animation, summed across sampled frames.

    The page is loaded once and advanced to each frame time; at every frame we
    probe the live DOM for the fixed-canvas defects (text overlap, clipping/
    out-of-bounds, container overflow, scrollbar, and box-group misalignment).
    The dominant animation failures---an element leaving the viewport mid-motion,
    two elements colliding during a transition, an animated grid going ragged---
    register as per-frame defects. The reward is the SUM of per-frame defect
    counts (a defect present for several frames counts for each), so the harness
    is pushed to keep the layout valid throughout the animation, not just at t=0.

    NOTE: measures the rendered DOM/SVG, so it applies to SVG/CSS/DOM animations,
    not pixels drawn inside a <canvas> (which the DOM cannot see).
    """
    from src.rendering.browser import BrowserRenderer
    own = renderer is None
    renderer = renderer or BrowserRenderer(default_width=width, default_height=height)
    page = renderer._new_page(width, height)
    per_frame, total = {}, 0
    try:
        page.set_content(html, wait_until="networkidle")
        page.wait_for_timeout(50)
        elapsed = 0
        for t in frame_times_ms:
            wait = max(0, t - elapsed)
            if wait:
                page.wait_for_timeout(wait)
            elapsed = t
            try:
                data = page.evaluate(_PROBE_JS, {"web": False})
            except Exception as e:  # noqa: BLE001
                per_frame[t] = {"error": str(e)}
                continue
            n = (data["text_overlap"] + data["out_of_bounds"] + data["overflow"]
                 + (1 if data["scrollbar"] else 0))
            if include_alignment:
                n += data.get("misalignment", 0)
            data["n_defects"] = n
            per_frame[t] = data
            total += n
    except Exception as e:  # noqa: BLE001
        logger.warning("animation probe failed: %s", e)
        return {"error": str(e), "n_defects": None}
    finally:
        page.close()
        if own:
            renderer.close()
    frames_with_defects = sum(1 for d in per_frame.values()
                              if isinstance(d, dict) and d.get("n_defects"))
    return {"n_defects": total, "clean": total == 0,
            "frames_with_defects": frames_with_defects,
            "n_frames": len(frame_times_ms), "per_frame": per_frame}


def web_geometric_defects(html: str, widths=((1920, 1080), (375, 812)),
                          renderer=None, include_alignment: bool = True) -> dict:
    """Deterministic web layout defects, summed across desktop and mobile widths.

    A responsive page must hold up at multiple widths; the classic single-shot
    failures are horizontal overflow / broken grids on mobile and card
    misalignment on desktop. We score the page at each width with ``web=True``
    and sum the per-width defect counts. Returns per-width detail plus the
    combined ``n_defects`` / ``clean``.
    """
    from src.rendering.browser import BrowserRenderer
    own = renderer is None
    renderer = renderer or BrowserRenderer()
    per_width, total = {}, 0
    try:
        for (w, h) in widths:
            g = geometric_defects(html, width=w, height=h, renderer=renderer,
                                   include_alignment=include_alignment, web=True)
            per_width[f"{w}x{h}"] = g
            total += g.get("n_defects", 0) if "error" not in g else 0
    finally:
        if own:
            renderer.close()
    return {"n_defects": total, "clean": total == 0, "per_width": per_width}


if __name__ == "__main__":
    import sys
    print(json.dumps(geometric_defects(open(sys.argv[1]).read()), indent=2))
