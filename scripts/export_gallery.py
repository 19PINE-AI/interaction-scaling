"""Export every visual test case (single-shot vs reviewed) to PNGs + a manifest
for the companion website and the paper's comparison grid.

Outputs:
  website/public/images/<category>/<task_id>_{ss,rv}.png
  website/public/data/manifest.json
"""
import base64, io, json, logging
from pathlib import Path
from PIL import Image
from src.evaluation.geometric_checker import geometric_defects
from src.rendering.browser import BrowserRenderer

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("export"); log.setLevel(logging.INFO)

ROOT = Path("website/public")
IMG = ROOT / "images"; DATA = ROOT / "data"
IMG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

MAXW = 1280  # downscale for web


def save_png(png_bytes, rel):
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)))
    p = IMG / rel; p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, "PNG", optimize=True)
    return f"images/{rel}"


def load_tasks(path, key="task_id"):
    d = json.load(open(path)); d = d if isinstance(d, list) else d.get("tasks", list(d.values())[0])
    return {t[key]: t for t in d}


def desc_of(task):
    return (task.get("description") or task.get("name") or "")[:600]


def rubric_lookup(path, run):
    """task_id -> (ss_rubric, rv_rubric) for a given on-policy run."""
    out = {}
    for r in json.load(open(path)):
        if r.get("run") == run and r.get("ss_rubric") is not None:
            out[r["task_id"]] = (round(r["ss_rubric"], 2), round(r["rv_rubric"], 2))
    return out


def main():
    rend = BrowserRenderer()
    items = []
    WEB_RUBRIC = rubric_lookup("results/hard_benchmarks/web_rubric_rescore.json",
                               "webpages_onpolicy_run1")
    ANIM_RUBRIC = rubric_lookup("results/hard_benchmarks/animations_rubric_rescore.json",
                                "animations_onpolicy_run1")

    # 1) Academic figures: screenshots already rendered + geometric defects
    figtasks = load_tasks("data/hard_benchmarks/diagrams/paper_figure_tasks.json")
    for r in json.load(open("results/hard_benchmarks/paperfig_geom_run1.json")):
        tid = r["task_id"]
        ss = save_png(base64.b64decode(r["ss_screenshot_b64"]), f"figures/{tid}_ss.png")
        rv = save_png(base64.b64decode(r["rv_screenshot_b64"]), f"figures/{tid}_rv.png")
        items.append({"category": "Academic figures", "task_id": tid,
                      "name": r["category"], "description": desc_of(figtasks.get(tid, {})),
                      "metric": "geometric defects", "ss_image": ss, "rv_image": rv,
                      "ss_score": r["ss_n_defects"], "rv_score": r["rv_n_defects"],
                      "lower_is_better": True, "iters": r.get("rv_iterations")})
        log.info("figure %s ss=%s rv=%s", tid, r["ss_n_defects"], r["rv_n_defects"])

    # 2) Slides: fixed-canvas DOM geometric defects (lower is better).
    slidetasks = load_tasks("data/hard_benchmarks/slides/slide_tasks.json")
    for rec in json.load(open("results/hard_benchmarks/slides_onpolicy_run1.json")):
        tid = rec["task_id"]
        try:
            ss_png = rend.render_html(rec["ss_final_code"], height=1080)
            rv_png = rend.render_html(rec["rv_final_code"], height=1080)
            ss_def = geometric_defects(rec["ss_final_code"], height=1080, renderer=rend).get("n_defects")
            rv_def = geometric_defects(rec["rv_final_code"], height=1080, renderer=rend).get("n_defects")
        except Exception as e:  # noqa: BLE001
            log.warning("Slides %s failed: %s", tid, e); continue
        items.append({"category": "Slides", "task_id": tid, "name": tid,
                      "description": desc_of(slidetasks.get(tid, {})),
                      "metric": "geometric defects",
                      "ss_image": save_png(ss_png, f"slides/{tid}_ss.png"),
                      "rv_image": save_png(rv_png, f"slides/{tid}_rv.png"),
                      "ss_score": ss_def, "rv_score": rv_def, "lower_is_better": True})
        log.info("Slides %s ss=%s rv=%s", tid, ss_def, rv_def)

    # 3) Web: scrollable pages -> binary rubric (fraction satisfied; higher is better).
    webtasks = load_tasks("data/hard_benchmarks/webpages/webpage_tasks.json")
    for rec in json.load(open("results/hard_benchmarks/webpages_onpolicy_run1.json")):
        tid = rec["task_id"]
        try:
            ss_png = rend.render_html(rec["ss_final_code"], height=1080)
            rv_png = rend.render_html(rec["rv_final_code"], height=1080)
        except Exception as e:  # noqa: BLE001
            log.warning("Web %s failed: %s", tid, e); continue
        ss_r, rv_r = WEB_RUBRIC.get(tid, (None, None))
        items.append({"category": "Web pages", "task_id": tid, "name": tid,
                      "description": desc_of(webtasks.get(tid, {})),
                      "metric": "binary rubric (frac. satisfied)",
                      "ss_image": save_png(ss_png, f"web/{tid}_ss.png"),
                      "rv_image": save_png(rv_png, f"web/{tid}_rv.png"),
                      "ss_score": ss_r, "rv_score": rv_r, "lower_is_better": False})
        log.info("Web %s rubric ss=%s rv=%s", tid, ss_r, rv_r)

    # 4) Animations: representative mid-sequence frame, ss vs rv
    atasks = load_tasks("data/hard_benchmarks/animations/animation_tasks.json")
    for rec in json.load(open("results/hard_benchmarks/animations_onpolicy_run1.json")):
        tid = rec["task_id"]
        ft = atasks.get(tid, {}).get("frame_times_ms", [0, 1000, 2000, 3000])
        mid = [ft[len(ft) // 2]]
        try:
            ssf = rend.render_animation_frames(rec["ss_final_code"], mid)
            rvf = rend.render_animation_frames(rec["rv_final_code"], mid)
            if not ssf or not rvf:
                continue
            ss = save_png(ssf[0], f"animations/{tid}_ss.png")
            rv = save_png(rvf[0], f"animations/{tid}_rv.png")
        except Exception as e:  # noqa: BLE001
            log.warning("anim %s failed: %s", tid, e); continue
        ss_r, rv_r = ANIM_RUBRIC.get(tid, (None, None))
        items.append({"category": "Animations", "task_id": tid, "name": tid,
                      "description": desc_of(atasks.get(tid, {})),
                      "metric": "binary rubric (frac. satisfied; one frame shown)",
                      "ss_image": ss, "rv_image": rv,
                      "ss_score": ss_r, "rv_score": rv_r, "lower_is_better": False})
        log.info("anim %s rubric ss=%s rv=%s", tid, ss_r, rv_r)

    manifest = {
        "title": "Interaction Scaling: Visual Generation Gallery",
        "categories": ["Academic figures", "Slides", "Web pages", "Animations"],
        "summary": {
            "Academic figures": {"metric": "DOM geometric defects", "ss": 1.20, "rv": 0.27, "delta": "-78%", "p": 0.008},
            "Slides": {"metric": "DOM geometric defects", "ss": 2.63, "rv": 0.23, "delta": "-91%"},
            "Web pages": {"metric": "binary rubric (frac. satisfied)", "ss": 0.469, "rv": 0.532, "delta": "+0.063", "p": 0.003},
            "Animations": {"metric": "binary rubric (frac. satisfied)", "ss": 0.642, "rv": 0.834, "delta": "+0.192", "p": 0.008},
        },
        "items": items,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d items to manifest", len(items))


if __name__ == "__main__":
    main()
