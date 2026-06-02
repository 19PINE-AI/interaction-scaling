"""Build the companion-website data: site.json + before/after images.

Compiles every experiment we ran (geometric four-modality with bootstrap CIs and
baselines, code, cross-model code, cross-model grounded geometry, the
feedback-grounding ablation, scaling curves, allocation, distillation, video,
research), the generation/review/judge/instrument prompts, and per-task
single-shot-vs-reviewed render pairs, into website/public/.

Usage: python -m scripts.build_site_data
"""
import base64, io, json, os
from pathlib import Path
from PIL import Image

R = "results/hard_benchmarks/"
WEB = Path("website/public")
IMG = WEB / "images"; DATA = WEB / "data"
IMG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)


def save_img(b64, rel, w=900):
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if im.width > w:
            im = im.resize((w, round(im.height * w / im.width)))
        (IMG / rel).parent.mkdir(parents=True, exist_ok=True)
        im.save(IMG / rel, "JPEG", quality=82)
        return f"images/{rel}"
    except Exception:
        return None


def gallery(files, modality, name_key=None, names=None):
    items = []
    for f in files:
        p = R + f
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            tid = r["task_id"]
            if any(it["task_id"] == tid for it in items):
                continue  # one (first seed) per task
            ss = save_img(r.get("ss_screenshot_b64", ""), f"{modality}/{tid}_ss.jpg")
            rv = save_img(r.get("rv_screenshot_b64", ""), f"{modality}/{tid}_rv.jpg")
            if not ss or not rv:
                continue
            items.append({"task_id": tid, "name": (names or {}).get(tid, r.get("category") or tid),
                          "ss_image": ss, "rv_image": rv,
                          "ss_defects": r.get("ss_n_defects"), "rv_defects": r.get("rv_n_defects"),
                          "iters": r.get("rv_iterations")})
    return items


def prompts():
    from src.experiments.hard_benchmark_runner import (
        DESIGN_PRINCIPLES, SLIDE_SYSTEM_PROMPT, WEBPAGE_SYSTEM_PROMPT,
        ANIMATION_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, VISUAL_REVIEW_PROMPT, REVISION_PROMPT)
    from src.evaluation.checklist_judge import CHECKLIST_SYSTEM, CHECKLIST_PROMPT
    from scripts.run_diagram_benchmark import DIAGRAM_SYSTEM_PROMPT
    from src.evaluation.gemini_video_judge import _SYSTEM as VIDEO_JUDGE_SYSTEM
    geom = ("Deterministic DOM-geometry instrument (no model). Render the HTML headless; "
            "via the layout engine read every element's bounding box; compute EXACTLY: "
            "text-on-text overlap (>=6px), out-of-bounds/clipping at the viewport, container "
            "overflow, document overflow (>16px), and box-group misalignment (rows/columns of "
            "card/pillar boxes flagged for unequal size, misaligned far edges, or uneven gutters). "
            "Web: horizontal overflow at 1920 and 375 widths, summed. Animations: probe the DOM at "
            "each sampled frame, summed. The exact defect list is fed back to the proposer and is "
            "also the score.")
    return [
        {"id": "design", "title": "Design principles (in every visual generation prompt)", "group": "generation", "text": DESIGN_PRINCIPLES},
        {"id": "slide", "title": "Slide generation system prompt", "group": "generation", "text": SLIDE_SYSTEM_PROMPT},
        {"id": "web", "title": "Web-page generation system prompt", "group": "generation", "text": WEBPAGE_SYSTEM_PROMPT},
        {"id": "anim", "title": "Animation generation system prompt", "group": "generation", "text": ANIMATION_SYSTEM_PROMPT},
        {"id": "diagram", "title": "Academic-figure generation system prompt", "group": "generation", "text": DIAGRAM_SYSTEM_PROMPT},
        {"id": "code", "title": "Code generation system prompt", "group": "generation", "text": CODE_SYSTEM_PROMPT},
        {"id": "revise", "title": "Revision prompt (proposer, given feedback)", "group": "feedback", "text": REVISION_PROMPT},
        {"id": "visualreview", "title": "VLM reviewer prompt (ungrounded baseline)", "group": "feedback", "text": VISUAL_REVIEW_PROMPT},
        {"id": "geom", "title": "Deterministic DOM-geometry instrument (grounded feedback + score)", "group": "evaluation", "text": geom},
        {"id": "checklistsys", "title": "Binary-rubric judge — system", "group": "evaluation", "text": CHECKLIST_SYSTEM},
        {"id": "checklist", "title": "Binary-rubric judge — per-requirement prompt", "group": "evaluation", "text": CHECKLIST_PROMPT},
        {"id": "videojudge", "title": "Gemini 3.1 Pro native full-video judge — system", "group": "evaluation", "text": VIDEO_JUDGE_SYSTEM},
    ]


def main():
    site = {
        "title": "Grounding the Loop on Both Sides",
        "subtitle": "Interaction as a third test-time compute axis, and why its gains are invisible without grounded evaluation",
        "thesis": ("Test-time compute has a third axis -- interaction with a grounded environment. "
                   "It is governed by GROUNDING, which must hold on BOTH sides of the loop: grounded "
                   "FEEDBACK drives quality past the reasoning/sampling ceiling, and grounded EVALUATION "
                   "is required even to MEASURE the gain. The default VLM-on-a-screenshot judge is "
                   "structurally blind to layout defects, so the entire visual-modality effect is "
                   "invisible until the metric is grounded in deterministic DOM geometry."),
        # ---- headline geometric four-modality (NEW, grounded) + baseline (single-shot) ----
        "geometric": {
            "note": "One identical configuration: Claude Sonnet 4, T=0, design-principle prompt, 3 seeds, alignment-inclusive reward, propose->measure->feed-exact-defects-back->revise (<=3 iters). Lower defects = better.",
            "rows": [
                {"modality": "Academic figures", "n": 60, "ss": 0.57, "rv": 0.15, "delta_pct": -74, "ci": [52, 89], "impr": 17, "regr": 2, "p": "7e-4"},
                {"modality": "Dense slides", "n": 36, "ss": 1.25, "rv": 0.33, "delta_pct": -73, "ci": [45, 93], "impr": 13, "regr": 1, "p": "1.8e-3"},
                {"modality": "Web pages", "n": 60, "ss": 16.1, "rv": 8.5, "delta_pct": -47, "ci": [30, 62], "impr": 33, "regr": 2, "p": "4e-8"},
                {"modality": "Animations", "n": 60, "ss": 16.9, "rv": 10.2, "delta_pct": -40, "ci": [10, 64], "impr": 34, "regr": 7, "p": "3e-5"},
            ],
        },
        # ---- VLM judge vs deterministic instrument (the blindness result) ----
        "blindness": {"vlm_perfect": 14, "geom_clean": 3, "n": 15,
                      "note": "Same 15 single-shot academic figures: the VLM-on-a-screenshot judge rates 14/15 'perfect'; the deterministic DOM-geometry check finds only 3/15 actually clean."},
        # ---- baseline (old pipeline / holistic) vs updated, where directly comparable ----
        "baseline_vs_updated": {
            "note": "How the measurement and prompt changes moved the numbers. The holistic VLM judge mis-states visual quality; the design-principle pipeline raised single-shot quality so suites were hardened to keep headroom.",
            "rows": [
                {"item": "Academic figures (geometric Δ)", "baseline": "-78% (1 seed, no alignment, old prompt)", "updated": "-74% (3 seeds, +alignment, p=7e-4, CI[52,89])"},
                {"item": "Web harness lift (rubric)", "baseline": "+0.063 (p=0.003, old pipeline)", "updated": "+0.054 n.s. (new design-aware pipeline) -- but deterministic geometry shows -47% (p=4e-8)"},
                {"item": "Video editing", "baseline": "+0.47 'load-bearing' (holistic; was a missing-moviepy artifact)", "updated": "+0.04 n.s. (Gemini native full-video rubric); saturated"},
                {"item": "Slides under VLM rubric", "baseline": "noisy holistic, ~saturated", "updated": "rubric saturates (0.97) but deterministic geometry -73% (p=1.8e-3)"},
            ],
        },
        # ---- feedback-grounding ablation (two modalities) ----
        "ablation": {
            "note": "Same tasks/proposer; only the reviewer's grounding varies. An ungrounded VLM reviewer makes layout WORSE; the deterministic geometric reviewer fixes it.",
            "rows": [
                {"modality": "Slides", "vlm_ss": 1.89, "vlm_rv": 2.44, "geom_ss": 1.25, "geom_rv": 0.33},
                {"modality": "Academic figures", "vlm_ss": 0.52, "vlm_rv": 0.62, "geom_ss": 0.57, "geom_rv": 0.15},
            ],
        },
        # ---- code (execution-grounded) ----
        "code": {
            "rows": [
                {"suite": "Hard code (15)", "ss": "66.7%", "rv": "100.0%", "delta": "+33.3pp", "p": "0.008"},
                {"suite": "Deep-spec code (11)", "ss": "69.7%", "rv": "100.0%", "delta": "+30.3pp (CI[15,46])", "p": "0.002"},
            ],
        },
        "crossmodel_code": [
            {"model": "Claude Sonnet 4", "ss": "66.7%", "rv": "100.0%", "delta": "+33.3pp"},
            {"model": "Qwen3-235B", "ss": "66.7%", "rv": "93.3%", "delta": "+26.7pp"},
            {"model": "GPT-5", "ss": "86.7%", "rv": "100.0%", "delta": "+13.3pp"},
        ],
        "crossmodel_geometry": {"model": "Gemini 3.1 Pro", "modality": "Dense slides", "ss": 3.50, "rv": 0.25,
                                "delta_pct": -93, "impr": 19, "regr": 1, "p": "4e-5",
                                "note": "Grounded geometry replicates off Claude, scored by the same model-free instrument."},
        "scaling": {"note": "15 hard code tasks, 20K-token budget.",
                    "rows": [{"strategy": "Reasoning-only", "pass": 73.3, "grounded": False},
                             {"strategy": "Best-of-N", "pass": 86.7, "grounded": False},
                             {"strategy": "Single-agent loop", "pass": 97.8, "grounded": True},
                             {"strategy": "Proposer-reviewer harness", "pass": 100.0, "grounded": True}]},
        "allocation": {"spread_pp": 86.6, "note": "9-point proposer/reviewer split sweep at B=10K; pass-rate monotone in the proposer's per-call share; optimum is propose-heavy."},
        "distill": {"note": "8B Qwen3-VL student, SFT on judge-filtered teacher trajectories, ~10x cheaper.",
                    "rows": [{"metric": "pass@1 (18-task hard held-out)", "value": "44%"},
                             {"metric": "pass@3 (18-task hard held-out)", "value": "56%"},
                             {"metric": "mean@1 vs teacher (44-task OOD)", "value": "0.50x"},
                             {"metric": "mean@2 vs teacher (44-task OOD)", "value": "0.70x"}]},
        "saturated": [
            {"modality": "Video editing", "result": "single-shot 0.70 (Gemini native rubric), +0.04 n.s.; hardened suite drops single-shot 'fully correct' 49%->11%"},
            {"modality": "Deep research", "result": "single-shot 0.982 (exact-fact rubric), +0.018 n.s. -- frontier model knows well-documented facts; a genuine factual-modality limit"},
        ],
        "prompts": prompts(),
    }

    # ---- galleries (before/after renders from the grounded runs) ----
    figtasks = {t["task_id"]: t.get("name", "") for t in json.load(open("data/hard_benchmarks/diagrams/paper_figure_tasks.json"))}
    site["galleries"] = {
        "Academic figures": gallery(["paperfig_geom_hard_run1.json"], "figures", names=figtasks),
        "Dense slides": gallery(["slides_hard2_geom_run1.json"], "slides"),
        "Web pages": gallery(["web_hard_geom_run1.json", "web_hard_geom_b_run1.json"], "web"),
        "Animations": gallery(["anim_clean_geom_run1.json"], "animations"),
    }
    (DATA / "site.json").write_text(json.dumps(site, indent=1))
    n_imgs = sum(len(v) for v in site["galleries"].values())
    print(f"wrote site.json; galleries: " + ", ".join(f"{k}={len(v)}" for k, v in site["galleries"].items()))


if __name__ == "__main__":
    main()
