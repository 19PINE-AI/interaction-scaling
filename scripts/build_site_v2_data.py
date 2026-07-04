"""Build data for the v2 companion website (three-section explainer site).

Outputs:
  website/public/data/v2/site.json            - aggregates for charts + explorer index
  website/public/data/v2/cases/<mod>/<id>.json - per-case detail (task, geom runs,
                                                 final artifacts, interaction traces)

Images are reused from website/public/images/ (exported earlier from the same
result files by scripts/build_site_data.py).

Aggregates are computed from raw result JSON where the raw files exist in this
repo; numbers that come from analysis pipelines not checkpointed here (VLM-vs-
geometric reviewer ablation, distillation, type-control execution arm) are
transcribed from the paper (main.tex v2) and marked "source": "paper".

Usage: python -m scripts.build_site_v2_data
"""
import json
import os
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
HB = R / "hard_benchmarks"
TASKS = REPO / "data" / "hard_benchmarks"
OUT = REPO / "website" / "public" / "data" / "v2"
CASES = OUT / "cases"
IMG = REPO / "website" / "public" / "images"

MAX_OUTPUT_CHARS = 12000


def load(p):
    with open(p) as f:
        return json.load(f)


def trim(s, n=MAX_OUTPUT_CHARS):
    if not isinstance(s, str):
        return s
    if len(s) <= n:
        return s
    return s[:n] + f"\n\n… [truncated, {len(s) - n:,} more characters]"


# ---------------------------------------------------------------- scaling
def scaling():
    files = [R / "scaling_curves" / f for f in
             ("code_4strategy.json", "code_4strategy_seed2.json", "code_4strategy_seed3.json")]
    agg = {}
    for f in files:
        if not f.exists():
            continue
        d = load(f)
        for strat, buds in d.items():
            for b, tasks in buds.items():
                agg.setdefault(strat, {}).setdefault(int(b), []).append(
                    round(100 * sum(1 for t in tasks.values() if t.get("passed")) / len(tasks), 1))
    META = {
        "H": ("Proposer–reviewer harness", "external"),
        "L": ("Single-agent loop", "external"),
        "S": ("Best-of-N (oracle verifier)", "internal"),
        "R": ("Reasoning-only", "internal"),
    }
    budgets = [1000, 5000, 20000]
    out = []
    for k in ("H", "L", "S", "R"):
        label, typ = META[k]
        seeds = [agg[k].get(b, []) for b in budgets]
        out.append({
            "key": k, "label": label, "type": typ, "budgets": budgets,
            "means": [round(mean(s), 1) if s else None for s in seeds],
            "seeds": seeds,
        })
    return {"series": out, "ceiling": 86.7,
            "note": "Hard code suite, 15 tasks, matched per-task token budget, 3 seeds "
                    "(reasoning-only: 1 seed). Oracle best-of-N flattens at 86.7%; both "
                    "interaction strategies climb past it."}


# ------------------------------------------------------- geometric modalities
GEOM_RUNS = {
    "figures": ["paperfig_geom_hard_run1.json", "paperfig_geom_hard_run2.json",
                "paperfig_geom_hard_run3.json"],
    "slides": ["slides_hard2_geom_run1.json", "slides_hard2_geom_run2.json",
               "slides_hard2_geom_run3.json"],
    "web": ["web_hard_geom_run1.json", "web_hard_geom_run2.json", "web_hard_geom_run3.json",
            "web_hard_geom_b_run1.json", "web_hard_geom_b_run2.json", "web_hard_geom_b_run3.json"],
    "animations": ["anim_clean_geom_run1.json", "anim_clean_geom_run2.json",
                   "anim_clean_geom_run3.json"],
}


def geom_records(mod):
    recs = []
    for f in GEOM_RUNS[mod]:
        p = HB / f
        if p.exists():
            recs.extend(load(p))
    return recs


def geom_reduction():
    """Per-modality defect stats across all (task, seed) pairs."""
    # CIs and p-values from the paper (bootstrap pipeline not re-run here).
    PAPER = {
        "figures": {"ci": [-96, -59], "p": "7×10⁻⁴"},
        "slides": {"ci": [-101, -53], "p": "0.0018"},
        "web": {"ci": [-64, -32], "p": "<2×10⁻³"},
        "animations": {"ci": [-70, -16], "p": "<2×10⁻³"},
    }
    LABEL = {"figures": "Academic figures", "slides": "Dense slides",
             "web": "Web pages", "animations": "Animations"}
    rows = []
    for mod in ("figures", "slides", "web", "animations"):
        recs = geom_records(mod)
        ss = [r["ss_n_defects"] for r in recs]
        rv = [r["rv_n_defects"] for r in recs]
        impr = sum(1 for a, b in zip(ss, rv) if b < a)
        regr = sum(1 for a, b in zip(ss, rv) if b > a)
        red = round(100 * (mean(rv) - mean(ss)) / mean(ss)) if mean(ss) else 0
        rows.append({"modality": mod, "label": LABEL[mod],
                     "ss_mean": round(mean(ss), 2), "rv_mean": round(mean(rv), 2),
                     "reduction_pct": red, "improved": impr, "regressed": regr,
                     "pairs": len(recs), **PAPER[mod]})
    return rows


# ---------------------------------------------------------------- cross-model
def crossmodel():
    fams = {"GPT-5": ["code_gpt-5.json", "code_gpt-5_seed2.json", "code_gpt-5_seed3.json"],
            "Qwen3-235B": ["code_qwen3-235b.json", "code_qwen3-235b_seed2.json",
                           "code_qwen3-235b_seed3.json"]}
    out = []
    for name, files in fams.items():
        ss_rates, rv_rates = [], []
        for f in files:
            p = R / "cross_model" / f
            if not p.exists():
                continue
            d = load(p)
            ss_rates.append(100 * sum(1 for r in d if r["single_shot_passed"]) / len(d))
            rv_rates.append(100 * sum(1 for r in d if r["reviewed_passed"]) / len(d))
        out.append({"model": name,
                    "ss": round(mean(ss_rates), 1), "ss_seeds": [round(x, 1) for x in ss_rates],
                    "rv": round(mean(rv_rates), 1), "rv_seeds": [round(x, 1) for x in rv_rates]})
    # Sonnet 4 headline (paper §Robustness; raw per-seed runs live in code_results_run*.json)
    out.insert(0, {"model": "Claude Sonnet 4", "ss": 66.7, "ss_seeds": [60.0, 66.7, 73.3],
                   "rv": 100.0, "rv_seeds": [100.0, 100.0, 100.0], "source": "paper"})
    return out


# ---------------------------------------------------------------- allocation
def allocation():
    p = R / "allocation_sweep" / "code_allocation.json"
    cells = load(p)["cells"]
    rows = [{"label": c["allocation_label"].split("_", 1)[1].replace("_", " "),
             "propose": c["b1_propose"], "execute": c["b2_execute"], "review": c["b3_review"],
             "pass_rate": round(100 * c["pass_rate"], 1)} for c in cells]
    rows.sort(key=lambda r: (r["propose"], r["pass_rate"]))
    return {"rows": rows, "budget": 10000,
            "spread_pp": round(max(r["pass_rate"] for r in rows) -
                               min(r["pass_rate"] for r in rows), 1)}


# ------------------------------------------------------------- explorer index
MOD_META = {
    "figures": {"title": "Academic figures", "img_dir": "figures",
                "task_file": "diagrams/paper_figure_tasks.json", "trace": None},
    "slides": {"title": "Dense slides", "img_dir": "slides",
               "task_file": "slides/slide_tasks_hard2.json",
               "trace": "slides_hard2_onpolicy_run1.json"},
    "web": {"title": "Web pages", "img_dir": "web",
            "task_file": "webpages/webpage_tasks_hard.json",
            "trace": "webpages_hard_onpolicy_run1.json"},
    "animations": {"title": "Animations", "img_dir": "animations",
                   "task_file": "animations/animation_tasks_clean.json", "trace": None},
}


def task_map(rel):
    p = TASKS / rel
    if not p.exists():
        return {}
    return {t["task_id"]: t for t in load(p)}


def nice_name(tid, t):
    for k in ("name", "paper", "category"):
        if t.get(k):
            return t[k]
    return tid.split("_", 1)[-1].replace("_", " ").title()


def clean_trace(trace):
    steps = []
    for st in trace or []:
        steps.append({
            "step": st.get("step"),
            "feedback": st.get("feedback"),
            "output": trim(st.get("full_output", "")),
        })
    return steps


def build_visual_cases():
    index = {}
    for mod, meta in MOD_META.items():
        tmap = task_map(meta["task_file"])
        # extra suites sharing the same id namespace
        if mod == "web":
            tmap.update(task_map("webpages/webpage_tasks_hard_b.json"))
        recs = geom_records(mod)
        by_task = {}
        for r in recs:
            by_task.setdefault(r["task_id"], []).append(r)
        traces = {}
        if meta["trace"] and (HB / meta["trace"]).exists():
            for r in load(HB / meta["trace"]):
                traces[r["task_id"]] = r
        items = []
        img_dir = IMG / meta["img_dir"]
        ids = sorted(set(by_task) | set(
            f.name.rsplit("_", 1)[0] for f in img_dir.glob("*_ss.jpg")))
        for tid in ids:
            ss_img = img_dir / f"{tid}_ss.jpg"
            rv_img = img_dir / f"{tid}_rv.jpg"
            if not (ss_img.exists() and rv_img.exists()):
                continue
            runs = by_task.get(tid, [])
            t = tmap.get(tid, {})
            tr = traces.get(tid)
            entry = {
                "id": tid, "name": nice_name(tid, t or (runs[0] if runs else {})),
                "ss_img": f"images/{meta['img_dir']}/{tid}_ss.jpg",
                "rv_img": f"images/{meta['img_dir']}/{tid}_rv.jpg",
                "has_trace": bool(tr),
            }
            if runs:
                entry["ss_defects"] = round(mean(r["ss_n_defects"] for r in runs), 1)
                entry["rv_defects"] = round(mean(r["rv_n_defects"] for r in runs), 1)
                entry["iters"] = round(mean(r["rv_iterations"] for r in runs), 1)
            items.append(entry)
            # ---- per-case detail file
            detail = {
                "id": tid, "modality": mod, "name": entry["name"],
                "task": {k: t.get(k) for k in
                         ("description", "requirements", "difficulty", "paper", "category")
                         if t.get(k) is not None},
                "runs": [{k: r.get(k) for k in
                          ("run", "ss_n_defects", "ss_text_overlap", "ss_clipped",
                           "ss_overflow", "ss_total_tokens", "rv_n_defects",
                           "rv_text_overlap", "rv_clipped", "rv_overflow",
                           "rv_iterations", "rv_total_tokens")} for r in runs],
                "ss_img": entry["ss_img"], "rv_img": entry["rv_img"],
            }
            if runs:
                r0 = runs[0]
                detail["ss_html"] = r0.get("ss_final_html")
                detail["rv_html"] = r0.get("rv_final_html")
            if tr:
                detail["trace"] = clean_trace(tr.get("rv_interaction_trace"))
                detail["trace_model"] = tr.get("model")
                detail["ss_quality"] = tr.get("ss_quality")
                detail["rv_quality"] = tr.get("rv_quality")
            d = CASES / mod
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{tid}.json", "w") as f:
                json.dump(detail, f)
        index[mod] = {"title": meta["title"], "items": items}
    return index


# ------------------------------------------------------------------ code cases
def build_code_cases():
    dev_tasks = task_map("code/code_tasks.json")
    hard_tasks = task_map("code/code_tasks_hard.json")
    strat = load(R / "scaling_curves" / "code_4strategy.json")
    cm = {}
    for mfile, mname in (("code_gpt-5.json", "GPT-5"), ("code_qwen3-235b.json", "Qwen3-235B")):
        p = R / "cross_model" / mfile
        if p.exists():
            for r in load(p):
                cm.setdefault(r["task_id"], {})[mname] = r
    traces = {r["task_id"]: r for r in load(HB / "code_hard_onpolicy_run1.json")} \
        if (HB / "code_hard_onpolicy_run1.json").exists() else {}

    items = []
    d = CASES / "code"
    d.mkdir(parents=True, exist_ok=True)

    for tid, t in dev_tasks.items():
        strategies = {}
        for k in ("H", "L", "S", "R"):
            per_b = {}
            for b in ("1000", "5000", "20000"):
                rec = strat.get(k, {}).get(b, {}).get(tid)
                if not rec:
                    continue
                cell = {"passed": rec["passed"], "tokens": rec["tokens_used"],
                        "turns": rec.get("num_turns"),
                        "error": rec.get("error_message")}
                if b == "20000":
                    cell["code"] = trim(rec.get("code_emitted"), 8000)
                    if rec.get("turns"):
                        cell["turn_log"] = [
                            {kk: tt.get(kk) for kk in
                             ("turn", "passed", "error_message", "propose_tokens",
                              "review_tokens", "output_tokens")}
                            for tt in rec["turns"]]
                per_b[b] = cell
            strategies[k] = per_b
        xm = {}
        for mname, r in cm.get(tid, {}).items():
            xm[mname] = {"ss_passed": r["single_shot_passed"], "rv_passed": r["reviewed_passed"],
                         "rv_iterations": r["rv_iterations"],
                         "ss_code": trim(r.get("ss_final_code"), 8000),
                         "rv_code": trim(r.get("rv_final_code"), 8000)}
        detail = {"id": tid, "modality": "code", "suite": "dev",
                  "name": tid.replace("_", " "),
                  "task": {"description": t.get("description"),
                           "test_code": t.get("test_code"),
                           "difficulty": t.get("difficulty")},
                  "strategies": strategies, "cross_model": xm}
        with open(d / f"{tid}.json", "w") as f:
            json.dump(detail, f)
        h20 = strategies.get("H", {}).get("20000", {})
        r20 = strategies.get("R", {}).get("20000", {})
        items.append({"id": tid, "name": detail["name"], "suite": "dev",
                      "difficulty": t.get("difficulty"),
                      "desc": (t.get("description") or "")[:140],
                      "h_passed": h20.get("passed"), "r_passed": r20.get("passed"),
                      "has_trace": False})

    for tid, t in hard_tasks.items():
        tr = traces.get(tid)
        detail = {"id": tid, "modality": "code", "suite": "deep-spec",
                  "name": tid.replace("codeh_", "").replace("_", " "),
                  "task": {"description": t.get("description"),
                           "test_code": t.get("test_code"),
                           "difficulty": t.get("difficulty")}}
        if tr:
            detail["trace"] = clean_trace(tr.get("rv_interaction_trace"))
            detail["ss_passed"] = bool(tr.get("ss_quality"))
            detail["rv_passed"] = bool(tr.get("rv_quality"))
            detail["ss_code"] = trim(tr.get("ss_final_code"), 8000)
            detail["rv_code"] = trim(tr.get("rv_final_code"), 8000)
        with open(d / f"{tid}.json", "w") as f:
            json.dump(detail, f)
        items.append({"id": tid, "name": detail["name"], "suite": "deep-spec",
                      "difficulty": t.get("difficulty"),
                      "desc": (t.get("description") or "")[:140],
                      "ss_passed": detail.get("ss_passed"),
                      "rv_passed": detail.get("rv_passed"),
                      "has_trace": bool(tr)})
    return {"title": "Code", "items": items}


# --------------------------------------------------------------- task catalog
def task_catalog():
    """Every test suite in the paper, for the 'all test cases' browser."""
    SUITES = [
        ("code", "Code — development suite", "code/code_tasks.json",
         "15 hard algorithmic tasks; instrument: pytest execution."),
        ("code", "Code — deep-spec suite", "code/code_tasks_hard.json",
         "11 from-scratch implementations validated against reference solutions."),
        ("code", "Code — held-out suite", "code/code_tasks_heldout_v2.json",
         "32 tasks built after the method was frozen; zero tuning."),
        ("figures", "Academic figures", "diagrams/paper_figure_tasks.json",
         "20 dense architecture figures; instrument: DOM geometry + alignment."),
        ("slides", "Dense slides", "slides/slide_tasks_hard2.json",
         "12 real-paper slides; instrument: DOM geometry + alignment."),
        ("web", "Web pages", "webpages/webpage_tasks_hard.json",
         "20 realistic pages scored at desktop + mobile widths."),
        ("animations", "Animations", "animations/animation_tasks_clean.json",
         "20 SVG/CSS animations probed frame-by-frame."),
        ("video", "Video editing", "video/video_tasks_hard.json",
         "15 programmatic editing tasks; Gemini-native full-clip rubric (scoped: strong single-shot)."),
        ("research", "Deep research", "research/research_tasks_hard.json",
         "15 fact-finding tasks with planted traps (scoped: saturates for frontier models)."),
    ]
    out = []
    for mod, title, rel, blurb in SUITES:
        p = TASKS / rel
        if not p.exists():
            continue
        tasks = load(p)
        out.append({"modality": mod, "suite": title, "blurb": blurb, "n": len(tasks),
                    "tasks": [{"id": t["task_id"],
                               "name": nice_name(t["task_id"], t),
                               "difficulty": t.get("difficulty"),
                               "description": t.get("description"),
                               "requirements": t.get("requirements")}
                              for t in tasks]})
    return out


# ------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    site = {
        "meta": {
            "title": "Grounding the Loop on Both Sides",
            "subtitle": "Interaction as a third test-time compute axis — and why its gains "
                        "are invisible without grounded evaluation",
            "authors": [{"name": "Bojie Li", "affil": "Pine AI"},
                        {"name": "Noah Shi", "affil": "University of Washington"}],
            "repo": "https://github.com/19PINE-AI/interaction-scaling",
        },
        "scaling": scaling(),
        "geom_reduction": geom_reduction(),
        "crossmodel": crossmodel(),
        "allocation": allocation(),
        # Transcribed from the paper (analysis pipelines not checkpointed in-repo):
        "blindness": {"n": 15, "vlm_perfect": 14, "geom_clean": 3, "source": "paper"},
        "typecontrol": {"source": "paper", "arms": [
            {"label": "Ungrounded critique", "grounded": False, "pass": 86.7, "tokens": 7101},
            {"label": "+ Linter (form only)", "grounded": True, "pass": 86.7, "tokens": 7101},
            {"label": "+ Execution (behavior)", "grounded": True, "pass": 93.3, "tokens": 2600},
        ]},
        "ablation": {"source": "paper", "rows": [
            {"suite": "Dense slides", "arm": "VLM reviewer", "ss": 1.89, "rv": 2.44, "sig": "worse"},
            {"suite": "Dense slides", "arm": "Geometric reviewer", "ss": 1.25, "rv": 0.33,
             "sig": "−73%, p=0.0018"},
            {"suite": "Academic figures", "arm": "VLM reviewer", "ss": 0.52, "rv": 0.62, "sig": "worse (n.s.)"},
            {"suite": "Academic figures", "arm": "Geometric reviewer", "ss": 0.57, "rv": 0.15,
             "sig": "−74%, p=7×10⁻⁴"},
        ]},
        "crossmodel_geometry": {"source": "paper", "rows": [
            {"model": "Claude Sonnet 4", "reduction_pct": -73},
            {"model": "Gemini 3.1 Pro", "reduction_pct": -93, "note": "19 of 20 decisive task-runs improve"},
        ]},
        "efficiency": {"source": "paper", "rows": [
            {"label": "Proposer–reviewer", "tokens": 1029, "perfect_seeds": "3/3"},
            {"label": "Sample-and-select", "tokens": 1416, "perfect_seeds": "1/3"},
            {"label": "Single-agent loop", "tokens": 1431, "perfect_seeds": "2/3"},
        ]},
        "distill": {"source": "paper",
                    "ood": {"tasks": 44, "mean1": 0.51, "pass2": 0.70},
                    "hard": {"tasks": 18, "pass1": 44, "pass3": 56},
                    "variance": {"k": [1, 2, 3], "sft": [31, 50, 56], "rft": [28, 39, 39]}},
        "heldout": {"tasks": 32, "note": "Harness recovers every first-shot failure, zero "
                                         "regressions, perfect pass rate, no tuning.",
                    "source": "paper"},
    }
    explorer = build_visual_cases()
    explorer["code"] = build_code_cases()
    site["explorer"] = {m: {"title": v["title"],
                            "items": v["items"]} for m, v in explorer.items()}
    site["catalog"] = task_catalog()

    with open(OUT / "site.json", "w") as f:
        json.dump(site, f)
    n_cases = sum(1 for _ in CASES.rglob("*.json"))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"site.json + {n_cases} case files, {size/1e6:.1f} MB total")
    # quick sanity echo
    for row in site["geom_reduction"]:
        print(f"  {row['label']}: {row['ss_mean']} → {row['rv_mean']} "
              f"({row['reduction_pct']}%), {row['improved']}/{row['regressed']} impr/regr, "
              f"n={row['pairs']}")
    for s in site["scaling"]["series"]:
        print(f"  {s['key']}: {s['means']}")


if __name__ == "__main__":
    main()
