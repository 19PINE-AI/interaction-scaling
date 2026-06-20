# Experiments & reproduction

Runnable entry points live in [`../scripts/`](../scripts/) (and a few in
[`../src/experiments/`](../src/experiments/)). All commands assume the environment from the
[README quickstart](../README.md#quickstart) (`uv sync`, `playwright install chromium`, and
the relevant `*_API_KEY` exported).

Run any script with:

```bash
uv run python scripts/<script>.py
```

## Core scaling & ablation results

| Goal | Script | Reproduces |
|---|---|---|
| Reasoning vs sampling vs interaction scaling on hard code | `run_scaling_curves_code.py` | The 20K-budget comparison (R 73.3% / S 86.7% / interaction ≥ 97.8%) |
| Seed-robust scaling curves | `analyze_scaling_curves_seeds.py` | Multi-seed CIs for the above |
| Isolate execution from extra reviewing | `run_feedback_type_controls.py` | In-modality Type 1 = Type 2 vs Type 3a controls |
| Budget allocation across think/do/review | `run_allocation_sweep.py` + `analyze_allocation_sweep.py` | The allocation findings |
| Cross-model replication (3 families) | `run_cross_model_code.py` | Sonnet / Qwen3-235B / GPT-5 deltas |
| Reasoning-only matched-budget baseline | `run_reasoning_only_code.py` | Reasoning ceiling at matched tokens |
| IAD baseline | `run_iad_code.py` | Iterative-agent-debate comparison |
| Held-out code suite | `run_heldout_phase1.py` + `analyze_heldout_phase1.py` | 90.6% → 100% on zero-overlap tasks |

## Visual & geometric modalities

The "grounded evaluation" results rely on the deterministic geometry instrument
(`src/evaluation/geometric_checker.py`) rather than a VLM judge.

| Goal | Script | Reproduces |
|---|---|---|
| Geometric harness over visual artifacts | `run_geometric_harness.py` | Figures −74% / slides −73% / web −47% / animations −40% |
| Diagram / SVG-figure benchmark | `run_diagram_benchmark.py` | Diagram modality results |
| Rescore visual outputs under the binary rubric | `rescore_*_rubric.py`, `rescore_slides_hard.py`, `rescore_research_hard.py` | De-saturated rubric scores |
| Webpage benchmark | `run_webpage_benchmark.py` | Web-page harness deltas |
| All hard benchmarks in one pass | `run_all_hard_benchmarks.py` | Full hardened-suite sweep |

## Deep research modality

| Goal | Script |
|---|---|
| Fact-verification (Type 3d) probe | `run_research_probe.py` |
| Rescore hardened research tasks | `rescore_research_hard.py` |

## Distillation (Contribution 5)

The internalization pipeline lives in `src/training/`. Typical order:

1. **Collect** teacher traces — `src/training/collect_review_traces_v3.py` (code) /
   `collect_vl_traces.py` (visual).
2. **Prepare** SFT data — `src/training/prepare_agentic_sft.py` /
   `build_sft_v4_data.py`.
3. **SFT** — `src/training/train_sft_review_masked.py`.
4. **GRPO** with grounded rewards — `scripts/run_grpo_v2.sh` →
   `src/training/train_grpo_review_v2.py`.
5. **Evaluate** the student on held-out tasks — `src/training/run_agent_student.py`,
   `validate_heldout_tasks.py`.

> **Heads-up:** GRPO with a weak reward signal can drift the policy away from correct SFT
> answers. The notes (`notes/contribution_5_training.md`, `notes/phase5_*`) document where
> this happens and why distillation transfers the *format* of interaction scaling but is
> bounded by the *quality of the environmental feedback*.

## Figures, tables, and the website

| Goal | Script |
|---|---|
| Paper tables | `src/analysis/generate_tables.py`, `generate_paper_tables.py` |
| Paper plots | `src/analysis/plot_results.py` |
| Comparison grid figure | `scripts/make_grid_figure.py` |
| Test paper figures | `scripts/test_paper_figures.py` |
| Build results-website data | `scripts/build_site_data.py` |
| Export artifact gallery | `scripts/export_gallery.py` |

## Human-preference study

The study kit ([`../study/`](../study/)) tests the grounded-geometry caveat with human
raters.

```bash
uv run python scripts/build_human_study.py      # generate stimuli + manifest (gitignored output)
uv run python scripts/analyze_human_study.py     # analyze collected responses
```

See [`../study/PROTOCOL.md`](../study/PROTOCOL.md) and [`../study/README.md`](../study/README.md).

## Where results land

Eval outputs are written under `results/` and rendered images under various
`data/training/*_images/` directories — both **gitignored**. The durable record of what
each run found is the markdown logs in [`../notes/`](../notes/), one file per phase.
