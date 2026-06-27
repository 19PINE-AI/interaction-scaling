# Interaction Scaling

**A third test-time compute axis, grounded in environment feedback.**

This repository contains the full research codebase, experiment harnesses, and paper
source for *Interaction Scaling* — a study of how agents improve their outputs by
iteratively interacting with an external environment rather than only reasoning longer
or sampling more.

> **Paper:** *Grounding the Loop on Both Sides: Interaction as a Third Test-Time Compute
> Axis, and Why Its Gains Are Invisible Without Grounded Evaluation*
> Bojie Li, Pine AI (`boj@19pine.ai`). Source in [`paper/`](paper/).

---

## TL;DR

Test-time compute is dominated by two axes that both operate inside a model's own token
space: **reasoning** (longer chains of thought) and **sampling** (best-of-*N*). Both are
bounded — by the data-processing inequality (DPI) — by the information already in the
weights and the prompt.

We study a third axis: **interaction** with an environment that returns *grounded*
feedback (execution results, rendered layout geometry, fact verification). Because that
feedback imports information from outside the model, it can exceed the reasoning/sampling
ceiling. The central claim is that a single variable — **grounding** — governs this axis,
and it must hold on *both* sides of the loop:

- **Grounded feedback** drives improvement past the reasoning/sampling ceiling.
- **Grounded evaluation** is required even to *measure* that improvement. A default VLM
  judge is structurally blind to the geometric defects the loop fixes (it rates 14 of 15
  single-shot academic figures "perfect" when only 3 are actually clean); a deterministic
  DOM-geometry instrument reveals the real gains.

## Headline results

| Setup | Result |
|---|---|
| Code harness (Sonnet 4, 3-run mean) | 66.7 ± 6.7% → **100.0 ± 0.0%** (+33.3 pp) |
| Academic figures (DOM-geometry defects) | **−74%** (paired sign-test *p* < 0.002) |
| Dense slides | **−73%** |
| Responsive web pages | **−47%** |
| SVG/CSS animations | **−40%** |
| Reasoning-only at matched 20K-token budget (code) | saturates at 73.3% / best-of-*N* 86.7% |
| Every interaction strategy at 20K budget | **≥ 97.8%**; proposer–reviewer wins on token cost, zero seed variance |
| Cross-model replication (3 seeds each) | Sonnet +33.3 pp / Qwen3-235B +22.2 pp / GPT-5 +20.0 pp (reviewed ceiling has zero seed variance) |
| Held-out code suite (32 tasks, zero overlap) | 90.6% → 100.0% |
| Distillation into 8B student | partial transfer of harness behavior |

A grounded VLM **reviewer** can even make visual layouts *worse* — only the deterministic
geometric instrument both fixes and measures the defect. See the paper for the full
account.

---

## The framework in one table

The grounded-feedback taxonomy that organizes the whole study:

| Type | Feedback source | Grounded? | Example |
|---|---|---|---|
| **0** | Self-review (same model re-reads its output) | No | Re-reading your own essay |
| **1** | LLM cross-review (different model/prompt) | No | Re-reading it wearing a different hat |
| **2** | Static tools (lint, type-check, structural validators) | Yes (pre-execution) | A copy editor checking grammar |
| **3a** | Execution feedback (tests, errors, runtime) | Yes | Running the experiment |
| **3b** | Visual rendering feedback (rendered layout, geometry) | Yes | Looking at the rendered page |
| **3c** | Temporal feedback (video / animation keyframes) | Yes | Watching the playback |
| **3d** | Factual verification (search, knowledge base) | Yes | Checking a claim against evidence |

Types 0–1 form a Markov chain `T → A → F` and are DPI-bounded. Types 2–3 introduce an
external variable `E`, breaking the chain and importing new, reliable information.

---

## Repository layout

```
interaction-scaling/
├── src/                      # Core library
│   ├── agents/               # Proposer, reviewer, single-agent, meta-controller
│   ├── feedback/             # Type 0–3d feedback providers (the taxonomy above)
│   ├── budget/               # Token-budget allocator + tracker (think/do/review)
│   ├── benchmarks/           # HumanEval / MBPP loaders
│   ├── evaluation/           # Code eval, geometric checker, VLM/Gemini judges, rubrics
│   ├── rendering/            # Headless-browser rendering (Playwright)
│   ├── training/             # Distillation: trace collection, SFT, GRPO/RFT, students
│   ├── experiments/          # Ablations, scaling curves, hard-benchmark runners
│   ├── analysis/             # Tables + plots
│   └── utils/                # LLM client, code utilities
├── scripts/                  # Runnable entry points (sweeps, rescoring, figures, study)
├── data/                     # Benchmark tasks + (small) training splits
│   ├── hard_benchmarks/      # Hardened, de-saturated task suites
│   └── training/             # SFT / GRPO data splits (large dumps are gitignored)
├── notes/                    # Per-phase findings logs (the running research record)
├── paper/                    # LaTeX source, figures, refs, built PDF
├── study/                    # Human-preference study kit (protocol + static site)
├── website/                  # Results website (Vite)
├── results/                  # Eval outputs (small summaries in git; large dumps via Git LFS)
└── docs/                     # Architecture & usage documentation
```

> **Note on artifacts.** Model checkpoints and large training dumps remain
> **not** version-controlled (see [`.gitignore`](.gitignore)). The eval **results** behind
> every table and figure *are* tracked: small summary outputs live in normal git, while the
> large per-run dumps (`results/hard_benchmarks/`, `results/phase5/`, `results/phase6/`, and
> rendered result images) are stored via **Git LFS** — see [Getting the data](#getting-the-data-git-lfs).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deeper tour of the modules and the
proposer–reviewer loop, and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for how to
reproduce the main results.

---

## Getting the data (Git LFS)

The eval results behind every paper table and figure are committed to the repository. The
large per-run dumps (`results/hard_benchmarks/`, `results/phase5/`, `results/phase6/`, and
rendered result images, ~560 MB) are stored with [Git LFS](https://git-lfs.com); the small
summary outputs are in normal git and need no special handling.

To fetch the full data, install Git LFS **once per machine**, then clone or pull:

```bash
# 1. Install git-lfs
#    macOS:  brew install git-lfs
#    Ubuntu: sudo apt-get install git-lfs
git lfs install

# 2a. Fresh clone — LFS objects download automatically
git clone git@github.com:19PINE-AI/interaction-scaling.git

# 2b. Already cloned (or cloned without git-lfs installed)? Fetch the large objects:
git lfs pull
```

Without Git LFS the large files appear as small text *pointer stubs* instead of the real
data; `git lfs pull` replaces them with the actual result dumps. The code, task suites,
paper source, and small summary results all work fine without LFS.

---

## Quickstart

The project uses [`uv`](https://github.com/astral-sh/uv) and Python ≥ 3.11.

```bash
# Install dependencies
uv sync

# Install the headless browser used for visual rendering
uv run playwright install chromium

# Provide API keys for the model providers you intend to use
export ANTHROPIC_API_KEY=...    # Claude proposers/reviewers
export OPENAI_API_KEY=...       # GPT / cross-model baselines
# (Gemini key as required by the video/visual judges)
```

A minimal end-to-end run of the proposer–reviewer code harness:

```bash
# Scaling-curve comparison (reasoning vs sampling vs interaction) on hard code tasks
uv run python scripts/run_scaling_curves_code.py

# Feedback-type controls (isolate Type 3a execution from extra reviewing)
uv run python scripts/run_feedback_type_controls.py

# Budget allocation sweep across think/do/review
uv run python scripts/run_allocation_sweep.py
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full catalog of runnable scripts
and what each one reproduces.

---

## Building the paper

```bash
cd paper
latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/main.pdf` is the built output. See [`paper/README.md`](paper/README.md).

---

## Citation

```bibtex
@misc{li2026interactionscaling,
  title  = {Grounding the Loop on Both Sides: Interaction as a Third Test-Time
            Compute Axis, and Why Its Gains Are Invisible Without Grounded Evaluation},
  author = {Li, Bojie},
  year   = {2026},
  note   = {Pine AI},
}
```

## License

Internal Pine AI research. All rights reserved unless stated otherwise.
