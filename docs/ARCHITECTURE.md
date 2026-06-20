# Architecture

This document tours the `src/` library and explains how the proposer–reviewer interaction
loop is assembled from its parts. For the research narrative and results, see the paper in
[`../paper/`](../paper/) and the per-phase logs in [`../notes/`](../notes/).

## The interaction loop

The unit of study is a single **budgeted proposer–reviewer loop**:

```
            ┌────────────────────────────────────────────────┐
            │                                                 │
  task ──▶ Proposer ──▶ artifact ──▶ Environment ──▶ feedback │──▶ Reviewer ──▶ structured
            ▲            (code,        (execute /              │      (analyze     critique
            │             page,         render /               │       grounded    + revise
            └─────────────slides,──────  fact-check)───────────┘       signal E)     plan)
                          video)
                  revise ◀──────────────────────────────────────────────────────────┘
```

The same architecture applies across all five task modalities; only the **feedback
provider** (and the renderer it needs) changes. A meta-controller decides, given the
remaining budget, whether to propose, execute, review, or submit.

## Modules

### `src/agents/` — the loop participants
- `base.py` — shared agent interface.
- `proposer.py` — generates / revises the artifact from the task spec and any prior
  feedback.
- `reviewer.py` — consumes the environment response `E` and produces a *structured*
  improvement signal (not raw output).
- `single_agent.py` — single-shot baseline (no review loop) for comparison.
- `meta_controller.py` — budget-aware controller that allocates the next action across
  think / do / review / submit.

### `src/feedback/` — the grounded-feedback taxonomy
One module per feedback type; this is the experimental core of the paper.
- `type0_self.py` — self-review (same model, same weights). DPI-bounded.
- `type1_cross.py` — LLM cross-review (different model/prompt). Still DPI-bounded.
- `type2_static.py` — static tools (lint, type-check, structural validation).
- `type3a_execution.py` — test execution, error messages, runtime behavior.
- `type3b_visual.py` — rendered-artifact visual feedback (VLM **and** the deterministic
  geometric instrument).
- `type3c_video.py` — temporal feedback from video/animation keyframes.
- `type3d_factual.py` — fact verification via search / knowledge base.

### `src/budget/` — test-time compute accounting
- `tracker.py` — counts tokens (and wall-clock / API cost) across all calls in a run.
- `allocator.py` — splits a fixed budget `B` across proposal `b1`, execution-processing
  `b2`, and review `b3`. Powers the allocation-sweep experiments.

### `src/evaluation/` — grounded measurement
The "grounded evaluation" half of the thesis lives here.
- `code_eval.py` — pass/fail against test suites.
- `geometric_checker.py` — **deterministic DOM-geometry instrument**: measures overlap,
  clipping, overflow, and box-group misalignment exactly from the rendered layout. This is
  the metric that makes visual gains visible where a VLM judge is blind.
- `checklist_judge.py` — multi-axis binary rubric scoring (replaces holistic 0–1 VLM
  scores).
- `gemini_video_judge.py` — native full-video rubric judging for the video modality.
- `interaction_loop_eval.py` / `autonomous_review_eval.py` — drive a full loop and score
  the result.
- `metrics.py`, `run_checkpoint_eval.py` — aggregation and checkpoint evaluation.

### `src/rendering/`
- `browser.py` — headless-browser (Playwright) rendering used by the visual and geometric
  feedback/evaluation paths.

### `src/benchmarks/`
- `humaneval.py`, `mbpp.py` — code-benchmark loaders. Hardened, de-saturated task suites
  live under [`../data/hard_benchmarks/`](../data/hard_benchmarks/).

### `src/training/` — internalizing the loop (distillation)
The Contribution-5 pipeline: can an 8B student learn to check its own work?
- Trace collection: `collect_traces.py`, `collect_review_traces_v*.py`,
  `collect_agentic_traces.py`, `collect_vl_traces.py`.
- Data prep: `prepare_agentic_sft.py`, `prepare_vl_sft.py`, `build_sft_v4_data.py`,
  `stitch_trajectory*.py`, `scrub_leaky_comments.py`.
- Training: `train_sft_review*.py`, `train_grpo*.py`, `train_vl_sft.py`, `rft_vl.py`.
- Students + judging: `run_agent_student.py`, `run_vl_student.py`,
  `judge_*_traces.py`, `evaluate.py`.

### `src/experiments/` — orchestration
- `runner.py`, `hard_benchmark_runner.py` — experiment drivers.
- `exp1_feedback_ablation.py` — feedback-type ablation.
- `exp2_scaling_curves.py` — reasoning vs sampling vs interaction scaling curves.
- `exp5_verification_gap.py` — the verification-gap analysis.

### `src/analysis/` and `src/utils/`
- `analysis/generate_tables.py`, `analysis/plot_results.py` — paper tables and figures.
- `utils/llm_client.py` — unified multi-provider LLM client.
- `utils/code_utils.py` — code extraction / sandbox helpers.

## How a run flows

1. A script in `scripts/` (or `src/experiments/`) loads a task suite from `data/`.
2. The **meta-controller** runs the proposer–reviewer loop under a token **budget**.
3. The configured **feedback provider** renders/executes the artifact and returns grounded
   feedback `E`.
4. The **reviewer** turns `E` into a structured revision signal; the proposer revises.
5. On submit, an **evaluation** module scores the final artifact with a grounded metric.
6. **Analysis** aggregates results into tables/plots; findings are written up in `notes/`.
