# Research record

This directory is the running lab notebook for *Interaction Scaling* — one Markdown file
per experiment or phase, written as the work happened. It is the primary-source trail behind
the paper: every headline number in [`../paper/`](../paper/) traces back to a findings file
here and to the eval outputs in [`../results/`](../results/).

Read the paper for the polished narrative; read these for the raw record, dead ends, and
exact configurations. Files are listed newest-thinking-last within each group.

## Phased build-up (the core arc)

| File | What it covers |
|---|---|
| [`phase1_findings.md`](phase1_findings.md) | Proposer–reviewer vs single-shot on code; the base harness result |
| [`phase1_reviewer_weaknesses.md`](phase1_reviewer_weaknesses.md) | Where the reviewer helps and where it does not |
| [`phase2_findings.md`](phase2_findings.md) | Extending the harness across modalities |
| [`phase3_heldout_findings.md`](phase3_heldout_findings.md) | Held-out generalization checks |
| [`phase4_final_findings.md`](phase4_final_findings.md) · [`phase4_internalized_review_plan.md`](phase4_internalized_review_plan.md) | Internalizing the review step |
| [`phase5_*`](.) | Distillation into an 8B student (v3/v4 sweeps, RFT infra, final + paper cuts) |
| [`phase6_findings.md`](phase6_findings.md) · [`phase6_status.md`](phase6_status.md) | Agentic pipeline experiments |

## Scaling & feedback-type controls

- [`scaling_curves_findings.md`](scaling_curves_findings.md) · [`scaling_curves_seeds_findings.md`](scaling_curves_seeds_findings.md) — reasoning vs sampling vs interaction at matched token budgets (multi-seed)
- [`reasoning_baseline_findings.md`](reasoning_baseline_findings.md) — reasoning-only ceiling
- [`feedback_type_controls.md`](feedback_type_controls.md) — isolating Type 3a execution from extra reviewing
- [`allocation_findings.md`](allocation_findings.md) — think/do/review budget allocation sweep
- [`iad_baseline_findings.md`](iad_baseline_findings.md) — iterative-agent-debate baseline

## Grounded evaluation & benchmark hardening

- [`finding_08_hard_benchmark_design.md`](finding_08_hard_benchmark_design.md) · [`finding_09_hard_code_results.md`](finding_09_hard_code_results.md) · [`finding_10_hard_benchmark_summary.md`](finding_10_hard_benchmark_summary.md) — designing the de-saturated hard suites
- [`rubric_geometric_findings.md`](rubric_geometric_findings.md) — geometric instrument vs VLM judge
- [`slides_web_hardening_findings.md`](slides_web_hardening_findings.md) · [`video_slides_hard_findings.md`](video_slides_hard_findings.md) — hardened slides/web/video suites

## Cross-model replication

- [`cross_model_findings.md`](cross_model_findings.md) · [`cross_model_v2_findings.md`](cross_model_v2_findings.md) — Sonnet / Qwen3-235B / GPT-5 replication with seeds

## Held-out & distillation

- [`heldout_phase1_findings.md`](heldout_phase1_findings.md) — zero-overlap held-out code suite
- [`held_out_distillation_plan.md`](held_out_distillation_plan.md) · [`contribution_5_training.md`](contribution_5_training.md) — distillation write-up

## Inspection dumps

- [`experiment_log.md`](experiment_log.md) — chronological index of runs
- [`agentic_inspection_full/`](agentic_inspection_full/), [`agentic_pilot_inspection/`](agentic_pilot_inspection/), [`sft_trace_inspection/`](sft_trace_inspection/) — per-example trace inspections
