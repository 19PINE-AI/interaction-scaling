# Held-Out Distillation Plan

**Date:** 2026-04-17
**Problem:** Phase 2 eval reported +26.7% SFT gain on the 15 code tasks, but all 15 were in the training set. We conflated memorization with generalization. This plan rebuilds the student pipeline so the evaluation set is strictly disjoint from what the student ever sees.

## 1. Split discipline

| Set | Role | Size | Source |
|---|---|---:|---|
| `code_heldout_15` | Final held-out eval. **Student never trains on this.** | 15 | Existing `data/hard_benchmarks/code/code_tasks.json` (our hard benchmark) |
| `code_train_N` | Student SFT + GRPO training pool | 60–100 | Newly generated, same distribution, disjoint instances |

Invariants:
- Every `task_id` in `code_train_N` is globally unique and not a string-match or near-duplicate of any `code_heldout_15` task.
- Same distribution: RFC/spec-level correctness bugs, state-machine bugs, timezone/encoding edge cases, parsing/routing bugs, numerical-precision bugs — matched by difficulty tier.
- Each training task has a deterministic `test_code` harness (pass/fail), same schema as the held-out set.
- Deduplication: cosine similarity on task description embeddings; drop any pair with cos > 0.85.

## 2. Training task generation

Two paths, in order of preference:

**(a) Curated handwrite (~60 tasks).** Use the held-out taxonomy to seed fresh instances: if the held-out has "CSV parser escape-quote bug," we hand-author "CSV parser multi-line field bug" and "TSV parser tab-in-quoted-field bug" — same failure mode family, different surface.

**(b) Teacher-generated augmentation (+30–40 tasks).** Prompt Claude Sonnet with the held-out taxonomy minus the specific instances, ask it to generate new bug-in-function tasks with test harnesses, then manually review and dedupe against held-out. This is cheap but needs human review for distinctness.

Target: 80 tasks across 6–8 bug-family buckets (≥8 per bucket).

## 3. Teacher trajectory collection

For each training task, run the Claude Sonnet 4 multi-turn agent loop (`GENERATE → EXECUTE → REVIEW → SUBMIT`) with budget=5 and the deterministic `test_code` reward.

- Keep only passing trajectories for SFT.
- Keep all trajectories (pass + fail) for GRPO, with the pass/fail signal as reward.
- Record turn count and final token budget — these become the "interaction scaling curve" the student tries to internalize.

Output files:
- `data/training/sft_heldout.json` — passing teacher trajectories only
- `data/training/grpo_heldout.json` — all trajectories with reward

## 4. Student retraining

Reuse the existing QLoRA + GRPO pipeline on Qwen3-8B:
- SFT: 3 epochs, same hyperparameters as before.
- GRPO: 3+ epochs, `max_completion_length=3072` (up from 2048 — 89% of prior rollouts were clipped), `num_generations=2`, grounded reward only (deterministic `test_code` pass/fail).

New adapter paths:
- `models/qwen3-8b-interaction-scaling-sft-heldout/`
- `models/qwen3-8b-interaction-scaling-grpo-heldout/`

## 5. Evaluation matrix

2D: `{base, SFT, GRPO, teacher (Claude Sonnet 4)} × {N=1, 2, 3, 5 turns}` on the **15 held-out tasks**.

|           | N=1 (single-shot) | N=2 | N=3 | N=5 |
|-----------|:-:|:-:|:-:|:-:|
| Base Qwen3-8B |  |  |  |  |
| + SFT  |  |  |  |  |
| + GRPO |  |  |  |  |
| Teacher (Sonnet) |  |  |  |  |

What this tells us:
- **Rows (fix N, vary model):** did distillation actually transfer the skill? Base vs SFT vs GRPO at N=1 tells us single-shot capability transfer.
- **Columns (fix model, vary N):** what is the interaction scaling curve per model? This is the main paper figure — slope of pass@1 vs N.
- **Headline claim:** student SFT/GRPO at N=1 should match or exceed base at N=5, i.e. the student internalized what the teacher needed multi-turn loops to achieve. If that holds, interaction scaling transferred into the weights.

## 6. Why this is the right eval

The Phase 2 numbers (Base 40%, SFT 66.7%, GRPO 60%) cannot distinguish:
- The student learned to fix *this class of bugs* (generalization), vs.
- The student memorized the exact 15 answers (overfitting),
- The student learned the `[GENERATE]/[SUBMIT]` format without learning the content.

A held-out split where the training set is the same *distribution* but disjoint *instances* forces the student to actually transfer the skill. If SFT still beats base by a large margin on held-out tasks, generalization claim holds. If it collapses to near-base, we know Phase 2 was memorization.

## 7. Milestones

1. Write 60 handcrafted tasks (~2 days effort, user-led).
2. Teacher-generate 30 augmentation tasks + dedupe (~half day, automated).
3. Run teacher over 80 tasks to collect trajectories (~6h wall, automated).
4. Retrain SFT (~1.5h) and GRPO (~8h with higher `max_completion_length`).
5. Run 2D eval matrix (~2h wall: 4 models × 4 N values × 15 tasks).
6. Update `notes/phase3_heldout_findings.md` with honest generalization numbers.

## 8. Open decisions

- **Code-only vs all modalities.** Code has a deterministic reward, so held-out works cleanly. Extending to slides/video/research requires a held-out *reviewer* signal too — harder, postponable.
- **Teacher-generated vs public benchmark.** Teacher-generated is same distribution but comes from the same model family (risk of stylistic leakage). A SWE-Bench-Lite subset would be independent but different distribution. Recommendation: do both — teacher-gen for the headline claim, SWE-Bench-Lite as a robustness check.
