# Phase 4 final findings: autonomous review, and the limits of distribution-bound adapters

**Date:** 2026-04-22
**Scope:** Phase 4 training runs (SFT v2/v3/v4, GRPO v1/v2) + two held-out eval sets (original 15 hand-curated, expanded 32 auto-generated).
**Eval logs:** `logs/heldout_sft_v4_only.log`, `logs/heldout_grpo_v2.log`, `logs/heldout_v2_base_vs_grpo_v2.log`.
**Results:** `results/autonomous_review/heldout_sft_v4_only.json`, `heldout_grpo_v2.json`, `heldout_v2_base_vs_grpo_v2.json`.

## Headline

On hand-curated hard tasks (15 tasks, 5 samples each):

| Model | pass@1 | pass@5 | tool@1 | tool@5 |
|---|---|---|---|---|
| base Qwen3-8B | 22.7% | 40.0% | 80.0% | 100.0% |
| SFT v2 (scrubbed) | 24.0% | 46.7% | 100.0% | 100.0% |
| SFT v3 (hidden-fix teacher + forced-retry) | 32.0% | **53.3%** | 100.0% | 100.0% |
| SFT v4 (v3 + stratum rebalance + masked loss + 3-revise) | 30.7% | **53.3%** | 89.3% | 100.0% |
| **GRPO v2 (retry-aware reward from SFT v4)** | **36.0%** | **53.3%** | 94.7% | 100.0% |

**+13.3 pp pass@5 over base.** pass@5 plateaus at 53.3% from v3 onward — the ceiling is set by 7 tasks (h001–h004, h007, h013, h015) that no adapter variant passes even once out of 5 samples.

On auto-generated tasks (32 tasks, Qwen3-235B via OpenRouter, 5 samples each):

| Model | pass@1 | pass@5 | tool@1 | tool@5 |
|---|---|---|---|---|
| base Qwen3-8B | **86.9%** | 93.8% | 100.0% | 100.0% |
| GRPO v2 | 77.5% | 93.8% | 99.4% | 100.0% |

Both saturate at pass@5 = 93.8% (30/32 tasks). GRPO v2 actually *regresses* pass@1 by 9.4 pp vs base on this distribution.

## What we changed across SFT iterations

Starting point (SFT v1, 2026-04-19): teacher (Claude Sonnet 4.6) wrote post-hoc narrations given both the buggy and reference-fixed code, then emitted one draft tool_call + one revise tool_call. Stitching produced 203 SFT examples split 30% no_revise / 70% one_revise. Training was plain SFT on chat-template-formatted trajectories. Held-out pass@1 = 26.7% (tied with base, +20 pp tool@1, zero pass-rate gain).

**Fix 1 (v2, 2026-04-20):** scrubbed `# BUG: ...` comments from training `buggy_code`. 46% of v1 training examples had Sonnet-emitted bug-location comments; the student learned to emit self-labeled buggy code literally. After scrubbing, pass@1 = 40%, +13.3 pp. Zero algorithm change, pure data fix.

**Fix 2 (v3, 2026-04-21):** two simultaneous structural changes:
- `collect_review_traces_v3.py`: teacher no longer sees `task["fixed_code"]`. Given only the buggy code and the stderr from executing it, Sonnet derives both diagnosis and fix. Previous teacher diagnoses contained tokens that existed only in the reference fix (35% of v2 diagnoses), so the student couldn't replicate the reasoning at inference.
- `augment_forced_retry.py`: synthesized a `two_revise` stratum by asking Sonnet to write a deliberately-partial fix that still fails, then a real fix. 48/57 augmented trajectories had ≥3 tool_calls. This was the first training pipeline with multi-retry trajectories.

v3 pass@1 = 32.0%, pass@5 = **53.3%**. First adapter to beat base at pass@5 on held-out.

**Fix 3 (v4, 2026-04-21):** three changes on top of v3:
- Added a `three_revise` stratum via `augment_three_revise.py` (two forced partial fixes before the real fix; 108 new 4-call trajectories).
- Rebalanced to 0% no_revise / 30% one_revise / 40% two_revise / 30% three_revise (150 examples total).
- Masked loss (`train_sft_review_masked.py`): labels=-100 on user + draft-assistant + first-tool-response; loss only on revision+confirm turns. 44% of tokens masked. Plain `Trainer`, not `SFTTrainer`, to support pre-tokenized labels.

Teacher also switched Anthropic → OpenRouter Qwen3-235B (same-family distillation; ANTHROPIC_API_KEY expired).

v4 pass@1 = 30.7%, pass@5 = 53.3%. Tied v3 on pass@5, slightly lower pass@1. Masked loss may under-train draft-turn tool-call syntax — tool@1 fell from 100% → 89.3%. The 3-revise stratum did not unlock new tasks.

**Fix 4 (GRPO v2, 2026-04-22):** retry-aware reward in `train_grpo_review_v2.py`:
- +1.5 if ≥2 tool_calls emitted AND last call passes tests
- +1.0 if exactly 1 tool_call AND it passes
- +0.1 any valid tool_call
- 0.0 no valid tool_call

Warm-started from SFT v4. 51 steps, ~82 min. pass@1 = 36.0%, pass@5 = 53.3%. Densified wins within the already-solvable 8 tasks (27/75 sample passes vs v4's 23/75) but did not unlock any new task. Same 8/15 solved, same 7/15 unsolvable.

## Why pass@5 plateaus at 53.3%

Two independent lines of evidence that the 7 unsolvable tasks are at Qwen3-8B's capability ceiling, not pathological:

1. **Every task is legitimately hard**: RFC-compliant HTTP header continuation (h002), POSIX shell tokenization with quote-context-aware backslash (h003), skip list (h007) duplicate-key semantics, grapheme clusters (h013), HTTP Accept q-value negotiation (h015). Each requires 8–14 test assertions and multiple coordinated fixes. None are malformed or ambiguous.

2. **Qwen3-235B can't even organically generate tasks that hard.** We asked Qwen3-235B to generate 30 fresh held-out tasks in the same schema and categories as the hand-curated set. Base Qwen3-8B passed 30/32 of the generated tasks (pass@5 = 93.8%) — a 53.8 pp jump over its performance on the hand-curated set. The distribution of tasks the generator produces sits far below the difficulty of the hand-curated hard set.

Conclusion: the 7 unsolvables were selected (implicitly, during hand-authoring) to be hard for an 8B-class model. Training on 235B-generated traces can't push the capability ceiling above what 235B can itself generate.

## Why GRPO v2 regressed on the easy distribution

On the auto-generated set, GRPO v2 loses 9.4 pp pass@1 to base (77.5% vs 86.9%). The retry-aware reward incentivized emitting ≥2 tool_calls, which on trivially-solvable tasks means writing a correct fix on the first call and then either:
- "second-guessing" with a worse second call that trips a test, or
- Running the 2048-token completion budget into an unproductive third tool_call.

Per-task: base gains hv2_032 where GRPO has 0/5; GRPO gains hv2_014 where base has 0/5; same 30/32 pass@5. The adapter trades first-shot consistency for retry willingness — beneficial when the base is in the 20–50% regime where retries can surface a missed bug, harmful when the base is at 80%+ where retries just introduce noise.

## Why the calls=1 collapse persists

Despite the 3-revise stratum (v4) and retry-aware GRPO reward (v2), 21/27 passing GRPO v2 samples on the hard set use calls=1. The model learned that for the kind of tasks it *can* solve, one well-written tool_call suffices; multi-call trajectories appear only when the first call fails a test the model wouldn't have anticipated. In other words: the retry behavior is triggered by real feedback from a failing tool_call, not by a learned disposition to double-check. This is reasonable policy for the tasks we have, but means the training never exercised the "I'll try a totally different approach" behavior we were hoping to distill.

One hypothesis: SFT on forced-retry traces teaches *surface form* of retry, but the policy collapses back to its first-shot median at inference because the reward signal doesn't actually pay for retries-that-change-strategy — only for retries-that-eventually-pass. A retry that repeats a variant of the same code gets rewarded equally to one that switches approach.

## Interaction scaling thesis: current status

The original paper thesis: *SFT/GRPO at N=1 matches or exceeds base at N=5 — i.e., the scaffold is internalized into the weights and no longer needs explicit inference-time multi-call.*

- Base: pass@1 = 22.7%, pass@5 = 40.0%. Gap: +17.3 pp.
- GRPO v2: pass@1 = 36.0%, pass@5 = 53.3%. Gap: +17.3 pp.

The adapter shifts the whole curve up by ~13 pp but **does not close the pass@1 ↔ pass@5 gap** (the gap is identical for base and GRPO v2). The thesis is therefore **not supported in the strong form**: multi-call at inference still buys the same ~17 pp that it buys for base. What the adapter actually did was improve single-shot code quality; the multi-call behavior it inherited from SFT is only triggered by real failure feedback and fires at the same rate as base's natural "I'll call the tool again if the first failed" policy.

**What the adapter did achieve:** a +13.3 pp pass@5 gain over base, uniformly, via cleaner first-shot code. This is a defensible "distillation works for this problem" result but it's not interaction scaling per se.

## Artifacts

- Training scripts: `src/training/{train_sft_review,train_sft_review_masked,train_grpo_review,train_grpo_review_v2}.py`
- Data pipeline: `src/training/{collect_review_traces_v3,augment_forced_retry,augment_three_revise,build_sft_v4_data,stitch_trajectory_v3}.py`
- Task generator: `src/training/{generate_tasks_v2,generate_heldout_v2}.py`
- Eval harness: `src/evaluation/autonomous_review_eval.py` (supports `base`, `sft_review`, `sft_review_v2/3/4`, `grpo_review`, `grpo_review_v2`)
- Held-out sets: `data/hard_benchmarks/code/{code_tasks_heldout,code_tasks_heldout_v2}.json`
- Adapters: `models/qwen3-8b-autonomous-review-{sft,sft-v2,sft-v3,sft-v4,grpo,grpo-v2}/`

## Strategic next steps

Three classes of moves, ranked by expected research value per unit engineering cost.

### High leverage, low cost

**S1. Extend max_tool_calls and re-eval.** Currently capped at 3. Are any of the 7 unsolvable tasks actually within reach given 5–10 tool calls? If yes, the ceiling is *budget-bound*, not *capability-bound*, and a budget-scaling experiment becomes the paper's centerpiece. If no, the ceiling is confirmed capability-bound. Cost: single eval run, ~4h. Decision-altering regardless of outcome.

**S2. pass@k scaling curve at k = 1, 5, 10, 20, 50.** The current pass@1 ↔ pass@5 gap is identical for base and GRPO v2 (+17.3 pp). Does that hold at larger k? If base's pass@50 catches GRPO v2's pass@5, the adapter is strictly a compute-shift; if GRPO v2 keeps opening a gap at higher k, interaction scaling has a real (if modest) signal. Cost: one eval run at k=50 on 15 tasks = 750 samples, ~1.5 days. Worth it — this is the actual thesis test.

### Medium leverage, medium cost

**S3. Medium-difficulty training data.** Current training tasks are easy (Qwen3-235B generates, Qwen3-8B passes most of them first-shot). The student never genuinely struggles during training, which is why the multi-call behavior doesn't transfer. Approach: filter training tasks to ones where base Qwen3-8B passes only 20–60% of the time (seed the model first, score, keep the medium band). Retrain SFT on only this subset. Re-eval. Hypothesis: training in the "struggle regime" produces a student that actually exercises retry behavior at inference. Cost: 1–2 days of pipeline work + one SFT run.

**S4. Expert-iteration / rejection-sampling on hard tasks.** Use GRPO v2 itself to generate K trajectories per hard task, keep only trajectories where final_pass=True AND calls ≥ 2, SFT on those. Reinforces *successful* multi-call behavior rather than synthetic-forced retries. Cost: 1 day of sampling + 1 SFT run.

### High leverage, high cost

**S5. Real multi-turn GRPO with interleaved tool execution.** The current GRPO v2 rewards flat completions that happen to contain ≥2 tool_call blocks — but the model has no real tool feedback between its own emitted calls, so the retry emission is disconnected from outcomes. A proper multi-turn rollout (generate → execute → feed response back → generate next turn → execute → … → score final state) would actually train the tool-use policy. TRL 1.1.0 doesn't support this natively; would need a custom trainer built on `generate()` + subprocess tool execution. Cost: 1–2 weeks of code + 12+ hr training per iteration. Highest upside; highest risk.

**S6. Stronger base model.** Distill to Qwen3-14B instead of 8B. Would lift the capability ceiling above the current 7 unsolvables. But loses the "small model with big-model scaffold" storyline if that's the paper's angle. Cost: identical pipeline, 2–3× longer training per step, 2× memory.

### What the paper probably needs next

If the paper's thesis is **"interaction scaling internalized into weights,"** then S1 + S2 are the tests we have to run. S2 especially — the pass@k scaling curve at k=50 is the definitive measurement of whether the adapter is doing anything at inference that base couldn't do with more samples.

If the paper's thesis can be reframed as **"cleaner teacher data produces cleaner distilled code, independent of interaction structure,"** then the v2 → v3 data-side fixes are the centerpiece and S3/S4 sharpen them. The interaction-scaling framing becomes a negative-result subsection: "we tried to teach retry behavior three different ways (forced strata, masked loss, retry-aware reward); all three produced smaller-than-hoped transfer because the multi-call behavior only fires when the first tool call actually fails, regardless of training signal."

**Recommendation:** run S1 + S2 first (~2 days, one GPU). Those results decide the framing. If S2 shows GRPO v2 opening a growing pass@k gap over base, the paper is an interaction-scaling positive. If it closes, the paper pivots to a teacher-data-quality story and S3/S4 become the next training experiments.
