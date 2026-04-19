# Phase 3 findings: held-out interaction-scaling eval

**Date:** 2026-04-19
**Eval log:** `logs/interaction_scaling_eval.log`
**Results:** `results/interaction_scaling/heldout.json`
**Harness:** `src/evaluation/interaction_loop_eval.py`
**Held-out tasks:** `data/hard_benchmarks/code/code_tasks_heldout.json` (15 tasks, disjoint from training; validated so each task's buggy implementation fails its tests)

## Headline

| Model | N=1 | N=2 | N=3 | N=5 |
|-------|:---:|:---:|:---:|:---:|
| Base Qwen3-8B | **40.0%** | 40.0% | 40.0% | **46.7%** |
| + SFT (v1)    | 26.7% | 20.0% | 26.7% | 26.7% |
| + GRPO (v1)   | 33.3% | 20.0% | 26.7% | 33.3% |

(greedy decoding, `do_sample=False`, max_new_tokens=1536, budget-aware system prompt matching training format)

Three conclusions:

1. **Phase 2 SFT/GRPO gains were memorization, not generalization.** On 15 in-training tasks, SFT reached 66.7% vs. base 40% (+26.7 pp). On 15 held-out tasks of the same distribution, SFT collapses to 26.7% — *worse* than base. The +26.7 pp in-dist gain did not transfer.

2. **Multi-turn revision actively hurts distilled students.** Both SFT and GRPO drop at N=2 (to 20%) and never recover to base's level. The base model is flat-to-slightly-rising across N. Distillation appears to have taught the student to confidently re-emit similar code on review rather than debug.

3. **The headline interaction-scaling claim is not supported by this configuration.** The paper's intended thesis — *SFT/GRPO at N=1 matches or exceeds base at N=5, i.e. interaction scaling internalised into the weights* — fails: SFT(N=1)=26.7% is below base(N=5)=46.7%; GRPO(N=1)=33.3% is below base(N=5)=46.7%. The skill did not transfer into the weights.

## Why

Three contributing factors, in order of how sure we are:

a. **GRPO was starved.** 2048 token completion length clipped 89% of rollouts; `num_generations=2` gave a one-sample baseline so advantages were almost pure noise; 79 steps over 1 epoch. Any signal from the reward function was drowned out. Fix is already staged (`src/training/train_grpo.py`: 8192 / 8 / 3) and a v2 run launched 2026-04-19 05:04.

b. **SFT memorised the 15 training tasks.** 970 trajectories across only 15 distinct tasks means an average 65 trajectories per task. The model learned to produce the exact text of each task's solution. On held-out tasks the learned surface patterns don't apply; the student reverts to something worse than the base model because the finetune has shifted its behaviour away from general coding toward a narrow template.

c. **Greedy decoding kills the interaction signal on base.** Because `do_sample=False`, when the base model is re-prompted with an error and asked to revise, it produces near-identical code. The base curve (40/40/40/47%) is flat for N=1..3 and only flips one task at N=5. A temperature-sampled run is needed before we can claim anything about the base model's interaction scaling.

## Implications for the paper

- **Cannot claim "distillation transfers interaction scaling" from current Phase 2 artifacts.** The claim is false on held-out.
- **Can still claim** (i) reviewer-based multi-turn loops improve teacher output across all six modalities (Phase 1 post-fix results), (ii) the paper's main negative finding: naive imitation distillation of multi-turn trajectories from 15 tasks does not transfer, and standard GRPO at the tried config does not fix it.
- **Must either** scale the training pool and re-run (honest path to the positive claim), **or** reframe the paper as a measurement study of where distillation fails.

## Immediate next steps

1. GRPO v2 retrain (8192 / 8 / 3) — launched 05:04. Expected runtime under new config: likely ~12–24h on a single RTX Pro 6000 Blackwell. Will be re-evaluated on the same 15 held-out tasks under the same harness.
2. Sampled eval (temperature 0.7) on base + SFT + GRPO-v1 as a follow-up — separates the interaction-scaling signal from greedy-decoding flatness.
3. **Expand the training pool to 80 tasks** per `notes/held_out_distillation_plan.md` §2 before drawing stronger conclusions. 15 training tasks is too narrow to generalise from, regardless of how many trajectories each has.

## Per-task detail

See `results/interaction_scaling/heldout.json` for full per-task traces. Tasks where *only* base passes and distilled students fail are particularly diagnostic — those are cases where the finetune demonstrably *regressed* a capability that was in the base model.
