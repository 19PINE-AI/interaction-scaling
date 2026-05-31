# Phase 2 Findings: SFT + GRPO on Qwen3-8B for Interaction Scaling

**Date:** 2026-04-17
**Model:** Qwen/Qwen3-8B (8.2B params, loaded in 4-bit nf4)
**Adapters:** LoRA r=16, α=32 on `q,k,v,o,gate,up,down_proj` (43.6M trainable, 0.53%)
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (97GB)

## Training

| Stage | Examples | Epochs | Steps | Wall time | Final loss | Final reward |
|-------|---------:|-------:|------:|----------:|-----------:|-------------:|
| SFT   | 970      | 3      | 183   | 1h 22m    | 0.30       | —            |
| GRPO  | 638      | 1      | 79    | 6h 17m    | ~0 (stable)| 0.23–0.28    |

GRPO scoped down (1 epoch, num_generations=2 instead of 3/4) due to shared-GPU contention. Step time halved from 360s to 200s partway through when the co-tenant finished.

## Evaluation: in-distribution code benchmark

Single-turn inference on the 15-task `code_tasks.json` benchmark (deterministic `test_code` harness). **All 15 tasks appear in the SFT/GRPO training data** — this is an in-distribution comparison measuring what the training actually taught, not held-out generalization.

Prompt format matches training: system prompt with budget-aware `[GENERATE]/[EXECUTE]/[REVIEW]/[SUBMIT]` protocol, `enable_thinking=False` for Qwen3, greedy decoding, max_new_tokens=1536.

### Headline numbers

| Checkpoint | pass@1 | Δ vs base | Δ vs SFT |
|------------|-------:|----------:|---------:|
| Base Qwen3-8B | 6/15 = **40.0%** | —       | —        |
| + SFT        | 10/15 = **66.7%**| **+26.7%** | —      |
| + GRPO       | 9/15 = **60.0%** | +20.0%  | **−6.7%** |

### Per-task outcomes

```
task     base  SFT  GRPO
001       P    P    P
002       .    P    P
003       .    .    .
004       .    .    .
005       P    P    P
006       P    P    P
007       .    P    P
008       .    .    .
009       .    P    .    ← GRPO regression
010       .    .    .
011       .    .    .
012       P    P    P
013       .    P    P
014       P    P    P
015       P    P    P
```

Six tasks are solved by all three (trivially memorizable); five tasks are beyond all three (003, 004, 008, 010, 011 — the hardest bugs). **SFT gains +4 tasks over base; GRPO gains +3 (the same four minus code_009).**

### code_009 regression

SFT passes, GRPO fails with `AssertionError` on the test harness. Response length nearly identical (1966 vs 1968 chars) — GRPO didn't run out of tokens, it produced subtly wrong code. The policy gradient pushed the model away from the SFT answer that worked.

## Observations

1. **SFT is the load-bearing stage.** +26.7 absolute points from SFT alone; GRPO adds nothing positive and regresses one task. This is consistent with the small GRPO run (79 steps, 1 epoch) and the low/flat reward signal (0.23→0.28 mean across training) — the policy barely moved.

2. **Format internalization is real.** Response length drops from 2161 chars (base, verbose prose) to ~1760 (SFT/GRPO, terse `[GENERATE]`+code). The base model sometimes overflows the 1536-token budget (1/15); SFT/GRPO never do. The multi-turn budget-aware protocol was learned.

3. **GRPO was starved of signal.** 89% of training completions hit `max_completion_length=2048` (clipped). Reward hovered at 0.25 — about chance if rewards are 0/1 bernoulli on the reviewer's threshold. With weak differential signal and truncated rollouts, GRPO nudged the policy without improving it.

4. **The hardest tasks remain hard.** Tasks 003, 004, 008, 010, 011 are unsolved by all three checkpoints. These involve timezone arithmetic, RFC-6902 JSON patch escaping, and other multi-constraint edge-case-heavy problems. SFT on 970 trajectories wasn't enough to teach the model to reason through these; neither was 79 GRPO steps.

## Limitations

- **No held-out split.** All eval tasks were in training. The +26.7% SFT gain measures memorization + format learning, not generalization to new code. A proper test would use fresh bug-fix tasks (e.g., SWE-Bench Lite subset) or hold out a stratified slice before training.
- **Single-turn inference.** The training data is multi-turn interaction trajectories, but this eval asks for a single-shot answer. The interaction-scaling hypothesis (that inference-time loops recover more reward than one-shot) is not tested here. A follow-up should run the actual agent loop (execute → review → revise) at inference.
- **GRPO undertrained.** 1 epoch × 79 steps on 638 examples. A longer run, higher `num_generations`, and a less-truncating `max_completion_length` may yield different conclusions.
- **Placeholder rewards.** Visual/video/factual reward functions still return 0.5 heuristic values per the earlier phase-1 notes. Only the code reward (deterministic test pass) is grounded.

## Suggested next steps

1. **Held-out eval.** Carve ~5 new code bugs, or pull a small HumanEval/SWE-Bench-Lite subset, and rerun base/SFT/GRPO.
2. **Multi-turn inference loop.** Implement the GENERATE→EXECUTE→REVIEW→SUBMIT loop at test time and measure pass rate vs budget. This is the actual interaction-scaling curve the paper claims to internalize.
3. **Longer GRPO with grounded reward only.** Restrict GRPO to the modalities with real rewards (code), raise `max_completion_length` to 3072, and run 3+ epochs. Compare to this run.
4. **Ablate GRPO regression.** Is code_009 a one-off, or does GRPO systematically overwrite correct SFT answers? Rerun with different seeds and `beta` (KL penalty) values.

## Artifacts

- Eval script: `src/evaluation/run_checkpoint_eval.py`
- Raw results: `results/hard_benchmarks/checkpoint_eval.json`
- Adapters:
  - `models/qwen3-8b-interaction-scaling-sft/` (87MB)
  - `models/qwen3-8b-interaction-scaling-grpo/` (87MB)
- Training logs: `logs/training.log` (SFT), `logs/grpo.log` (GRPO)
