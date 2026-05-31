# Phase 5: RL Infrastructure (RFT) — Implementation

## Choice: RFT instead of full GRPO

Built **Rejection-sampling Fine-Tuning** (RFT, also "expert iteration") rather than full GRPO. Reasoning:

1. **Multimodal trajectory rollouts are expensive** (~5 min per trajectory). PPO requires tightly-coupled rollout/update loops; that means GPU-resident rollouts blocking model updates. With our agent loop (5 turns × generate + render + generate), each PPO step would be hours.
2. **Reward is binary** (judge keep / reject). RFT treats it as supervised positive examples; no need for advantage estimation, KL clipping, ratio terms.
3. **TRL's GRPOTrainer doesn't support multi-turn agent rollouts** out of the box. Building a custom multimodal RL loop is a 1-2 week project; RFT is a 1-day project.
4. **Expert iteration is theoretically equivalent** to GRPO with deterministic baseline subtraction when rewards are binary and you only train on positives.

## Implementation

`src/training/rft_vl.py` — end-to-end pipeline:

1. **Rollout step**: load V3+fc=4000 student, run agent loop on each training task with N seeds. Saves trajectories to JSON (incremental, resumable).
2. **Judge step**: re-renders final artifact, sends to Gemini 3 Flash multimodal. Marks each trajectory with `keep=True/False`.
3. **Build step**: convert kept trajectories into Qwen3-VL chat-format SFT JSONL (re-renders images on disk).
4. **Train step**: run `train_vl_sft.py` on the produced JSONL with the V3+fc=4000 adapter as starting point and lower LR.

## Run plan

```bash
# 1. Rollout (~12 hours for 37 tasks × 4 seeds with 1 worker)
PYTHONPATH=. python -m src.training.rft_vl \
  --adapter models/qwen3-vl-8b-vl-sft-v3 \
  --tasks data/training/rft_tasks.json \
  --rollouts-per-task 4 \
  --temperature 0.7 \
  --trajectories-out data/training/vl_rft_trajectories.json \
  --judged-out data/training/vl_rft_trajectories_judged.json \
  --rft-jsonl-out data/training/vl_rft_sft.jsonl

# 2. RFT training (~12 minutes)
PYTHONPATH=. python -m src.training.train_vl_sft \
  --data data/training/vl_rft_sft.jsonl \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --output models/qwen3-vl-8b-vl-rft-v1 \
  --epochs 3 --batch-size 1 --grad-accum 4 --lr 5e-5 \
  --lora-rank 16 --lora-alpha 32 --max-seq-len 22000

# 3. Eval (~70 min)
PYTHONPATH=. python -m src.training.run_vl_student \
  --adapter models/qwen3-vl-8b-vl-rft-v1 \
  --tasks data/training/heldout_phase5.json \
  --max-turns 5 --temperature 0.3 --repetition-penalty 1.1 \
  --force-close-think-after 4000 \
  --output results/phase5/student_rft_v1_full.json

PYTHONPATH=. python -m src.training.judge_vl_traces \
  --traces results/phase5/student_rft_v1_full.json \
  --output results/phase5/student_rft_v1_full_judged.json
```

## Expected behavior

- **If RFT helps**: lifts judge-keep above V3+fc=4000's 44%. The student bootstraps better trajectories on the same training tasks; over multiple iterations capability ratchets up.
- **If RFT plateaus**: the student is already doing as well as it can on the training distribution. To push further would need either bigger student model or fundamentally different teacher signal (e.g. multi-step DPO with paired good/bad trajectories).
- **Risk**: catastrophic forgetting of V3 capabilities. Mitigated by low LR (5e-5 vs SFT's 2e-4) and few epochs (3).

## Why we're not running it now

GPU currently occupied by generalization pass@k sample 2 (44 tasks, ~3 hours remaining). RFT rollout would take ~12 more hours after that. Total ~15 hours of inference for one RFT round.

If pursued, recommended sequence:
1. First do a 1-task pilot (~5 min) to verify pipeline plumbing
2. Then a 5-task pilot (~25 min) to see if any judge-kept trajectories appear
3. Only then commit to the full 37-task × 4-seed rollout

## Why not full GRPO

Full GRPO on multimodal multi-turn agent loops would need:
1. **Custom rollout collector** that runs agent loop, captures per-token log-probs, and tags assistant-vs-user tokens.
2. **Reference policy snapshot** for KL divergence (extra 16GB GPU memory).
3. **PPO-style update**: per-token advantage × log-prob ratio with clipping. Token-level credit assignment is fraught when reward is end-of-trajectory only — typically requires per-token Monte Carlo or value function.
4. **Multimodal batching**: image-token positions vary per trajectory; standard GRPO trainers assume fixed-shape inputs.

Estimated effort: 1-2 weeks of careful infra work, with a real risk that the result matches RFT performance anyway (binary-reward GRPO often does). Deferred until RFT shows the headroom is worth it.

