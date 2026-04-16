# Contribution 5: Internalizing Interaction Scaling via GRPO

## Motivation
The proposer-reviewer architecture (Contributions 1-4) is an external
scaffolding that bolts interaction scaling onto frozen frontier models.
The model itself has no awareness of budgets, no understanding of when
to seek feedback, and no learned preference for interaction over pure
reasoning.

We ask: **Can interaction scaling be internalized as a model capability?**

## Approach

### Two-Stage Training
1. **SFT on successful trajectories**: Teach Qwen3 8B the basic
   think-do-review pattern by training on successful interaction traces
   from our experiments.

2. **GRPO with grounded rewards**: Fine-tune with Group Relative Policy
   Optimization using the environment as the reward signal:
   - Code: test execution pass/fail
   - Visual: VLM quality score
   - Research: factual accuracy

### What the Model Learns
- **Budget awareness**: Behave differently with 3-step vs 10-step budgets
- **Action selection**: When to [EXECUTE] vs [REVIEW] vs [SUBMIT]
- **Early stopping**: High confidence → stop iterating (save budget)
- **Feedback utilization**: Parse execution errors → targeted revisions

## Expected Results

| Condition | Pass Rate (code) |
|-----------|-----------------|
| Qwen3 8B base (single-shot) | ~30-40% |
| Qwen3 8B base + external scaffold | ~50-60% |
| Qwen3 8B GRPO-trained | ~60-70% |
| Claude Sonnet 4 (single-shot) | 67% |
| Claude Sonnet 4 + external scaffold | 100% |

The key comparison: **trained 8B without scaffold approaches or exceeds
untrained 8B with scaffold.** This proves interaction scaling is a
learnable capability.

## Implications
1. Future foundation models should be trained with interaction scaling
   awareness from the start
2. The think-do-review pattern can be a training objective, not just
   a prompting pattern
3. Budget-conditioned behavior is feasible — models can learn to
   allocate compute differently under different constraints
4. Small models + training can partially compensate for capability
   gaps vs frontier models

## Paper Section: "From Scaffolding to Capability"
This becomes Section 7 in the paper, following the experimental results.
It transforms the paper from "a useful framework" into "a principle that
can be learned", which is a stronger research contribution.

## Training Details
- Model: Qwen3 8B
- Method: SFT → GRPO (using trl)
- Data: Interaction traces from all 6 benchmark categories
- Reward: Grounded environment feedback (test execution, VLM, fact-check)
- Hardware: GPU cluster (or Apple Silicon for smaller-scale experiments)
