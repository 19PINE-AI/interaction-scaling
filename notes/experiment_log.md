# Experiment Log

## 2026-04-15: Initial Experiments

### Setup
- Framework: Python + Anthropic API + evalplus
- Benchmark: HumanEval+ (164 problems), running first 50
- Proposer model: Claude Sonnet 4
- Reviewer model: Claude Sonnet 4
- Budget: 200K tokens per problem
- Max iterations: varies by condition

### Baselines Running
1. **B1 (single-shot)**: No feedback, single generation
2. **B2 (self-review)**: Type 0 feedback, 3 iterations max
3. **B5 (agentic loop)**: Type 3a execution feedback, 5 iterations max
4. **Ours (P-R fixed)**: Proposer-reviewer with Type 3a, fixed allocation, 5 iterations max

### Sanity Check Results (3 problems)
- B1: pass@1 = 0.667, avg_tokens = 372
- B5: pass@1 = 1.000, avg_tokens = 683
- Ours: pass@1 = 1.000, avg_tokens = 1038

Key observation: Even on 3 problems, execution feedback (B5, Ours) fixes
HumanEval/1 which B1 fails on. The proposer-reviewer uses more tokens but
both interaction approaches achieve perfect accuracy on this tiny sample.

### Results (50 problems)
*Pending — experiments running*
