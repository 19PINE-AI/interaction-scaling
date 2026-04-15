# Preliminary Research Findings: Interaction Scaling

## Summary of Experiments Conducted

### Benchmark: HumanEval+ (164 problems)
- **Model**: Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Budget**: 200K tokens per problem
- **Date**: 2026-04-15

### Initial Results (First 50 Problems)

| Condition | Pass@1 | Avg Tokens | Avg Iterations | Avg Time |
|-----------|--------|------------|----------------|----------|
| B1: Single-shot | 94.0% | 368 | 1.0 | 2.9s |
| B5: Agentic loop (Type 3a) | 98.0% | 544 | 1.1 | 4.1s |
| Ours: P-R fixed (Type 3a) | 98.0% | 804 | 1.2 | 5.4s |

### Key Observations

#### 1. Execution Feedback Fixes Specific Problems
Problems fixed by interaction scaling (B5 and Ours):
- **HumanEval/1**: Fixed on iteration 2. Initial generation had a logic error
  caught by test execution.
- **HumanEval/32**: Fixed on iteration 3. Required multiple revision cycles.

Problem NOT fixed by interaction scaling:
- **HumanEval/38**: Failed after 5 iterations for both B5 and Ours. This
  represents a problem where the model fundamentally misunderstands the
  specification — interaction scaling cannot help when the model lacks the
  capability to solve the problem even with feedback.

#### 2. Type 2 (Static Analysis) Causes Unnecessary Churn
- Type 2 feedback (pylint) on 10 easy problems: pass@1=100%, avg_tokens=7,226
- Compared to B1: same accuracy, 16x more tokens
- Lint warnings trigger unnecessary revisions even when code is correct
- Key insight: Type 2 has no clear stopping criterion (lint always finds issues)

#### 3. Token Efficiency
- B5 uses 1.5x tokens vs B1 for +4pp pass@1 (44 extra tokens per 1pp gain)
- Ours uses 2.2x tokens vs B1 for +4pp pass@1 (109 extra tokens per 1pp gain)
- Proposer-reviewer adds overhead from the separate reviewer LLM call
- On easy problems, the reviewer overhead is not justified

#### 4. Proposer-Reviewer vs Single-Agent Loop
On the first 50 (easier) problems, B5 and Ours achieve identical pass@1.
The additional reviewer in our approach adds ~50% more tokens without
additional accuracy. This suggests the proposer-reviewer architecture
provides value primarily on harder problems where:
- The proposer needs structured guidance (not just raw error messages)
- Context management becomes critical
- Multiple feedback types must be synthesized

## Predictions for Full Results
Based on the research plan's hypotheses:
1. The gap between B5 and Ours should widen on harder problems (later HumanEval+
   problems and MBPP+)
2. Type 3 (execution) >> Type 2 (static) > Type 1 (cross-model) ≈ Type 0 (self)
3. The optimal allocation is task-dependent

## Next Steps
- [ ] Full 164-problem HumanEval+ results (running)
- [ ] MBPP+ 50-problem results (running)
- [ ] B2 self-review results (running)
- [ ] B3 cross-model review
- [ ] Feedback type ablation (Exp 1)
- [ ] Scaling curves at different budgets (Exp 2)
- [ ] Verification gap analysis (Exp 5)
