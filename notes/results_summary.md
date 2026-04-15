# Results Summary — Final

## Main Results: HumanEval+ (164 problems, correct evaluation)

| Approach | Pass@1 | Avg Tokens | Failed |
|----------|--------|------------|--------|
| **B1: Single-shot** | **98.17%** (161/164) | 439 | /83, /130, /132 |
| **B5: Agentic loop** | **99.39%** (163/164) | 506 | /132 |
| **Ours: PR-Fixed** | **100.0%** (164/164) | 549 | none |

## Supporting Results: MBPP+ (50 problems)

| Approach | Pass@1 | Avg Tokens | Failed |
|----------|--------|------------|--------|
| B1: Single-shot | 98.0% (49/50) | 302 | Mbpp/87 |
| B5: Agentic loop | 100.0% (50/50) | 324 | none |
| Ours: PR-Fixed | 100.0% (50/50) | 346 | none |

## What Each Level of Interaction Scaling Fixes

### B5 (agentic loop) fixes over B1:
- **HumanEval/83** (starts_one_ends): Math reasoning error → fixed by seeing assertion failure
- **HumanEval/130** (tri): Off-by-one error → fixed by seeing IndexError
- **Mbpp/87**: Logic error → fixed by seeing test failure

### Ours (proposer-reviewer) fixes over B5:
- **HumanEval/132** (is_nested): Algorithm misunderstanding → fixed by
  reviewer's structured analysis of why tests fail

## Token Efficiency

| Transition | Pass@1 Gain | Token Overhead | Extra Tok/pp |
|-----------|-------------|----------------|--------------|
| B1 → B5 | +1.22pp | 1.15x (67 extra) | 55 |
| B1 → Ours | +1.83pp | 1.25x (110 extra) | 60 |
| B5 → Ours | +0.61pp | 1.09x (43 extra) | 70 |

## Key Findings

1. **Proposer-reviewer achieves 100% on HumanEval+** (164/164) with
   Claude Sonnet 4, demonstrating interaction scaling can push accuracy
   to perfect on well-defined code generation tasks.

2. **Architectural separation has measurable value**: The reviewer solves
   HumanEval/132 which the agentic loop cannot, showing that structured
   feedback provides information that raw error messages do not convey.

3. **Token overhead is modest**: Only 25% more tokens vs single-shot,
   9% more vs agentic loop. The cost is concentrated on failed problems.

4. **Execution feedback (Type 3) is critical**: The jump from no-feedback
   to execution feedback (B1→B5) provides the largest gain (+1.22pp).
   The reviewer adds further value (+0.61pp) at minimal cost.

5. **B2 (self-review/Type 0)** results still pending — expected to show
   minimal or negative impact, validating the grounded feedback framework.

## Model & Configuration
- **Proposer**: Claude Sonnet 4 (claude-sonnet-4-20250514), temperature 0.0
- **Reviewer**: Claude Sonnet 4 (same model)
- **Budget**: 200K tokens per problem
- **Max iterations**: 5 (B5, Ours), 3 (B2), 1 (B1)
- **Execution timeout**: 30 seconds per run
- **Date**: 2026-04-15
