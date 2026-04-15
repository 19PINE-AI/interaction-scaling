# Finding 04: Quantifying Interaction Scaling Value

## Core Result
On HumanEval+ (164 problems, pre-prompt-fix evaluation):

| Approach | Pass@1 | Failures | Avg Tokens |
|----------|--------|----------|------------|
| B1: Single-shot | 95.73% | 7 (4 eval + 3 real) | 439 |
| B5: Agentic loop | 97.56% | 4 (3 eval + 1 real) | 590 |

B5 fixes 3 genuine failures (HumanEval/1, /83, /130) through execution
feedback loops. It fails on HumanEval/132 despite 5 attempts.

## What Execution Feedback Fixes

### HumanEval/1 (separate_paren_groups)
- B1 fails with NameError (List not imported)
- B5 iteration 2: sees the error, adds import, passes
- Root cause: model generates correct logic but forgets import

### HumanEval/83 (starts_one_ends)
- B1 fails with AssertionError (wrong counting logic)
- B5: sees test failure, corrects the counting formula
- Root cause: mathematical reasoning error corrected by feedback

### HumanEval/130 (tri - tribonacci)
- B1 fails with IndexError (off-by-one)
- B5: sees runtime error, fixes boundary condition
- Root cause: edge case in sequence generation

### HumanEval/132 (is_nested) - NOT FIXED
- Both B1 and B5 fail with AssertionError
- B5 uses all 5 iterations, generating different wrong solutions
- The model fundamentally misunderstands the nesting criterion
- This represents the ceiling of interaction scaling: when the model
  lacks the capability to solve the problem, more attempts don't help

## Token Efficiency

| Metric | Value |
|--------|-------|
| B5 token overhead vs B1 | 1.34x (590 vs 439) |
| Pass@1 gain | +1.83pp |
| Extra tokens per 1pp gain | 82 tokens |
| Problems fixed | 3 (HumanEval/1, /83, /130) |
| Extra tokens per fix | ~2,500 tokens per problem fixed |

## MBPP+ Comparison (50 problems)
B5 achieves 100% vs B1's 98% on MBPP+, fixing Mbpp/87 with only 1.1x
token overhead. The first 50 MBPP+ problems are easier, showing that
interaction scaling has minimal overhead on easy problems.

## Implications
1. Execution feedback provides genuine value for ~2-5% of problems
2. The overhead is modest (~30% more tokens on average)
3. The value concentrates on problems where the model makes fixable errors
4. Interaction scaling cannot fix problems beyond the model's capability
5. Proposer-reviewer adds overhead without additional pass@1 on easy problems
