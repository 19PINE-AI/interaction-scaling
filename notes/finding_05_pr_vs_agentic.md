# Finding 05: Proposer-Reviewer vs Agentic Loop — Separation Has Value

## Key Result
On full HumanEval+ (164 problems, pre-prompt-fix eval):

| Approach | Pass@1 | Failures | Avg Tokens |
|----------|--------|----------|------------|
| B1: Single-shot | 95.73% (157/164) | 7 | 439 |
| B5: Agentic loop | 97.56% (160/164) | 4 | 590 |
| **Ours: PR-Fixed** | **98.78% (162/164)** | **2** | 718 |

## The Proposer-Reviewer Fixes Problems the Agentic Loop Cannot

### HumanEval/32 (find_zero)
- **B5 (agentic loop)**: Failed after 5 iterations. The agent sees the
  NameError from missing `poly()` but cannot resolve it because it keeps
  trying to define `poly()` itself instead of understanding that the
  function is in the prompt context.
- **Ours (proposer-reviewer)**: The reviewer analyzes the execution error
  structurally, identifies that `poly()` is a helper function, and provides
  specific guidance to the proposer about using it correctly. Fixed in
  iteration 2.

### HumanEval/132 (is_nested)
- **B5 (agentic loop)**: Failed after 5 iterations with AssertionError.
  The agent generates different wrong solutions but keeps making the same
  conceptual error about what "nested" means.
- **Ours (proposer-reviewer)**: The reviewer provides structured analysis
  of the test failures, identifies the specific assertion that fails, and
  suggests a different algorithmic approach. The proposer uses this guidance
  to generate a correct solution.

## Why Separation Helps

1. **Fresh context**: The reviewer starts with clean context containing
   only the latest code and feedback, avoiding context pollution from
   multiple failed attempts.
2. **Structured analysis**: The reviewer produces a JSON with issues,
   suggestions, and confidence — not just a "try again" message.
3. **Separation of concerns**: The proposer focuses on code generation,
   the reviewer focuses on diagnosis. Neither is distracted by the other's
   task.

## Token Efficiency
- B5 → Ours: +1.22pp pass@1, 1.22x tokens
- Extra tokens per 1pp gain: 105 tokens
- The reviewer adds ~128 tokens per problem on average, but this overhead
  is only incurred on problems that need revision

## Implications for the Paper
This directly validates Contribution 3 (architectural separation):
- The proposer-reviewer pattern provides genuine architectural advantage
  beyond just having execution feedback
- The gap is visible even on HumanEval+ where models are strong
- On harder benchmarks, this gap should be larger
