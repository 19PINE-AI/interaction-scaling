# Finding 01: Static Analysis (Type 2) Causes Unnecessary Revision Churn

## Observation
When using Type 2 feedback (pylint static analysis) as the sole grounding signal
in a proposer-reviewer loop, the system performs all 5 iterations even when the
code is already functionally correct. This is because pylint always finds
stylistic issues (naming conventions, missing docstrings, etc.) that trigger
the reviewer to suggest revisions.

## Data
- Type 2 alone on first 10 HumanEval+ problems: pass@1=1.0, avg_tokens=7,225
- B1 single-shot on first 10: pass@1=1.0, avg_tokens=~370
- B5 agentic loop (Type 3a): pass@1=1.0, avg_tokens=~450

Type 2 uses ~16x more tokens than B1 for the same result on easy problems.

## Interpretation
Static analysis feedback is **always non-empty** (pylint always finds something
to complain about), so the system never stops early. Unlike execution feedback
(which returns "all tests pass" -> stop), static analysis has no clear
"success" signal.

This supports the paper's claim that Type 2 feedback is less valuable than
Type 3 feedback for code generation, and specifically explains *why*:
- Type 3 has a clear binary success signal (tests pass/fail)
- Type 2 has no clear stopping criterion (lint is always noisy)

## Implication for the paper
- Need to add early stopping based on execution pass, not just lint pass
- Type 2 is most useful *in combination* with Type 3 (catches complementary
  issues without driving unnecessary revision)
- The token efficiency comparison is important: more feedback ≠ better feedback
