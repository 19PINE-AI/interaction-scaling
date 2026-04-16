# Finding 09: Hard Code Benchmark — 60% → 93% with Execution Feedback

## Result
On 15 hard coding tasks with subtle bugs:

| Approach | Pass Rate | Avg Tokens |
|----------|-----------|------------|
| Single-shot | 60% (9/15) | ~500 |
| With execution feedback (5 iters) | 93% (14/15) | ~1,500 |

**+33 percentage points improvement from execution feedback!**

## Per-Task Breakdown

| Task | Single-shot | Reviewed | Bug Type |
|------|------------|----------|----------|
| code_001 (binary search insert) | FAIL | FIXED (2 iters) | Off-by-one |
| code_002 (sliding window max) | FAIL | FIXED (2 iters) | Edge case |
| code_003 (spiral traversal) | PASS | PASS | — |
| code_004 (interval merge) | PASS | PASS | — |
| code_005 (next permutation) | PASS | PASS | — |
| code_006 (vector angle) | FAIL | FIXED (3 iters) | Numerical precision |
| code_007 (collinear check) | PASS | PASS | — |
| code_008 (fibonacci mod) | FAIL | FAIL (5 iters) | Algorithm fundamentals |
| code_009 (LRU cache) | PASS | PASS | — |
| code_010 (tree serialize) | PASS | PASS | — |
| code_011 (circular buffer) | PASS | PASS | — |
| code_012 (topological sort) | PASS | PASS | — |
| code_013 (expression eval) | FAIL | FIXED (2 iters) | Unary minus parsing |
| code_014 (IP addresses) | FAIL | FIXED (2 iters) | Leading zeros |
| code_015 (word break) | PASS | PASS | — |

## Key Insights

1. **Single-shot fails 40% of the time** on these carefully designed tasks
2. **Execution feedback fixes 5/6 failures** — the model CAN solve these
   problems, it just needs to see the failing test case to correct its logic
3. **Only code_008 is fundamentally unsolvable** — matrix exponentiation
   for large Fibonacci is beyond what the feedback loop can guide to
4. **Most fixes need only 2 iterations** — first attempt fails, model sees
   the error, second attempt succeeds
5. **This is exactly the pattern the paper predicts**: grounded feedback
   (Type 3) catches bugs that are invisible to self-review but obvious
   from execution output
