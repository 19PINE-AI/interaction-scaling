# Finding 06: Proposer-Reviewer Achieves 100% on HumanEval+

## Result
With correct evaluation (prompt prefix included):

| Approach | Pass@1 | Avg Tokens | Failed |
|----------|--------|------------|--------|
| B1: Single-shot | 98.17% (161/164) | 439 | /83, /130, /132 |
| B5: Agentic loop | 99.39% (163/164) | 506 | /132 |
| **Ours: PR-Fixed** | **100.0% (164/164)** | **549** | **none** |

## Significance
1. The proposer-reviewer with execution feedback achieves **perfect accuracy**
   on all 164 HumanEval+ problems using Claude Sonnet 4.
2. It solves HumanEval/132 (is_nested) which the agentic loop cannot solve
   even after 5 iterations.
3. The total token overhead vs B1 is only 1.25x (549 vs 439 avg tokens).
4. The total token overhead vs B5 is only 1.09x (549 vs 506 avg tokens).

## The Value of Architectural Separation
The only problem that differentiates B5 from Ours is HumanEval/132:
- B5 tries 5 iterations with raw execution feedback but keeps generating
  wrong solutions for the nesting criterion
- Ours: the reviewer provides structured analysis of WHY the tests fail,
  identifies the specific incorrect assumption, and suggests a different
  algorithmic approach

This directly validates the paper's core thesis: architectural separation
(separate proposer and reviewer with structured feedback) provides genuine
value beyond just having execution feedback.

## Token Efficiency Summary
| Transition | Pass@1 Gain | Token Overhead |
|-----------|-------------|----------------|
| B1 → B5 | +1.22pp | 1.15x |
| B1 → Ours | +1.83pp | 1.25x |
| B5 → Ours | +0.61pp | 1.09x |

## Implications
- On benchmarks where models are already strong (98%+), interaction scaling
  provides the final push to near-perfect accuracy
- The proposer-reviewer overhead is modest (~25% more tokens)
- The architectural value is concentrated in the hardest problems
- For a paper, this result should be presented alongside harder benchmarks
  where the gap is larger
