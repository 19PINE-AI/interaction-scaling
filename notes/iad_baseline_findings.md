# IAD baseline vs. our harness on 15 hard code tasks

## What was run

**IAD (Iterative Agent Decoding)** — faithful approximation of Ruan et al. 2025
(arXiv:2504.01931) on the 15 hand-curated hard code tasks from
`data/hard_benchmarks/code/code_tasks.json`. Model: Sonnet 4
(`claude-sonnet-4-20250514`), thinking off.

### Recipe

At each outer iteration we sample **K=3** candidate functions at **T=0.7**,
score every candidate with an oracle verifier R, and carry the best candidate
+ its execution feedback into the next iteration. Up to **N=3** outer
iterations or until a candidate passes all tests. Per-task output budget
**B=20K** (matched to the H curve's B=20K cell).

### Verifier R (oracle)

Per-assertion test scorer (`score_candidate` in
`scripts/run_iad_code.py`): the task's `test_code` is split into top-level
statements via AST; each `assert` is run in its own `try/except` so we
get a real partial-pass count `(n_pass, n_fail, n_error)`. Candidates are
ranked by `(all_pass, n_pass, -n_error, -n_fail)`. This is the "near-optimal
verifier" the paper argues is the limiting factor for IAD; we hold it at the
oracle to give IAD its best possible chance.

### Simplifications vs. the paper

- K fixed at 3 (paper sweeps 2–6); N fixed at 3 (paper uses 3–4).
- T fixed at 0.7 (paper's Sketch2Code uses 0.6; we need diversity at K=3).
- We use the oracle test scorer rather than a reward model. The paper
  emphasizes verifier quality matters; using the oracle removes that
  bottleneck and makes this a strong upper bound on what IAD can do.
- No "best-vs-worst" textual feedback construction; we just feed the best
  candidate's code + its first-failure error message back in. Simpler and
  closer to L's feedback shape, which keeps the comparison clean.
- Per-iter early-stop: as soon as any of the K candidates passes all tests
  in an iteration, we stop sampling that iteration. (Equivalent to the paper
  in the oracle-verifier limit; just saves tokens.)

## Result

15/15 = **100%** pass rate; mean **1,416** output tokens; mean **1.20**
iterations; mean **1.87** candidates per task. Total cost: **$0.39**.

## Three-row comparison at budget B = 20K

| Strategy | Pass rate | Mean tokens | Description |
|:---|---:|---:|:---|
| L (single-agent loop)         | 93.3% (14/15) | 1,431 | One agent's context grows: generate -> exec -> append stderr -> retry. No sampling. |
| **IAD (recipe approx)**       | **100% (15/15)** | **1,416** | Single agent; at each iter sample K=3 at T=0.7, select best by oracle test scorer, carry best forward. |
| H (proposer-reviewer harness) | 100% (15/15) | 1,029 | Separate proposer + reviewer agents; reviewer sees only latest artifact + exec output and emits structured JSON. |

(L and H numbers from `results/scaling_curves/code_4strategy.json`; IAD from
`results/iad_baseline/code_iad.json`.)

## Interpretation

IAD closes the +6.7pp gap between L and H — both IAD and H reach 100%, while
L plateaus at 93.3% by failing `code_011`. The mechanism is exactly what the
IAD paper claims: per-iteration K-sampling diversifies the candidate
distribution so the single hard task that L can't fix sequentially is
reachable by re-rolling at temperature. We can confirm this directly in the
trace: on `code_011`, all three iter-0 candidates partial-passed 10/11; iter-1
cand 1 (a fresh re-roll, not a sequential refinement) finally hit 11/11.

**But IAD does not beat H — it ties H on pass rate and loses to H on
efficiency.** IAD spent **1,416 mean output tokens** vs. H's **1,029**
(+37.6% per task). The extra cost is the wasted candidates from K-sampling:
on 6/15 tasks IAD spent 3+ candidates per iter, vs. H which on the same
budget makes at most one revision call per iter and uses a structured
review. The reviewer's structured critique is *more sample-efficient* than
re-rolling at temperature: H solved `code_011` in one revision (the only
task where it iterates), at lower cost than IAD's K=5 candidate spend across
two iterations.

Two further notes about this IAD setup:

1. **IAD here had an unrealistically strong verifier.** We gave it the
   per-assertion oracle pass count — i.e., during the run we essentially
   already knew which candidate was correct. In any deployment IAD would
   need a learned reward model or LLM judge, which is exactly the
   bottleneck the paper itself names. The 100% pass rate is therefore an
   *upper bound* on IAD with this much compute.

2. **Early-stop dominated.** On 9/15 tasks the very first K=1 candidate
   passed all tests, so K-sampling never fired. The K>1 sampling only
   contributed to the 6 harder tasks (`code_001, code_003, code_008,
   code_011, code_013`, plus iter-0 K=3 on a couple). On those 6 tasks
   alone, IAD averaged 2,839 output tokens — over 2x its global mean and
   nearly 3x H's global mean.

## Verdict

**Yes, H Pareto-dominates IAD too:** same pass rate (100%, 15/15) at
lower token cost (1,029 vs. 1,416, –27%), even when IAD is given an
oracle verifier that masks its real-world weakness. The architectural
separation between proposer and reviewer is more token-efficient than
K-sampling-and-select at the budget where both saturate.

## Files

- `scripts/run_iad_code.py` — IAD driver and per-assertion verifier.
- `results/iad_baseline/code_iad.json` — per-task records (passed,
  tokens, num_iterations, candidates list, scores, code emitted).
- `logs/iad_baseline.log` — full run log.

## Caveats

- Single run per task; for the paper, repeat the K-sampling cells on the
  hard subset (`code_001`, `code_003`, `code_008`, `code_011`, `code_013`)
  at multiple seeds.
- Recipe is faithful to IAD's *core mechanism* (per-iter sampling +
  reward-based selection) but not its full textual best-vs-worst feedback
  construction. The clean L/IAD/H comparison is at the level of "where
  does the verifier signal land": embedded in the prompt history (L), used
  to pick among K samples (IAD), or produced by a second agent in a fresh
  context (H).
- Sonnet 4 only. Whether IAD's K-sampling helps more on smaller / weaker
  models is an open question this run does not address.
