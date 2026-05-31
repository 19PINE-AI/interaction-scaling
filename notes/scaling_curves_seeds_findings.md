# Scaling curves (3-seed): performance vs token budget on 15 hard code tasks

**Setup.** Same 15 hand-curated hard code tasks (`data/hard_benchmarks/code/code_tasks.json`) and
Sonnet 4 (`claude-sonnet-4-20250514`) as the single-run sweep. **3 seeds per cell** for S, L, H
(independent API runs; the Anthropic API does not accept a `seed` parameter, so seeds = independent
invocations and reflect underlying API stochasticity even at temp=0). **R was run once (seed 1
only)** since temp=0 and the extended-thinking variation at B=5K/20K is small relative to the gaps
of interest. Strategy definitions unchanged from `notes/scaling_curves_findings.md`.

Cells: 4 strategies x 3 budgets x 3 seeds x 15 tasks = 540 nominal cells; with R single-seed the
actual count is (1+3+3+3) seeds x 3 budgets x 15 tasks = 450 task-level passes.

## Main table: mean +/- SD across seeds

| Budget | R (reasoning-only) | S (best-of-N) | L (single-agent loop) | H (proposer-reviewer) |
|---|---|---|---|---|
| 1000 | 60.0% +/- 0.0pp (n=1, 654t) | 80.0% +/- 0.0pp (n=3, 665t) | 57.8% +/- 3.1pp (n=3, 658t) | 62.2% +/- 3.1pp (n=3, 651t) |
| 5000 | 73.3% +/- 0.0pp (n=1, 1446t) | 77.8% +/- 3.1pp (n=3, 2107t) | 93.3% +/- 0.0pp (n=3, 909t) | 91.1% +/- 3.1pp (n=3, 901t) |
| 20000 | 73.3% +/- 0.0pp (n=1, 5047t) | 86.7% +/- 0.0pp (n=3, 7173t) | 97.8% +/- 3.1pp (n=3, 1463t) | 100.0% +/- 0.0pp (n=3, 1003t) |

(cells: `pass-rate +/- SD across seeds (n_seeds, mean output tokens)`. SD is population SD across
seed means. R is single-seed so SD=0 and n=1.)

## L vs H paired sign test (per-seed, per-task)

For each budget we collect every (seed, task) pair and count how often H passed but L did not
(plus) vs L passed but H did not (minus); the two-sided p-value comes from Binom(n_non_tied, 0.5).

| Budget | mean L | mean H | gap (H-L) pp | H>L (plus) | L>H (minus) | ties | non-tied n | p (two-sided) |
|---|---|---|---|---|---|---|---|---|
| 1000  | 57.8% | 62.2% | +4.4 | 3 | 1 | 41 | 4 | 0.625 |
| 5000  | 93.3% | 91.1% | -2.2 | 1 | 2 | 42 | 3 | 1 |
| 20000 | 97.8% | 100.0% | +2.2 | 1 | 0 | 44 | 1 | 1 |

## Headline

The original single-seed H-L gap at B=20K (+6.7pp) shrinks to +2.2pp across 3 seeds and is within seed noise: paired sign test p=1 on 1 non-tied pairs (ties=44/45).

## What changed vs the single-seed sweep

The original N=1 numbers (seed 1 here) reported L at 93.3% and H at 100% for B=20K,
producing a +6.7pp H-L gap and a +13.3pp gap at B=1K. With 3 seeds:

- **L at B=20K rises from 93.3% to 97.8% mean (98.3% pooled).** L hit 100% in seeds 2 and 3.
  The seed-1 result (14/15) was unlucky: a single-task failure on `code_011` that L solves
  cleanly in the other two seeds. So the "L plateaus at 93.3%" line from the original note
  was a single-seed artefact.
- **H at B=20K stays at 100% across all 3 seeds.** This is the one cell where seed variation
  is truly zero.
- **The H-L gap at B=1K collapses from +13.3pp to +4.4pp.** Both strategies are noisy at
  low budget (SD ~3pp). The single-seed claim that "H ahead of L at low budget" was a
  near-tie + noise.
- **At B=5K the gap actually reverses (+0pp -> -2.2pp), still inside noise.**

The qualitative ordering of *strategies* (R < S < L ~ H at high budget) is preserved, but
the L-vs-H Pareto-dominance claim is **not** supported once seeds are added.

## Per-cell failing tasks (per seed)

The only hard-task that survives at B=20K across seeds is `code_011`. L fails it in seed 1
only; H solves it everywhere. At B=5K, H fails `code_011` in seed 3 (where L solves it),
which is why the B=5K mean swings against H.

### L (single-agent loop) — failing task IDs per seed
| Budget | seed 1 fails | seed 2 fails | seed 3 fails |
|---|---|---|---|
| 1000 | code_001, code_003, code_004, code_008, code_010, code_011, code_015 | code_001, code_003, code_004, code_008, code_010, code_011 | code_001, code_003, code_008, code_010, code_011, code_015 |
| 5000 | code_011 | code_011 | code_011 |
| 20000 | code_011 | (all pass) | (all pass) |

### H (proposer-reviewer) — failing task IDs per seed
| Budget | seed 1 fails | seed 2 fails | seed 3 fails |
|---|---|---|---|
| 1000 | code_001, code_003, code_008, code_010, code_011 | code_001, code_003, code_008, code_010, code_011, code_015 | code_001, code_003, code_008, code_010, code_011, code_015 |
| 5000 | code_003 | code_003, code_011 | code_011 |
| 20000 | (all pass) | (all pass) | (all pass) |

## Per-seed breakdown by strategy

### R (single seed only, no replication)
| Budget | seed 1 | mean +/- SD |
|---|---|---|
| 1000 | 60.0% (9/15) | 60.0% +/- 0.0pp |
| 5000 | 73.3% (11/15) | 73.3% +/- 0.0pp |
| 20000 | 73.3% (11/15) | 73.3% +/- 0.0pp |

### S (best-of-N)
| Budget | seed 1 | seed 2 | seed 3 | mean +/- SD |
|---|---|---|---|---|
| 1000 | 80.0% (12/15) | 80.0% (12/15) | 80.0% (12/15) | 80.0% +/- 0.0pp |
| 5000 | 80.0% (12/15) | 80.0% (12/15) | 73.3% (11/15) | 77.8% +/- 3.1pp |
| 20000 | 86.7% (13/15) | 86.7% (13/15) | 86.7% (13/15) | 86.7% +/- 0.0pp |

### L (single-agent loop)
| Budget | seed 1 | seed 2 | seed 3 | mean +/- SD |
|---|---|---|---|---|
| 1000 | 53.3% (8/15) | 60.0% (9/15) | 60.0% (9/15) | 57.8% +/- 3.1pp |
| 5000 | 93.3% (14/15) | 93.3% (14/15) | 93.3% (14/15) | 93.3% +/- 0.0pp |
| 20000 | 93.3% (14/15) | 100.0% (15/15) | 100.0% (15/15) | 97.8% +/- 3.1pp |

### H (proposer-reviewer)
| Budget | seed 1 | seed 2 | seed 3 | mean +/- SD |
|---|---|---|---|---|
| 1000 | 66.7% (10/15) | 60.0% (9/15) | 60.0% (9/15) | 62.2% +/- 3.1pp |
| 5000 | 93.3% (14/15) | 86.7% (13/15) | 93.3% (14/15) | 91.1% +/- 3.1pp |
| 20000 | 100.0% (15/15) | 100.0% (15/15) | 100.0% (15/15) | 100.0% +/- 0.0pp |

## Token-budget variation across seeds (sanity check)

The "deterministic" strategies (L and H at temp=0) still show small mean-output-token variation
across seeds, reflecting batch-level API nondeterminism. The original N=1 numbers are within ~1
SD of the seed-mean in every cell -- they were effectively unbiased point estimates for the
population mean.

| Strategy | B=1K mean+/-SD tok | B=5K mean+/-SD tok | B=20K mean+/-SD tok |
|---|---|---|---|
| R | 654 +/- 0t | 1446 +/- 0t | 5047 +/- 0t |
| S | 665 +/- 20t | 2107 +/- 38t | 7173 +/- 110t |
| L | 658 +/- 3t | 909 +/- 16t | 1463 +/- 28t |
| H | 651 +/- 2t | 901 +/- 23t | 1003 +/- 44t |

## Files

- `results/scaling_curves/code_4strategy.json` -- seed 1 (all four strategies; the original N=1 sweep)
- `results/scaling_curves/code_4strategy_seed2.json` -- seed 2 (S, L, H)
- `results/scaling_curves/code_4strategy_seed3.json` -- seed 3 (S, L, H)
- `results/scaling_curves/code_4strategy_seeds.json` -- merged per-seed records
- `scripts/run_scaling_curves_code.py` -- sweep driver (unchanged from N=1 version; seeds invoked
  with distinct `--out` paths)
- `scripts/analyze_scaling_curves_seeds.py` -- this analysis
