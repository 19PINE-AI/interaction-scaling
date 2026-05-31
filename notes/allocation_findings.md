# Budget allocation sweep on hard code tasks

**Setup.** 15 hand-curated hard code tasks (`data/hard_benchmarks/code/code_tasks.json`, same set as Phase 1 / Section 4). Sonnet 4 proposer-reviewer harness (`MetaController` + `ExecutionFeedback`), temperature 0, max 5 iterations, total cumulative budget B = 10K tokens per task. The execute phase consumes 0 LLM tokens (pytest subprocess), so `b1` and `b3` are realized as per-call `max_tokens` caps on the proposer and reviewer respectively (`max_tokens = b_i * B / max_iter`, floor 256). `b2` is reserved budget — effectively slack for additional iterations. Total cumulative tokens hard-capped at B. Reviewed-loop reference at the 500K cap is 100% pass-rate (Section 4); at B = 10K the cap can bite.

## 9-row allocation table

| Label | b1 (prop) | b2 (exec) | b3 (rev) | Pass-rate | Mean tokens | Mean iters | Prop cap | Rev cap |
|---|---|---|---|---|---|---|---|---|
| A_propose_heavy   | 0.80 | 0.10 | 0.10 | **14/15 (93.3%)** | 3,101 | 1.47 | 1600 | 256 |
| G_prop_dominant   | 0.50 | 0.25 | 0.25 | **14/15 (93.3%)** | 2,697 | 1.40 | 1000 | 500 |
| D_prop_exec       | 0.40 | 0.40 | 0.20 | 12/15 (80.0%) | 4,025 | 1.67 | 800 | 400 |
| E_prop_review     | 0.40 | 0.20 | 0.40 | 12/15 (80.0%) | 4,388 | 1.80 | 800 | 800 |
| I_equal           | 0.33 | 0.34 | 0.33 | 10/15 (66.7%) | 4,845 | 1.87 | 660 | 660 |
| H_review_dominant | 0.25 | 0.25 | 0.50 |  6/15 (40.0%) | 7,485 | 2.60 | 500 | 1000 |
| F_exec_review     | 0.20 | 0.40 | 0.40 |  2/15 (13.3%) | 8,771 | 3.20 | 400 | 800 |
| B_execute_heavy   | 0.10 | 0.80 | 0.10 |  1/15 ( 6.7%) | 9,383 | 3.87 | 256 | 256 |
| C_review_heavy    | 0.10 | 0.10 | 0.80 |  1/15 ( 6.7%) | 9,383 | 3.87 | 256 | 1600 |

Spread: **86.6 pp** between best and worst. Across-allocation pass-rate mean 53.3%, stdev 34.9 pp. The curve is **sharp**, not flat.

## What's going on

1. **The proposer's per-call cap is the dominant axis.** Pass-rate is monotone in `b1` once you sort the table by it. b1=0.80 and b1=0.50 both deliver 93.3%; b1≤0.25 collapses pass-rate below 40%. The reviewer's cap is largely orthogonal: comparing B (b3=0.10, cap=256) vs C (b3=0.80, cap=1600) gives **identical** pass-rate (1/15) and **identical** mean tokens (9,383) — both run the budget down to the cap because the *proposer* (256-token cap in both) cannot emit a complete function and the harness keeps iterating on truncated outputs. The reviewer-heavy allocation is wasted compute when the bottleneck is upstream.

2. **The "saved-tokens" signal: faster runs pass more often.** The two winners (A, G) finish in ~3K tokens and ~1.4 iterations — they succeed on iteration 1 or 2 because the proposer is given enough room to write the whole function the first time. The losers spend ~9K tokens across ~4 iterations and still fail, because each iteration emits a truncated/half-written function and burns budget on increasingly desperate revisions.

3. **Two extreme tasks dominate the noise.** code_006 passes under all 9 allocations (trivial); code_011 fails under all 9 (capability ceiling, consistent with the prior finding that the original "unsolvable" set is at the model's ceiling, not pathological). The other 13 tasks separate cleanly along b1.

4. **B and C are operationally the same point.** Both floor the proposer cap to 256 (b1=0.10 * 10K / 5 = 200, floored). Their identical results are a clean control: the reviewer's cap doesn't move the needle when the proposer is starved.

## Headline

**Allocation matters a lot, and it matters in one direction: give the proposer most of the budget.** The propose-heavy corners (A, G) hit 93.3% (one task off the 100% Section-4 reviewed ceiling at 500K). Review-heavy corners drop to 6.7–40%. The 9-point simplex sweep yields an 86.6 pp spread, dominated almost entirely by the proposer's per-call `max_tokens` cap.

**Implication for budget-aware allocation.** Under a tight per-task budget the controller should preferentially fund the proposer's per-call output cap; the reviewer's per-call cap is a second-order knob and the execute phase is essentially free of LLM cost on code tasks. A practical rule: spend ≥50% of B on the proposer, split the rest between reviewer and headroom for iteration. The PHASE_ADAPTIVE and CONFIDENCE_CONDITIONED strategies that lean heavily on review on round 2+ (≥0.40 review share) are likely leaving pass-rate on the table when the budget is small.

## Caveats

- 15 tasks: with code_006 (always passes) and code_011 (always fails) excluded, the effective signal is on 13 tasks. The ranking would not change but the absolute pass-rate would (13/13 = 100% at A and G).
- The b2 axis is degenerate in this codebase: code's execute phase is a pytest subprocess (0 LLM tokens), so b2 manifests only as reserved budget. A modality where the executor is itself an LLM call (e.g., VLM critique on rendered output, or web search) would put b2 into the actionable simplex; here we are effectively sweeping a (b1, b3) 1-simplex with b2 as slack.
- Total spend on the sweep: ~$5 (well under the $20 cap).

## Artifacts

- Per-cell raw results: `results/allocation_sweep/code_allocation.json`
- Sweep script: `scripts/run_allocation_sweep.py`
- Analysis script: `scripts/analyze_allocation_sweep.py`
- Run log: `results/allocation_sweep/run.log`
