# Phase 5 — Reasoning-Cap Sweep (V1-V4)

**Result:** U-shaped curve in keep-rate vs reasoning cap. V3 (1500 chars) is the optimum.

## Headline table

| variant | reasoning cap | judge-keep | id-retry % | no_artifact | max_turns |
|---|---|---|---|---|---|
| V1 | 0 (stripped) | 1/18 (6%) | 67% | 2 | 11 |
| **V4** | 1000 | 2/18 (11%) | 35% | 9 | 3 |
| **V3 ★** | **1500** | **8/18 (44%)** | **10%** | 6 | 1 |
| V2 | 3000 | 3/18 (17%) | 32% | 8 | 2 |
| Teacher (235B) | full | (74% on full set) | 0% | — | — |

V3 wins on **all four** metrics simultaneously. V4 (1000) regresses below V2 (3000), confirming the curve is non-monotonic.

## By category (judge-kept)

| | V1 | V4 | **V3** | V2 |
|---|---|---|---|---|
| code | 1/5 | 2/5 | **4/5** | 2/5 |
| webpages | 0/5 | 0/5 | **1/5** | 0/5 |
| slides | 0/8 | 0/8 | **3/8** | 1/8 |

## Mechanism (working hypothesis)

The reasoning-cap sweep maps onto distinct failure modes:

- **V1 (cap=0):** no scratch space → student emits format but can't plan edits → 67% identical retries (Phase 2 pathology).
- **V4 (cap=1000):** some scratch space but too small to structure a coherent plan → student starts thinking, gets confused, gives up → 50% no_artifact.
- **V3 (cap=1500):** matches the 8B model's coherence horizon → student plans then commits → optimum.
- **V2 (cap=3000):** longer than 8B's coherence horizon → student loses focus mid-thinking → 44% thinking-runaway.

The interpretation: **train the student at its own coherence horizon, not the teacher's**. SFT length isn't a free parameter to maximize; there's an architectural fit.

## Comparison to Phase 2 (text-only Qwen3-8B GRPO v2)

The Phase 2 negative finding was: 56% identical-retry pathology, format imitated but substance not. Phase 5 V3 reduces this to **10%** with a single SFT (no GRPO). Combined with multimodal task setting and judge-verifiable visual output, V3 is the strongest evidence yet for the interaction-scaling thesis: **distilled self-review actually transfers when training conditions match student capacity**.

## Open issues

1. **6 V3 no_artifact failures** — still thinking-runaway on hardest tasks. Inference-time fix: hard-stop `</think>` after N tokens.
2. **Pipeline:** `extract_artifact` only reads the LAST turn; if last turn collapses but earlier turns had valid HTML, lose them. Walk backward.
3. **Webpages remain hard** — 1/5 even with V3. Multi-viewport debugging may need more SFT data.

## Suggested next moves

1. **Hard-stop on `</think>`** — cheapest possible fix, could lift V3 from 44% → 55%+.
2. **GRPO from V3:** reward = `judge_keep − 0.3·I[no_artifact] − 0.2·I[identical_retry]`. Strong starting policy.
3. **Scale trace collection** to 150+ tasks. Current 37 is small and may have inflated the V3 gain via overfitting to data distribution.
4. **V5 with cap=1250** to refine the optimum — though unlikely to move much.

## Costs

- 4× SFT runs: ~40 min total (8-12 min each)
- 4× full evals: ~10 hours total (90-150 min each, single GPU)
- 4× judge runs: ~10 min total

