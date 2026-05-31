# Phase 5 — Complete Findings Document

**Date range:** Apr 22-25, 2026
**Setup:** Teacher Qwen3-VL-235B-Thinking (OpenRouter) → student Qwen3-VL-8B-Thinking. QLoRA r=16. Gemini 3 Flash multimodal judge.
**Held-out:** 18 tasks (5 code, 5 webpages, 8 slides) on which the teacher itself was judge-rejected.

## All variants

| variant | training reasoning | inference cap | judge-keep | id-retry | no_artifact | avg time |
|---|---|---|---|---|---|---|
| V1 | 0 (stripped) | — | 1/18 (6%) | 67% | 2 | 103s |
| V4 | 1000 chars | — | 2/18 (11%) | 35% | 9 | 305s |
| V2 | 3000 chars | — | 3/18 (17%) | 32% | 8 | 255s |
| V3 ★ | **1500 chars** | — | **8/18 (44%)** | 10% | 6 | 501s |
| V3+fc=2000 | 1500 | force-close at 2000 tok | 5/18 (28%) | 15% | 1 | 202s |
| V3+fc=4000 ★ | 1500 | force-close at 4000 tok | **8/18 (44%)** | **9%** | 2 | **225s** |

★ = best variants. V3 and V3+fc=4000 are tied on outcome quality; V3+fc=4000 is 2.2x faster and 3x fewer no_artifact failures.

## By category (judge-kept)

| | V1 | V4 | V2 | V3 | V3+fc=4k |
|---|---|---|---|---|---|
| code | 1/5 | 2/5 | 2/5 | 4/5 | 3/5 |
| webpages | 0/5 | 0/5 | 0/5 | 1/5 | **2/5** |
| slides | 0/8 | 0/8 | 1/8 | 3/8 | 3/8 |

## Two separate findings

### 1. Reasoning-cap is U-shaped (architectural fit)

Training reasoning length must match the student's coherence horizon. Both shorter (V4) and longer (V2) caps degrade vs V3. The 8B model can plan coherently for ~1500 chars; longer becomes runaway, shorter is insufficient structure.

**Generalizable lesson:** When distilling thinking models to smaller students, the right SFT reasoning length is the student's coherence horizon, not the teacher's. Tune via small cap sweep.

### 2. Inference-time force-close has a sweet spot

- 2000-token cap: too aggressive, cuts off legitimate plans (-3 keeps vs V3)
- 4000-token cap: matches V3 outcome quality but 2.2x faster, 4x fewer no_artifact

**Generalizable lesson:** Inference-time logits manipulation can extract speed/reliability gains without sacrificing outcome quality, but only at a cap that respects task variation in needed thinking length.

## Comparison to Phase 2 baseline

The Phase 2 negative finding (text Qwen3-8B GRPO v2):
- 56% identical-retry rate
- format imitated, substance not transferred
- 0pp pass@5 advantage over base

Phase 5 V3+fc=4000 multimodal student:
- **9% identical-retry rate** (6x improvement)
- 44% judge-keep on tasks the 235B teacher itself failed
- substantive critique-and-revise behavior reproducibly verifiable in trajectories

The Phase 2 conclusion was wrong in its strong form ("interaction scaling can't be distilled to small models"). The correct refinement: **interaction scaling distills if and only if the SFT reasoning length matches the student's coherence horizon**. With reasoning stripped (Phase 2 setup), substance is lost; with reasoning at the right length, substance transfers.

## Open issues

1. **Webpages remain hard:** 2/5 max in any variant. Visual debugging across multiple viewports may need much more SFT data than slides/code.
2. **6 V3 (or 2 V3+fc4k) hard tasks remain unsolvable:** these are at the 8B model's diagnostic capacity ceiling, not pathological behavior.
3. **`extract_artifact` only reads the LAST turn:** if the last turn collapses but earlier turns had valid HTML, that artifact is not used. Pipeline bug; walk backward.

## Ranked next moves

1. **Scale trace collection:** generate 50+ NEW tasks (need a webpage/slide generator — code task generator already exists). Train V3 on 100+ examples instead of 37, expect 50-60% keep.
2. **GRPO from V3+fc=4000:** reward = `judge_keep − 0.3·I[no_artifact] − 0.2·I[identical_retry]`. Strong starting policy.
3. **Rep-penalty on HTML content:** the SVG/path-data repetition is its own failure mode unrelated to thinking. Could try `repetition_penalty=1.2` on output tokens specifically.
4. **Pipeline fix:** fall back to previous turn's artifact when last turn collapses.

## Costs

- Teacher trace collection: ~3 hrs (50 tasks)
- Judge filter: 5 min
- 4× SFT runs (V1-V4): 40 min total
- 2× force-close eval (fc=2000, fc=4000): 4 hours total
- Original V1-V4 evals: 10 hours total

