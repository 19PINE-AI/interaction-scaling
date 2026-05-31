# Phase 5 V3 — Breakthrough on Self-Review Distillation

**TL;DR:** Capping teacher `<think>` at 1500 chars during SFT (vs 3000 in V2 vs 0 in V1) produced the best multimodal student: 44% judge-keep on 18 held-out tasks, 10% identical-retry rate.

## Headline numbers

| metric | V1 (no `<think>`) | V2 (3K reasoning) | **V3 (1500 reasoning)** | Teacher (235B) |
|---|---|---|---|---|
| judge-keep | 6% (1/18) | 17% (3/18) | **44%** (8/18) | 74% |
| revision_addresses_review | 22% | 44% | **72%** | 98% |
| final_meets_spec | 11% | 17% | **44%** | — |
| identical-retry rate | 67% | 32% | **10%** | 0% |
| no_artifact failures | 2 | 8 | 6 | — |
| max_turns failures | 11 | 2 | **1** | — |
| final declared | 5 | 8 | **11** | — |

## By category (judge-kept)

| | V1 | V2 | V3 |
|---|---|---|---|
| code | 1/5 | 2/5 | **4/5** |
| webpages | 0/5 | 0/5 | **1/5** |
| slides | 0/8 | 1/8 | **3/8** |

## Why V3 wins

**V1 (no reasoning):** student emits compact critique + code, but on hard tasks it can't synthesize new edits — 67% identical retries, same Phase 2 pathology.

**V2 (3K-char reasoning):** student plans extensively, produces substantive edits when thinking succeeds — but 8/18 traces degenerate into thinking-loops (base64 spam, runaway analysis) and never close `</think>`. Net: better than V1 on substance, worse on completion.

**V3 (1500-char reasoning):** student thinks just enough to plan an edit, then stops decisively. Identical-retry drops to 10% (within shooting distance of teacher's 0%) AND completion rate stays high (only 6 no_artifact, max_turns down to 1/18). Best of both worlds.

## Mechanism

The hypothesis: thinking-block length in SFT teaches *deliberation budget*. V1 has 0 budget → can't plan, can only mimic format. V2 has 3K budget → student matches the verbose teacher pattern but the 8B model can't sustain coherent reasoning at 3K char length, so it loops. V3 has 1500 budget → matches the model's natural coherence horizon, planning succeeds, edits land.

This explains why **shorter reasoning beats longer reasoning** for distillation — counterintuitive but consistent: train at the student's competence horizon, not the teacher's.

## Open issues

1. **6 no_artifact failures in V3** — still thinking-runaway on hardest tasks. Hard-stop on `</think>` after N inference tokens would fix.
2. **`extract_artifact` only reads the LAST turn** — when the final turn collapses but earlier turns had valid HTML, the trace gets `no_artifact` status. Pipeline bug; should walk turns backward.
3. **Webpages still hard** — only 1/5 kept. Multi-viewport visual debugging may need more SFT data than slides/code.

## Next levers

1. **V4 with REASONING_CAP=1000** — test if even shorter reasoning continues the trend or starts to hurt.
2. **GRPO from V3** — reward = `judge_keep − 0.3·I[no_artifact] − 0.2·I[identical_retry]`. Now we have a strong starting policy.
3. **Scale trace collection** — current 37 SFT examples; bump to 150+ across more diverse tasks.
4. **Inference-time fix:** force `</think>` token after N new tokens (cheap, no retraining).

## Costs

- Trace collection: 50 teacher traces, ~3 hours (OpenRouter Qwen3-VL-235B-Thinking)
- Judge filter: ~5 min (Gemini 3 Flash, 50 calls)
- SFT V3: 12 min (37 examples × 5 epochs × LoRA r=16)
- Eval V3: ~150 min (18 traces, ~8 min/trace; thinking is expensive)

