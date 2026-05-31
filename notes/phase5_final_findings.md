# Phase 5: Multimodal Self-Review Distillation — Findings

**Setup:** Teacher Qwen3-VL-235B-Thinking (OpenRouter) → student Qwen3-VL-8B-Thinking (QLoRA r=16, 37 filtered SFT examples). Gemini 3 Flash vision judge. Combined gen+review system prompts for code/webpages/slides.

**Teacher quality (baseline for distillation):**
- 50 traces collected, 37 judge-kept (74%)
- 100% of critiques specific (cite concrete observations/exception/element)
- 98% of revisions address the preceding critique
- **0% identical retries** (0/300)
- Failure mode on rejected traces: `final_meets_spec` — good process, but 5-turn budget insufficient for complex visual tasks

**Base student (no SFT) is non-functional in agent loop:** 15-17K chars of rambling design prose per turn, no fenced code blocks, `status=no_artifact` 100% on smoke set.

**V1 SFT (teacher `<think>` stripped from labels):**
- judge-kept: 1/18 (6%)
- review_specific: 89%
- revision_addresses_review: 22%
- **identical-retry rate: 67%** (33/49 retries)
- Clean format + specific critiques, but collapses to identical re-emission after turn 1 on hard tasks. Same pathology as Phase 2 text student.

**V2 SFT (teacher `<think>[:3000]` retained in labels):**
- judge-kept: **3/18 (17%)** (3x v1)
- revision_addresses_review: **44%** (2x v1)
- **identical-retry rate: 32%** (halved vs v1)
- New failure mode: 8/18 traces degenerate into thinking-block repetition loops (base64 gibberish, etc.) and never close `</think>`.

**V2 + repetition_penalty=1.15 attempt:** didn't help — thinking runaway is structural, not repetition-driven. Model emits 55K chars of exploratory analytical prose without closing `</think>`.

**V3 (REASONING_CAP=1500, half of v2):** blocked by GPU contention (other users' jobs).

## Key finding

**The information that transfers substance in distillation is the teacher's PLANNING, not just the critique format.** V1 taught format + critique surface ("what the teacher said"). V2 taught the plan-before-emit pattern ("how the teacher thought"). The second is what closes the identical-retry pathology — which was the central failure mode of Phase 2.

## Remaining issues

- **Thinking runaway:** v2 model emits too-long thinking. Fixes: shorter SFT reasoning cap (v3, pending), hard-stop criteria on `</think>` at inference, or continue-train on examples with short decisive reasoning.
- **Content repetition:** some traces emit many long inline SVG base64 images. Fixes: stronger system prompt ban, higher repetition penalty on literal token bigrams.

## Memory-anchored numbers (for future sessions)

| | Phase 2 (8B text) | Phase 5 V1 (no `<think>`) | Phase 5 V2 (`<think>` retained) | Teacher |
|---|---|---|---|---|
| identical-retry % | 56% | 67% | **32%** | 0% |
| judge-keep % | — | 6% | **17%** | 74% |
| revision addresses review % | — | 22% | **44%** | 98% |

## Next moves (ranked)

1. **V3 when GPU free:** reasoning cap 1500 + rep_penalty 1.1 at inference
2. **Thinking-budget hard stop** at inference: force `</think>` token after N new tokens
3. **GRPO on v2**: reward = `judge_keep - 0.3 * I[no_artifact] - 0.2 * I[identical_retry]`
4. **Scale trace collection** to 150+ tasks (current 37 SFT examples is tiny)
