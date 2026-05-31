# Phase 6: Agentic SFT — Negative Result

## TL;DR

Phase 6 rebuilt Phase 5's pipeline with proper tool calling: 3 general tools (`write_file`, `read_file`, `bash`) and the agent itself orchestrating the write→render→read review loop. This addresses two structural critiques of Phase 5:

1. **Trajectory compression flaw**: Phase 1 SOTA scaffold used compressed single-turn calls; Phase 5 SFT used full multi-turn growing trajectories.
2. **No tool use flaw**: Phase 5 traces had ZERO tool calls; the harness did all I/O externally.

**Result: 1/18 (6%) judge-keep on the same hard held-out set vs Phase 5 V3+fc4k's 8/18 (44%).**

The agentic format with current SFT recipe is significantly worse than Phase 5's direct markdown emission. The negative result has clear mechanical causes that we identified.

## Pipeline (built and verified)

| Component | File | What it does |
|---|---|---|
| Sandbox | `src/training/agent_workspace.py` | Per-trajectory tmpdir + 3 tools (write_file/read_file/bash) with path traversal guard and python3 shim. |
| Teacher | `src/training/collect_agentic_traces.py` | Qwen3-VL-235B-Instruct via OpenRouter. Heavy `AGENT_SYSTEM_PROMPT`. Pre-populates `test.py` for code tasks. |
| Judge | `src/training/judge_agentic_traces.py` | Gemini 3 Flash multimodal. Per-step trace summary + final saved screenshot. Code uses deterministic `final_passed`. |
| SFT prep | `src/training/prepare_agentic_sft.py` | Collapses (tool-text + user-image_url) pair into single `tool` role with image content. Replaces heavy agent prompt with stripped student prompt. |
| Training | `src/training/train_vl_sft.py` | Standard LoRA SFT on assistant-only labels. Uses `tools=` kwarg in chat template. |
| Inference | `src/training/run_agent_student.py` | Parses `<tool_call>{json}</tool_call>` XML, dispatches via Workspace. |

## Pipeline checkpoints (passed)

- 94 teacher traces collected (~6.5 hours real time, 3 OpenRouter workers)
- 40 judge-kept (43%): code 13/15, webpages 11/38, slides 16/41
- 35 → 31 SFT examples after 50K-char length filter
- Chat template renders `<tool_call>` XML correctly with `tool` role + image content
- Base Qwen3-VL-8B-Instruct emits valid `<tool_call>` blocks natively (verified)

## Critical bugs found and fixed during Phase 6

1. **`load_messages_with_images` stripped `tool_calls` field** (train_vl_sft.py:49). Training data effectively rendered as text-only assistant turns followed by orphan tool responses. Model was being trained to emit `<|im_end|>` immediately after introductory prose. Fix: preserve tool_calls in the dataset's message reconstruction.
   - Before fix: 16% of tokens unmasked, no `<tool_call>` in labels
   - After fix: 32% of tokens unmasked, full tool_call JSON in labels
   - Loss after fix dropped from 0.55 to 0.12 (predictable JSON tokens)

2. **Robust tool_call parser** for unterminated `<tool_call>...` (no closing tag when generation hits token budget). Uses `json.JSONDecoder.raw_decode()` to find object boundaries.

3. **Qwen3-VL-Thinking auto-injects `<think>\n` into every assistant turn**, fighting our SFT signal (teacher = Instruct, no thinking). Switched base model to **Qwen3-VL-8B-Instruct** which doesn't auto-inject think blocks.

## Held-out eval (18 hard tasks, seed 42)

| | Phase 5 V3+fc4k (8B md) | Phase 6 v3 (8B verb 31ex) | Phase 6 v4 (32B 4-bit) | Phase 6 v6 (30B-A3B 8-bit) | Phase 6 v7 (8B compact 18ex) | **Phase 6 v8 (8B verb 51ex)** |
|---|---|---|---|---|---|---|
| judge-keep total | 8/18 (44%) | 1/18 (6%) | 3/18 (17%) | 3/18 (17%) | 1/18 (6%) | **1/18 (6%)** |
| judge-keep code | n/a | 1/5 | 2/5 | 3/5 | 1/5 | 1/5 |
| judge-keep webpages | n/a | 0/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| judge-keep slides | n/a | 0/8 | 1/8 | 0/8 | 0/8 | 0/8 |
| single-step overflow | n/a | 6/18 | 1/18 | 3/18 | 7/18 | **1/18** |
| status=final emitted | n/a | 1/18 | 3/18 | 3/18 | 1/18 | 1/18 |

**v7 compact teacher** (cap write_file at 5KB, 18 SFT examples after length filter): no improvement over verbose. Webpages still 0/5 — confirms the format floor is NOT teacher-style-driven. Compactness even slightly worsens single-step overflows (7/18 vs 6/18) because compact-teacher's smaller HTML covers fewer styling patterns the held-out tasks demand.

**v8 merged (8B + 51 SFT examples = 64% more data than v3)**: 1/18 (6%) — also no total improvement, but **single-step overflows dropped from 6 → 1** (more diverse training contexts taught the student to stay within budget). Code 1/5, webpages 0/5, slides 0/8. The surviving 17 trajectories all run the full 16-step loop but none satisfy the judge.

## Pass@k inference scaling on v8

Three seeds with greedy (seed 42, T=0) + sampling (seeds 43-44, T=0.7, repetition_penalty=1.0):

| variant | pass@1 | pass@2 | pass@3 |
|---|---|---|---|
| Phase 5 V3+fc4k (8B markdown) | 8/18 (44%) | — | 10/18 (56%) |
| **Phase 6 v8 (8B agentic, 51 ex)** | 1/18 (6%) | 4/18 (22%) | 4/18 (22%) |

- Variance is enormous: **0 task overlap between seed 42 and seed 43** — different seeds catch different tasks
- Pass@k lift is +16pp (vs Phase 5's +12pp) — sampling helps MORE in agentic format, but from a much lower base
- Pass@3 saturates: seed 44 added 0 new keeps
- **Webpages remain 0/5 across pass@3** — sampling cannot break the structural floor

### Final disambiguation

Across 5 Phase 6 configurations (varying model size, precision, teacher style, and SFT data scale), the webpages floor is **0/5 for every single configuration**, while code maxes at 3/5 (30B-A3B 8-bit). This decisively shows:
- **Capacity** modestly helps overall (1→3 keep) and disproportionately helps **code** (1→3).
- **Precision** (8-bit > 4-bit) further helps code (32B 4-bit code 2/5 → 30B-A3B 8-bit code 3/5 with smaller total params).
- **More SFT data** fixes the overflow tax (6 → 1) but does NOT improve judge quality.
- **Compact teacher** regresses on overflow without helping quality.
- **Webpages 0-floor** is structural to either the agentic format itself or to the held-out spec characteristics, **not addressable** by any of: capacity, precision, data scale, or teacher style at this 8B-30B regime.

### 32B disambiguation: capacity is one bottleneck, not the dominant one

| Failure mechanism | 8B v3 | 32B v4 | Capacity-bound? |
|---|---|---|---|
| Single-step overflow (HTML > budget) | 6/18 | 1/18 | **Yes** — 5/6 fixed by capacity |
| Iterates but doesn't converge / quality issues | 11/18 | 13/18 | No — unchanged |
| `gen_error` (KV-cache OOM during inference) | 0/18 | 1/18 | Capacity-induced (32B's KV grows faster) |

Going from 8B→32B (4× capacity) recovered 5/6 overflow tasks, but only **2 of those 5 converted into judge-keep**. The remaining gap from Phase 5's 44% (markdown format) is structural to the agentic format / SFT recipe, not pure capacity.

### Trace status breakdown (Phase 6 v3)
- single-step overflow (1 step): **6/18** — model wrote enormous HTML inside a `<tool_call>` and never emitted `</tool_call>` before token budget exhausted
- mid-iter (2-15 steps): 1/18
- full 16-step max iteration: 11/18 — agent loop structurally works but didn't converge

## Why the regression

### 1. Token budget overflow (6 fatal failures)

Tasks with detailed specs (web_002, web_003, web_005, slide_007, slide_014, code_h005) push the model into emitting 30-90KB of content inside a single `<tool_call>` JSON. With `max_new_tokens=20000` (~ 6800 chars budget for JSON-escaped HTML), the model hits the limit mid-string and never closes the tool call. The base Instruct model has the same failure on these tasks.

This is a **fundamental capacity issue** specific to the agentic format: JSON-escaping inflates HTML by ~40-80%, and tool calls for visual artifacts must contain the full file content. Phase 5's direct markdown emission has no such inflation.

### 2. Verbose teacher style baked into student

The SFT data is from Qwen3-VL-235B emitting verbose, comprehensive HTML. The 8B student tries to reproduce that distribution but lacks compactness skill. With 31 SFT examples over 3 epochs, the model imitates teacher style without internalizing teacher's compactness instincts.

### 3. No `<final>OK</final>` convergence

Even when the agent loop ran 16 steps and the artifact looks reasonable, the model never declared final. The judge marks `revision_addresses_review=False` because critiques don't trigger meaningful targeted revisions in the next turn — the agent oscillates rather than converges.

### 4. Iterative review demands more capacity than emission

Phase 5 V3 had to: emit 1500-token reasoning, emit code/HTML, emit critique. That's it.
Phase 6 has to: emit reasoning, emit `<tool_call>` for write_file (with file content as JSON-escaped string), emit `<tool_call>` for bash (render command), emit `<tool_call>` for read_file (image), interpret returned screenshot, emit critique, emit revised `<tool_call>` for write_file (full HTML again), repeat.

Each cycle has ~3-4× the token cost of Phase 5's approach. With the same model size and SFT budget, Phase 6 is doing more work with less effective learning signal per concept.

## Implications for the paper

This is a clean negative result that strengthens the paper's thesis on the **internalization limit at small scale**:

- **Phase 5 V3+fc4k showed interaction scaling distills to 8B with the right format** (44% single-shot, 56% pass@3) when the format is "emit critique + revised artifact in markdown".
- **Phase 6 shows that fully agentic interaction does NOT distill to 8B with comparable SFT budget** (6% single-shot). The student lacks the capacity to internalize tool dispatch + verbose artifact emission + multi-turn convergence simultaneously.

This suggests the operational answer for deploying interaction scaling at 8B scale is: **continue using external scaffolding rather than tool-using agents**. The interaction loop wins, but the harness wins over the agent.

## Open questions

1. **Would 32B internalize agentic interaction?** With ~4× more capacity, the per-concept signal-to-noise ratio improves. Worth testing if the result is to be claimed for "small models" specifically.
2. **Would a dedicated content-cap on write_file content help?** E.g., teach student to emit small first attempts then progressively edit. Would need different teacher data.
3. **Would DAPO (rejection-sampling-based RL)** boost the 1/18 single-shot to something higher? The model that did pass code_h001 in 3 steps shows the format works when it works — variance might be exploited.

## Files of record

- Trace data: `data/training/agentic_traces.json` (94 teacher), `data/training/agentic_traces_judged.json` (with judge verdicts)
- SFT data: `data/training/agentic_sft.jsonl` (31 examples)
- Adapter: `models/qwen3-vl-8b-agentic-sft-v3` (LoRA r=16, alpha=32, 3 epochs, train_loss=0.12)
- Eval traces: `results/phase6/student_traces_seed42.json`
- Judged eval: `results/phase6/student_traces_seed42_judged.json`
