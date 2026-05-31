# Phase 6: Agentic SFT Pipeline — Status

## Motivation

Phase 5 had two structural flaws identified post-hoc:

1. **Trajectory compression flaw**: Phase 1's SOTA scaffolding used single-turn fresh API calls with text-compressed history (`[{role:user, content: spec + last_code + feedback}]`). Phase 5 SFT used full multi-turn growing trajectories — a different distribution than what the original Phase 1 found works.

2. **No tool use flaw**: Phase 5 traces had ZERO tool calls. The harness handled file I/O, rendering, and image reading externally. The student couldn't learn WHEN/HOW to invoke tools — only what to say between turns.

Phase 6 rebuilds the pipeline so the agent itself orchestrates `write_file → bash render → read_file image → critique → revise` cycles via 3 general tools.

## Components built

| File | Purpose |
|---|---|
| `src/training/agent_workspace.py` | Per-trajectory tmpdir sandbox + 3 tools (write_file/read_file/bash) with path-traversal guard and python3 shim. TOOLS_SCHEMA constant in OpenAI tool format. |
| `src/training/collect_agentic_traces.py` | Teacher trace collection. Heavy `AGENT_SYSTEM_PROMPT`. Teacher = `qwen/qwen3-vl-235b-a22b-instruct` (instruct, not thinking — thinking variant burned reasoning tokens and emitted malformed JSON). Pre-populates `test.py` for code tasks. |
| `src/training/judge_agentic_traces.py` | Gemini 3 Flash multimodal judge. Sees per-step summary + final saved screenshot; for code, trusts deterministic `final_passed`. |
| `src/training/prepare_agentic_sft.py` | Converts judge-kept traces into Qwen3-VL SFT JSONL. Collapses `(tool-text + user-image_url)` pairs into a single `tool` role message with image content directly. Replaces heavy agent prompt with stripped `STUDENT_SYSTEM_PROMPT`. |
| `src/training/inspect_agentic_trace.py` | Compact JSON + screenshot copy generator for manual quality review. |
| `src/training/run_agent_student.py` | Student inference loop. Parses `<tool_call>{json}</tool_call>` XML emitted by Qwen3-VL student, dispatches via Workspace, builds tool responses including image content for visual feedback. |
| `src/training/train_vl_sft.py` | (modified) Now passes `tools=` kwarg to `apply_chat_template` so tool schemas auto-inject under `# Tools` section. |

## Trace collection results (94/94 done)

| category | n | final | max_steps | api_error |
|---|---|---|---|---|
| code | 15 | 13 (87%) | 2 | 0 |
| webpages | 38 | 18 (47%) | 18 | 2 |
| slides | 41 | 21 (51%) | 19 | 1 |

Total wall time: 1178 minutes ≈ 19.6 GPU-equivalent hours via OpenRouter (3 workers, 6.5 real hours).

## Judge filter (40/94 kept = 43%)

| category | n | judge-kept | rate |
|---|---|---|---|
| code | 15 | 13 | 87% |
| webpages | 38 | 11 | 29% |
| slides | 41 | 16 | 39% |

By status:
- `final` traces: 36/52 kept (69%)
- `max_steps` traces: 4/39 kept (10%) — most max_steps had unresolved layout issues
- `api_error`: 0/3 kept (correctly)

## SFT JSONL (35/40 = drop 5 over 80k chars)

`data/training/agentic_sft.jsonl` — 35 examples:
- code: 13 (incl. 11 one-shot solves + 2 multi-iteration debugs)
- webpages: 6 (5 dropped for length — large HTML files written multiple times)
- slides: 16

5 long examples dropped (60-90k tokens approx) — webpages with multi-write 15-18kB HTML files. Could revisit by truncating intermediate write_file content if webpages performance lags.

## Manual quality verification

Spot-checked traces include:
- `code_001` (one-shot fix, 3 steps, passed): clean write→bash→stop
- `code_010` (max_steps, failed): wrote solution with syntax error, never recovered. Correctly judge-rejected.
- `web_001` (final, 9 steps, judge-kept): write → render → read → critique-with-specific-issues → rewrite → render → read → approve. Substantive.
- `web_004` (max_steps, judge-rejected): real iterative behavior with substantive critique, but final masonry layout was broken. Judge correctly catches "standard grid with equal-height rows, not true masonry".
- `slide_001` (max_steps, judge-rejected): final image shows attention equation overlapping architecture boxes. Judge correctly catches "severe layout failure".
- `slide_002` (final, 7 steps, judge-kept): clean ML pipeline diagram with 6 stages, arrows, bullets, training loss curve. Verified by eye.

Judge rationales reviewed for 3 webpages + 3 slides judge-kept traces — all cite specific defects observed in screenshots.

## Verified plumbing

- Qwen3-VL chat template renders `tool_calls` as `<tool_call>{json}</tool_call>` XML
- `tool` role messages with image content render as `<tool_response>...<|vision_start|><|image_pad|><|vision_end|>...</tool_response>` in user-role wrapper
- `tools=[...]` kwarg auto-injects schemas under `# Tools` section in system message
- Student parser correctly extracts inline tool calls and strips them from assistant content

## Remaining steps

1. **Run SFT** — `train_vl_sft.py --data data/training/agentic_sft.jsonl ...`. Currently blocked by GPU contention (verl GRPO from another project at ~55GB used).
2. **Held-out eval** with `run_agent_student.py` on `heldout_phase5.json` (18 hard tasks).
3. **Pass@k inference scaling** — sample 2-3 trajectories per task at temperature 0.7.
4. **DAPO trace generation** (longer-term): collect K rollouts per task, judge each, pair good vs bad for outcome-conditional RL training.

## Open questions

- Will the agentic format help, hurt, or be neutral vs Phase 5 V3+fc4k? Phase 5 demonstrated ~44% judge-keep on hard tasks single-shot, 56% pass@3.
- Webpages are still the weakest category (only 6 SFT examples after length filter). Slides may still dominate behavior transfer.
- The student needs to learn the `<tool_call>` XML emission format the chat template expects — Phase 5 V3 was trained without ANY tool-call format exposure, so the base capability is unknown.
