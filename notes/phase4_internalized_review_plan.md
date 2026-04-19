# Phase 4: Internalizing review-and-revise into the agent's weights

**Date:** 2026-04-19
**Paper mapping:** This document is the plan for **paper Phase 2** — removing the externalized review loop and internalizing it in the student model. Paper Phase 1 (external loop with state-of-the-art agents) is already established; it's what the current `src/experiments/hard_benchmark_runner.py` and Phase 1 post-fix numbers demonstrate.

## Thesis (sharper than Phase 2/3)

Previously we framed distillation as "copy the teacher's answers." That was wrong. The paper's title, *Agents that check their work*, commits to a specific claim: **the student should spontaneously review its own output and revise without being externally prompted to do so.** Modern coding agents no longer need to be told "run the tests" — the behavior is internalized. We want the same pattern for multi-modal output (slides, webpages, video, audio, research, code).

That reframes the student's target behavior as autonomous multi-turn tool-use inside a single assistant turn:

```
user: "Make me X."
assistant:
  <text>   Here's my first draft: …
  <text>   Let me render this and check the layout before submitting.
  <tool>   render_slide(v1)
  <obs>    <image>
  <text>   I see the title is truncated. Fixing…
  <text>   <draft v2>
  <tool>   render_slide(v2)
  <obs>    <image>
  <text>   Looks good. Submitting.
```

No external `[REVIEW]` cue. The review is the assistant's own next token.

## Why Phase 2's approach couldn't have worked

- Our teacher loop fed `[REVIEW]` as a user message, so teacher trajectories showed review *conditioned on external prompting*, not autonomous review.
- The student SFT'd on those trajectories learns the `[GENERATE] → [SUBMIT]` surface conditioned on being told to review, not the *decision* to review.
- 15 tasks × 65 trajectories per task made this worse by forcing the 43M trainable QLoRA params to memorize 15 canonical answers.

Held-out numbers (2026-04-19) confirm: SFT collapsed from in-dist 66.7% to held-out 26.7% — below base 40%. Commit `72bd83b`.

## The fix: synthetic stitched trajectories

No off-the-shelf model spontaneously reviews across all modalities. So we construct training examples by stitching:

**Stage A — First pass.** Teacher (Claude Sonnet 4.6) solves the task single-shot.
**Stage B — Grounded review.** A tool-equipped reviewer produces a *specific* critique: code is executed; slides/webpages are screenshotted and a VLM reads them; research/factual output is checked via web search; video/audio via frame or transcript analysis.
**Stage C — Revision.** Teacher revises conditioned on the specific critique.
**Stage D — Stitch.** Post-process A + B + C into *one assistant turn* by:
  - deleting closing tokens ("submitting", "hope this helps", `[SUBMIT]`);
  - appending a connective phrase from a bank of ~30 variations ("let me verify this by…", "before submitting I'll check…");
  - inserting the reviewer's tool invocation as an assistant `tool_use` block (not a user prompt);
  - inserting the tool result as a `tool_result` block;
  - paraphrasing the critique to first-person ("I see the title is truncated");
  - appending the revision + re-verification + final text.

Stage D is load-bearing. The student must see the review live *inside* the assistant's voice.

## Trajectory format

Each training example:

```
user: <task description>
assistant: [text, tool_use, tool_result, text, tool_use, tool_result, …, text]
```

We use native chat-template tool-use so Qwen3-8B can learn the actual tool-call tokens its tokenizer already knows.

## Diversity (what the 200 trajectories must span)

- **Review outcomes**: ~30% "first pass correct, review confirms, no revise" (teaches stopping); ~50% "first pass wrong, one revise fixes" (core skill); ~20% "two revises". Without the 30% "no revise" stratum the model will over-review even when the answer is right.
- **Bug classes per modality** (code pilot starts here): logic, edge cases, off-by-one, state machine, parser escape rules, encoding, concurrency, comparator transitivity, numerical precision. For multi-modal: layout/overflow/alignment/truncation (slides/web), frame-transition/sync/duration (video), prosody/wrong-text (audio), missing/hallucinated citation (research).
- **Connective language**: ~30 phrasings so the student doesn't overfit one phrase.
- **Tool-call variance**: student must call tools with varied arguments — not a single canonical invocation.

## Tools (stable JSON schemas the student learns)

- `execute_code(code, test_code)` → {stdout, stderr, passed}
- `render_slide(artifact)` → {image_bytes}
- `screenshot_page(html, viewport)` → {image_bytes}
- `sample_frames(video_path, n)` → {images}
- `transcribe_audio(audio_path)` → {transcript}
- `web_search(query)` → {results}
- `view_image(image, question)` → {answer}  # VLM self-critique primitive

Pilot scope uses only `execute_code`.

## Modality rollout

Don't boil the ocean. Ordering:

1. **Code (pilot)** — deterministic reward, one tool, fastest feedback.
2. **Slides + webpages** — shared HTML/SVG → screenshot → VLM pipeline.
3. **Research** — web_search tool.
4. **Video** — frame sampling.
5. **Audio** — ASR + spectrogram.

Only advance after the prior modality holds on autonomous review.

## Training

- **SFT on stitched trajectories.** Reuse QLoRA pipeline on Qwen3-8B. New adapter path: `models/qwen3-8b-autonomous-review-sft/`.
- **GRPO on the same task pool.** Reward = passed-tests (deterministic for code). Stitched SFT adapter as starting point. Adapter: `models/qwen3-8b-autonomous-review-grpo/`.

## Evaluation change: no-scaffold single-shot

The current held-out harness cues `[REVIEW]` between turns. That measures the wrong thing for the new thesis. We need a **no-scaffold eval**:

- Single user turn: "fix the bug" / "make the slide".
- Tools available.
- Model emits as many `tool_use` blocks as it wants inside one turn.
- Terminate on natural stop or token cap.
- Metrics:
  - **Autonomous tool-call rate** — fraction of tasks where the student invokes ≥1 tool without being asked.
  - **Pass@1 with tools** — task passes if any `tool_use` execution returns passing, or if final code passes tests post-hoc.
  - **Regression control** — pass@1 on a no-tools variant should not drop vs. base.

New harness: `src/evaluation/autonomous_review_eval.py`.

## Risks

- **Synthetic trajectory artifacts.** If stitching leaves seams (abrupt voice shifts, repetitive connectives), student learns the seams not the impulse. Mitigate with a diverse connective bank and first-person paraphrasing.
- **Tool hallucination.** Student might emit `tool_use` and then hallucinate a `tool_result`. Mitigate: training only on real tool results; eval enforces real execution.
- **Over-reviewing.** 30% "no-revise confirm" stratum addresses this; check regression on easy tasks.
- **Modality bleed.** Do *not* mix modalities in the pilot. Code-only until the skill demonstrably transfers; then expand.

## Concrete build order (this is what I'm executing)

1. **[DONE — commit 72bd83b]** Baseline: held-out eval harness, greedy + sampled variants.
2. `src/training/generate_tasks_v2.py` — Sonnet 4.6 task generator, 200 fresh code bug-fix tasks, dedup vs existing, validate each bug.
3. `src/training/collect_review_traces_v2.py` — Sonnet 4.6 Stage A/B/C trace runner, parallel workers, prompt-cached shared system prompt.
4. `src/training/stitch_trajectory.py` — Stage D stitcher: strip_closers + connective bank + paraphrase_critique + assemble into single assistant turn.
5. `data/training/sft_review_v1.json` — 200 stitched trajectories ready for SFT.
6. SFT retrain on Qwen3-8B with stitched data (runs after GRPO v2 completes, same GPU).
7. `src/evaluation/autonomous_review_eval.py` — no-scaffold harness.
8. Held-out eval with autonomous review harness → the actual test of the paper's thesis.
9. If SFT works, run GRPO on the same task pool for further gains.

## Budget

Rough order-of-magnitude (Sonnet 4.6 at $3/M input, $15/M output, heavy cache hit via shared system prompt):
- Task generation: 200 tasks × ~2 calls × ~3K tokens ≈ 1.2M tokens ≈ $3–5.
- Trace collection: 200 tasks × ~4 calls × ~5K tokens ≈ 4M tokens ≈ $15–25, mostly cache hits after the first call.
- Paraphrase pass: 200 × ~1K tokens ≈ 200K ≈ <$1.
- **Total: ~$20–30**, under 4 hours wall-time via API concurrency.
