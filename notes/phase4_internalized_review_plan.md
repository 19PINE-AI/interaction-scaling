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

## Phase 4 SFT result — held-out autonomous-review eval (2026-04-19)

Pipeline ran end-to-end: 204 tasks generated → 203 raw traces collected → 203 stitched examples → Qwen3-8B QLoRA SFT (3 epochs, 39 optimizer steps, ~12 min wall, bs=1, grad_accum=16, lr=2e-4, max_length=6144). Adapter at `models/qwen3-8b-autonomous-review-sft/`.

Held-out eval (15 tasks, greedy, max 3 tool calls), `results/autonomous_review/heldout_sft_vs_base.json`:

| Model      | pass@1 | tool@1 |
|------------|--------|--------|
| base       | 20.0%  | 66.7%  |
| sft_review | 26.7%  | 86.7%  |

Two signals:
1. **+20 pp autonomous tool-call rate** (66.7% → 86.7%) — SFT clearly instills the "call `execute_code` before answering" habit. On 3 of the 5 tasks where base skipped the tool entirely, SFT still skips — the behavior isn't universal.
2. **+6.7 pp pass rate** is real but modest (3/15 → 4/15).

Artifact — **all 4 SFT passes come via `final_pass`, none via `tool_pass`**. The SFT model emits the correct code but its tool-call runtime result is always `passed=False`. The executor uses `args.get("test_code", task["test_code"])`, so if the model echoes a hallucinated/empty `test_code` string in its tool args, execution runs against the wrong assertions. Base model hits `tool_pass=True` on all 3 of its wins, so this is SFT-specific regression in tool-arg fidelity. Fix in the next iteration: make the executor ignore the model-supplied `test_code` and always use the ground-truth one, OR tighten the stitcher to emit compact test_code references rather than full strings.

SFT also collapsed onto `calls=1` for **every single held-out task** (vs. base which probes up to 3 times when failing). One-shot verification pattern was the 30% stratum in training; it appears to have dominated generalization. Consider rebalancing strata (e.g., 50/50) or adding an explicit "retry after failure" stratum with >1 execution.

**Verdict:** SFT on synthetic stitched trajectories produces a measurable but small behavioral + capability shift. The tool-arg bug masks the capability gain — re-eval after fixing the test_code handling will give a cleaner signal before deciding on Phase 4 GRPO.

### Post-patch rerun (2026-04-20, `heldout_sft_vs_base_v2.json`)

Hard-override applied in `autonomous_review_eval.py:207`: executor always uses `task["test_code"]`, ignoring model-supplied test_code.

| Model      | pass@1 | tool@1 |
|------------|--------|--------|
| base       | **26.7%**  | 66.7%  |
| sft_review | 26.7%  | 86.7%  |

**Pass rates tie at 26.7%.** The apparent SFT pass-rate advantage from the pre-patch run was a measurement artifact: base h011 had flipped FAIL→PASS because the broken executor had been feeding it false-fail signals, preventing recovery. With correct tool signals, base now recovers on h011 too and matches SFT.

**Task-level diff (only two changed):**
- `code_h011`: base FAIL→PASS (correct tool signal enabled recovery via `final_pass`).
- `code_h014`: sft_review calls=1→3 (iterated more, still FAIL).

**What the SFT actually does, cleanly measured:**
1. **+20 pp autonomous tool-call rate** (66.7% → 86.7%). Real behavioral shift — SFT teaches "verify before answering."
2. **Zero pass-rate gain** (26.7% vs 26.7%). The bug-fixing capability of the underlying 4-bit Qwen3-8B is unchanged; SFT only changes *when* it invokes the tool, not *what code it produces*.
3. **SFT `tool_pass` is still 0 everywhere** — not an executor artifact after the patch, so the SFT model's tool-call code is genuinely wrong. It writes a buggy draft in the tool call, sees failure, then produces corrected code in the follow-up visible text. Matches the `one_revise` training pattern, except the revision never gets a second tool call to verify itself.
4. **SFT collapsed to `calls=1`** on 13/15 tasks (only h014 iterated to 3 after patch). Training mix was 30% no_revise / 70% one_revise, but generalization overweights "one tool call then stop."

**Implications for Phase 4 GRPO:**
- Rewarding `tool_pass=True` directly (not just `final_pass`) should push the SFT distribution toward emitting correct code inside the tool call on the first shot.
- Could also help with the calls=1 collapse if rewards credit a second successful verify.
- But capability ceiling is the 4-bit quantized base model — GRPO won't unlock new knowledge, only better behavior. Expected upside: modest, similar in magnitude to +20pp tool-call rate.
- **Recommendation:** run Phase 4 GRPO with rewards = { `tool_pass`: +1.0, `final_pass`: +0.3, `tool_called`: +0.1 } to explicitly push toward correct-code-in-tool-call behavior rather than just tool invocation.

### SFT v2 — bug-comment scrub (2026-04-20, `heldout_sft_v1_vs_v2.json`)

**Data finding from trace inspection:** the Sonnet-generated `buggy_code` in 46% of one_revise training examples contained explicit bug-label comments like `# BUG: heap[0] is negative, so this is always True` or `# Missing pushdown before recursing`. The SFT model learned to literally emit self-labeled buggy code inside its first tool call — a bizarre pattern with no real-world analog.

Fix: `src/training/scrub_leaky_comments.py` strips bug-annotation comments from all `buggy_code` and `fixed_code` in the tasks file. Full-line and inline variants handled; 204/204 tasks still validate (bug fails, fix passes) after scrub. Zero leaks remain in `sft_review_v2.json`.

Retrained SFT (`models/qwen3-8b-autonomous-review-sft-v2/`, same hyperparams, ~10 min). Held-out eval:

| Model         | pass@1 | tool@1 | tool_pass@1 |
|---------------|--------|--------|-------------|
| base          | 26.7%  | 66.7%  | 20.0%       |
| sft_review v1 | 26.7%  | 86.7%  | 6.7%        |
| **sft_review v2** | **40.0%** | 86.7%  | **13.3%**   |

**+13.3 pp pass rate (v1→v2) from a data-side fix alone.** No algorithm change.

Task-level: two FAIL→PASS flips (h006, h014), both via `tool_pass=True` with calls=2 and calls=3 respectively — v2 genuinely iterates and converges to correct code inside tool calls, rather than v1's "buggy draft + better final text" pattern. Zero regressions.

**Implications:**
- The pre-scrub data was actively teaching pathological behavior — the model was spending capacity on reproducing bug-annotation comments.
- `tool_pass@1` doubled (6.7% → 13.3%) — the clearest capability signal. The model now produces correct code *inside* the tool call more often, not just in follow-up text.
- SFT v2 is the new baseline for Phase 4. GRPO should be trained on top of v2, not v1.

**Remaining head-room:** 6 still-failing tasks (h001/h003/h005/h011/h013/h015) — h001/h003 don't call the tool at all, h005/h011/h013/h015 call once and fail. A future training round might rebalance the no_revise/one_revise strata (v2 still has 30/70) or add multi-revision patterns.

---

## Phase 4 GRPO — first-shot tool_pass reward (2026-04-20)

Warm-started from SFT v2 adapter. Single-turn GRPO targeting the main remaining regression in SFT v2 (tool_pass@1 only 13.3% — the model still often emits buggy code inside its first tool call).

**Script:** `src/training/train_grpo_review.py`.
**Reward:**
- `+1.0` if the first `<tool_call>`'s `code` argument passes `task["test_code"]`.
- `+0.1` if a valid `<tool_call>` is emitted but the code fails tests.
- `0.0` if no parseable tool_call.
The reasoning: push the model toward correct first-shot code without rewarding the "buggy draft then correct final text" pattern.

**Config:** TRL 1.1.0 GRPOTrainer, 1 epoch, bs=1 grad_accum=8 num_generations=2 max_completion=2048 lr=5e-6 beta=0.04, 51 steps, ~73 min wall. Adapter: `models/qwen3-8b-autonomous-review-grpo/`.

**Ops note:** First attempt with num_generations=4 was silently OOM-killed after step 5. Reducing to 2 stabilized; GPU had 97 GB but the system's 8 GB swap was nearly full before launch, so any spike in Python allocator tripped the kernel OOM killer. Lesson: on this host, treat num_generations=2 as the safe ceiling for Qwen3-8B QLoRA GRPO with 2048 completion tokens.

**Held-out eval (`heldout_grpo_v2_base.json`):**

| Model          | pass@1 | tool@1 |
|----------------|--------|--------|
| base           | 26.7%  | 66.7%  |
| sft_review_v2  | **40.0%** | 86.7%  |
| grpo_review    | 33.3%  | 86.7%  |

**GRPO regressed 6.7pp from SFT v2** while still beating base by 6.7pp. Per-task diff vs SFT v2: GRPO lost h014 (SFT v2 passed via 3-call revision, GRPO stopped after calls=1). No task was flipped FAIL→PASS by GRPO that SFT v2 didn't already solve.

**Why GRPO hurt:** The reward placed 10× more weight on tool_pass than on valid-call-that-fails (1.0 vs 0.1). Under GRPO's group-relative advantage, trajectories where calls=1 and the code happened to pass were the only strongly-positive signal. This sharpened the already-problematic `calls=1` collapse from SFT v2 rather than encouraging multi-call revision. The +0.1 floor for failing tool_calls was not enough to preserve revision behavior when the alternative (stop after one call) sometimes got +1.0.

**SFT v2 remains the best adapter.** GRPO v1 is archived but not superseding.

---

## Phase 4 pass@k sampling eval — the interaction-scaling thesis test (2026-04-20)

**Setup:** 15 held-out tasks × 5 samples @ T=0.7 top_p=0.95, two checkpoints (base, sft_review_v2). 150 total trials, ~3h37m wall. Output: `results/autonomous_review/heldout_passatk.json`. Log: `logs/autonomous_review_passatk.log`.

The thesis: a model trained to autonomously invoke tools and revise should benefit disproportionately from multi-sample coverage — pass@k should widen the gap that pass@1 opened.

**Result:**

| Model          | pass@1 | pass@5 | tool@1 | tool@5 |
|----------------|--------|--------|--------|--------|
| base           | 21.3%  | **40.0%** | 66.7%  | 93.3%  |
| sft_review_v2  | 24.0%  | **40.0%** | 88.0%  | 93.3%  |

**pass@5 ties at 40% (6/15 tasks each). tool@5 ties at 93.3% (14/15 tasks each).** The thesis is not supported in this form.

**Per-task pass@5 breakdown:**
- SFT v2 wins at k=5 on h006 (0→1), h014 (0→2) — tasks where base never invoked the tool or gave up early, and SFT v2's higher tool-call rate let it stumble into a pass at least once.
- Base wins at k=5 on h005 (1→0), h011 (2→0) — tasks where base's higher variance in tool-call count (sometimes calls=3 with revision) produced a pass at least once, while SFT v2's `calls=1` collapse capped it.
- Net wins cancel out.

**Why tool@1 +21pp doesn't widen at k=5:** Base's per-sample tool-call rate at T=0.7 is volatile (some tasks 5/5, some 0/5). But across 5 samples, base invokes the tool at least once for 14/15 tasks. SFT v2 also does 14/15. The "always calls tool" behavior is valuable for pass@1 and for deployment cost predictability, but not for pass@k coverage.

**Interpretation for the paper:**
1. Per-sample tool-call consistency (an SFT gain) ≠ pass@k gain. The effective tool invocation rate at moderate k converges across behaviorally-divergent models.
2. SFT v2's `calls=1` collapse actively prevents its pass@k from growing: when base revises via multiple calls, SFT v2 stops after one, losing the coverage that revision would have provided.
3. The interaction-scaling story as stated (more interaction → higher ceiling) should be reframed: **more interaction only pays off when the model uses the extra tool calls to produce meaningfully different code**. An SFT that teaches "one call always" is worse than a base model that stochastically revises.

**Implications for future work:**
- To confirm the thesis, need a model that *adaptively* spends more calls on harder tasks — not a model that always does one call or always does three. The training signal would need to condition the revise pattern on observed failure.
- A joint metric worth tracking: `pass@k - base_pass@k` as a function of k. If it stays flat or negative, the training hasn't improved the coverage ceiling.
- The next training experiment worth running is not "more GRPO on first-shot," but "SFT with a 50/50 no_revise/one_revise split + an explicit `retry_after_failure` stratum forcing ≥2 calls on initially-failing drafts."

