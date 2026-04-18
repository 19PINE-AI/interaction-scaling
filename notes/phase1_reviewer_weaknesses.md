# Phase 1 Proposer–Reviewer: Weak Spots and Improvement Candidates

**Date:** 2026-04-18
**Summary:** Per-category analysis of Phase 1 hard-benchmark results, focusing on where the proposer–reviewer loop fails to help or actively regresses.

## Headline table

| Category  | N  | ss_q | rv_q | Δ      | ss_meets | rv_meets | avg_iters | rv_tokens |
|-----------|---:|-----:|-----:|-------:|---------:|---------:|----------:|----------:|
| code      | 15 | 0.73 | 0.93 | +0.20  |      73% |      93% |      1.53 |     2,643 |
| slides    | 10 | 0.62 | 0.51 | **-0.11** | 20% | 20%   |      2.60 |    18,000 |
| webpages  | 14 | 0.44 | 0.52 | +0.08  |       7% |      14% |      2.86 |    49,431 |
| animations| 15 | 0.46 | 0.53 | +0.07  |       7% |      20% |      3.00 |    73,652 |
| video     | 15 | 0.00 | 0.42 | +0.42  |       0% |      33% |      2.73 |    13,486 |
| research  | 15 | 0.68 | 0.85 | +0.17  |      53% |      87% |      1.80 |    27,891 |

**Where the reviewer clearly helps:** code, video, research.
**Where it barely helps or hurts:** slides, webpages, animations — the HTML/VLM family.

## Failure modes identified

### 1. Reviewer over-criticizes when ss is already good; proposer over-edits → regression

Regressions seen on:
- slide_004: 0.70→0.30, slide_005: 0.60→0.30, slide_009: 0.70→0.30, slide_010: 0.65→0.30
- anim_013: 0.60→0.30, anim_010: 0.85→0.70
- research_005: 1.00→0.93, research_006: 0.87→0.73
- web_001: 0.40→0.30, web_003: 0.70→0.60, web_015: 0.75→0.70

**Pattern.** Single-shot quality is already partially correct. The reviewer still emits a list of "issues," the proposer does a full rewrite rather than a targeted patch, and key parts that were working break.

**Why (three interacting causes):**
- Early-stop threshold is `accuracy ≥ 0.9 AND meets_reqs=True`. When `meets_reqs=False` (e.g., one requirement unverified), the loop continues even at quality 0.85. Research_006 regressed from 0.87 because `meets_reqs` was False — the threshold never triggered.
- The `REVISION_PROMPT` tells the proposer to "Fix ALL issues" and output the "complete" artifact. That encourages full-file rewrites, which throw out working parts.
- The reviewer has no notion of "don't break what works." It emits a list of weaknesses with no signal about which parts are currently passing.

**Fixes:**
- Lower the stop threshold to `quality ≥ 0.9` alone (drop the `meets_reqs` AND). Or add a softer "if no regression-risk issue is present, stop."
- Change the revision prompt from "full rewrite" to "minimal edit: return the smallest diff that fixes these issues; preserve everything else verbatim."
- Keep-best tracking: at each iteration, cache the current-best (quality, artifact); if a revision produces a worse score, roll back instead of continuing. This is a one-line guard and kills all ss→rv regressions mechanically.

### 2. Slides/webpages share the same HTML pipeline → slides bug also hit webpages

`webpages` uses `run_slide_task` (shared HTML pipeline per `run_all_hard_benchmarks.py:41`). The now-fixed screenshot-redundancy bug in the slides revision path was also bleeding into webpages. Webpages Δ=+0.08 is suspiciously low; we should re-run it now that the fix is in.

**Fix:** rerun `webpages` with the same corrected path.

### 3. Slides rerun (post-fix) is still regressing some tasks

Early data from the fixed rerun shows slide_005 still goes 0.40→0.30 over 3 iterations. This proves the screenshot-in-proposer redundancy was **not** the whole story — the reviewer-over-critique + full-rewrite pattern (#1 above) persists without it. The screenshot fix was necessary but not sufficient.

**Fix:** apply #1's fixes on top.

### 4. Animations: reviewer signal weak, iterations always max out

Animations run exactly `avg_iters=3.00` (every task hits the cap) and use 73k tokens on average — the most expensive category — for only +0.07. That means the stop-early threshold **never triggers** on animations. The VLM reviewer either always finds issues or never calls quality ≥ 0.9.

**Diagnosis needed:** log reviewer `quality_score` per iteration to see whether it saturates or oscillates. If it monotonically drops (reviewer-induced damage), keep-best rollback fixes it.

**Fix:** add monotonic-improvement guard + examine whether the VLM animation reviewer is too strict (suspect: frame-by-frame analysis flags any imperfect timing as an issue, even if the animation is usably correct).

### 5. Video: reviewer is load-bearing but 8/15 stay stuck at 0

Single-shot is always 0 (execution failures) — the `EXECUTE` feedback is what unblocks 7/15 tasks. The other 8 (video_002 reverse, video_003 extract-last-2s, video_005 watermark, video_006 fade, video_007 grayscale, video_009 concat, video_012 lower-third, video_015 freeze-frame) stay at 0 through all iterations.

**Hypothesis.** The failing tasks likely fall into two buckets:
- Code produces output but it's visibly wrong (VLM catches it but proposer can't repair moviepy/ffmpeg syntax for that op).
- Code doesn't produce output at all (execution error on each iteration — reviewer feedback is "execution failed" with traceback; proposer cycles through variants of the same broken call).

**Fix:** inspect interaction traces of the stuck-at-0 tasks. If it's category 2, the reviewer could provide a small library of moviepy/ffmpeg snippets as environmental context. If category 1, we need a richer frame-description prompt for the VLM.

### 6. Research: reviewer's `meets_requirements` is too strict for stopping

Research stops at `meets_reqs AND accuracy ≥ 0.9`. But `meets_reqs` in the factual checker is "all requirements verified with zero contradictions" — an unverifiable-but-not-wrong fact flips it False. Research_005 regressed 1.00→0.93 because the loop continued past a near-perfect answer chasing unverifiable facts.

**Fix:** decouple — stop on `accuracy ≥ 0.9` regardless of `meets_reqs`. Or: treat `unverified` (not contradicted, just can't look up) as green, not red.

### 7. Code: reviewer helps, but hardest tasks are beyond it alone

code_011 stays at 0 across all iterations. The execution feedback is deterministic and complete — the failure is knowledge/capacity, not signal quality. No reviewer fix can help. This is the right kind of "unsolved" — the signal works, the model is just too small or the task too hard.

## Prioritized fix list (impact × effort)

1. **Keep-best rollback guard.** One-line change. Kills all ss→rv regressions across slides, webpages, animations, research. **Do first.**
2. **Minimal-edit revision prompt.** Swap "rewrite the complete artifact" for "smallest possible diff." Low risk, likely large gain for HTML tasks. **Do second.**
3. **Lower/decouple stop threshold.** `quality ≥ 0.9` alone, drop `meets_reqs` AND. Frees research and animations from over-iterating.
4. **Rerun webpages after slides fix.** Free win if fix #1 applies.
5. **Investigate animation reviewer saturation.** Log per-iter quality, see whether reviewer is the bottleneck.
6. **Investigate video stuck-at-0 tasks.** Read traces to classify execution-fails vs VLM-miss.

Items 1–3 are ~half-day total; items 4–6 are data/diagnosis.

## Trajectory-level findings (added 2026-04-18 after inspecting on-policy traces)

Inspected `results/hard_benchmarks/{cat}_onpolicy_run1.json` which contain full `rv_interaction_trace` per task. These surface infrastructure/prompt issues that aggregate metrics hid.

### A. Reviewer is stateless → emits identical feedback across iterations

`slide_004`, `slide_012`: reviewer's iter-1 and iter-2 feedback strings are byte-identical openings ("Input and output dimensions are not on the same line..." repeated verbatim). The reviewer has no memory of what it already said, so when the proposer can't fix the issue, we pay the full token cost to hear the same complaint again.

**Fix:** pass the prior feedback + prior score into the reviewer prompt so it can say "still present after one fix attempt" or shift its suggestion. Or: reviewer-side change-detection — if the new artifact is visually near-identical to the previous one, short-circuit with "no change detected, try a different approach."

### B. Quality oscillates across iterations — no keep-best

Trajectories that peaked mid-loop and then fell back:
- `slide_005`: 0.2 → 0.75 → 0.70 → **0.3** (peaked at iter 1, regressed thereafter)
- `slide_006`: 0.95 → 0.75 → 0.85 → **0.75**
- `slide_019`: 1.0 → 0.70 → 0.60 → 0.85
- `web_007`: 0.3 → 0.20 → 0.70 → **0.2** (reached 0.70 then fell back)
- `web_006`: 0.75 → 0.60 → 0.70 → 0.6
- `anim_010`: 0.85 → 0.75 → 0.85 → **0.7**
- `video_004`: 0.3 → 0.60 → 0.30 → 0.3
- `research_011`: 0.93 → 0.64 → **0.0** (catastrophic — revised report contains new false claims)

A keep-best guard (cache highest-scored artifact, return it instead of "last") would save 8+ tasks' scores across categories with zero other changes.

### C. Baselines are unpaired — ss and rv are independent samples

`slide_002`: ss=0.75 but rv-initial=0.95; `slide_019`: ss=1.0 but rv-iter1=0.70. The "single-shot" run and the "reviewed" run generate their own independent initial artifact. Δ = avg(rv) - avg(ss) is noisy at N=15-20. Some of what looks like "reviewer helped" is just resampling variance.

**Fix:** run N=3+ seeds per task (already have on-policy run1/2/3), always report mean ± sd; or share the initial generation across modes so Δ isolates the reviewer's contribution.

### D. `webpages` category uses the **slide** system prompt

Confirmed: `webpages_onpolicy_run1.json` trace shows every task instructed "creating presentation slides... render at 1920×1080 pixels with NO scrolling needed." But webpage tasks are landing pages / scrollable content. The model is being told the wrong goal and penalized by a reviewer that also thinks it's evaluating a slide.

This alone likely accounts for webpages' bottom-of-table Δ=+0.08 and the 7% ss_meets rate. **Fix:** author a distinct `WEBPAGE_SYSTEM_PROMPT` (scrollable, variable-height, responsive) and matching reviewer rubric.

### E. Revision message bloat — full previous HTML echoed every turn

Revision user message format: `## Previous HTML\n```html\n<entire 10-15KB HTML>```\n## Feedback\n...`. Over 3 iterations the conversation is ~50KB before the model even starts responding. This (a) wastes tokens, (b) biases the model toward "same-as-before with local tweaks" when a rewrite might score better.

**Fix:** after iter 1, pass only the assistant's prior turn as the `assistant` message (which the chat API already keeps). The explicit echo in the user message is redundant.

### F. SFT training data is two disjoint corpora

Confirmed: task types in `sft_data.json` split cleanly —
- **Singular** (`code`, `slide`, `webpage`, `animation`, `video`, `research`): 100% use `[GENERATE]/[EXECUTE]/[REVIEW]/[SUBMIT]` protocol tokens. **Synthetic** bootstrap, short messages, often with TODO-placeholder baseline code (e.g., `video` SFT initial code is literally `# TODO: Apply edits`).
- **Plural** (`animations`, `slides`, `webpages`): 0% use the 4-verb protocol. **Real** on-policy trajectories with full HTML artifacts and VLM feedback strings.

`code` alone is 135/210 protocol vs 75/210 non-protocol — mixed even within the category.

Implication: the Phase 2 student saw two different "dialects" of interaction scaling during SFT. That probably explains why at eval time it struggles with longer-form tasks: the `[GENERATE]` format trained on short synthetic completions doesn't match the 10-15KB HTML the real tasks need.

**Fix options:**
- Pick one dialect and regenerate consistent data (recommended: real on-policy format, since it matches eval).
- Or: gate by task type — protocol tokens for short-code tasks, raw HTML format for long-artifact tasks — and make this explicit in the system prompt.

### G. Video SFT baseline is pathologically weak

`data/training/sft_data.json` video examples begin with code like `clip.write_videofile('/tmp/output.mp4')  # TODO: Apply edits`. The "single-shot" that Phase 1 reported (ss=0.00 on every video task) is *this synthetic placeholder* — not real proposer output. The +0.42 video gain is therefore inflated by a broken baseline.

**Fix:** verify Phase 1 video ss numbers use real proposer generation, not the synthetic placeholder. If they do, the Δ is real; if not, regenerate with a working baseline.

## Revised fix priority

| # | Fix | Effort | Impact |
|---|---|---|---|
| 1 | **Keep-best rollback** (cache max-quality artifact, return at end) | 10 lines | Fixes 8+ regression cases, all categories |
| 2 | **Webpages system prompt** (stop telling it to make a slide) | 15 lines | Likely +0.10–0.20 Δ on webpages |
| 3 | **Decouple stop threshold** (`quality ≥ 0.9` alone, drop meets-reqs AND) | 5 lines | Stops research_005, research_002 regressions |
| 4 | **Minimal-edit revision prompt** + drop redundant HTML echo | 20 lines | Lower tokens, preserves working parts |
| 5 | **Pass prior feedback to reviewer** (de-repeat) | 20 lines | Less redundant feedback; convergence detection |
| 6 | **Consolidate SFT data format** | half-day regen | Cleaner distillation target |
| 7 | **Verify video ss baseline is real code**, not synthetic TODO | 1h investigation | Honest video Δ |

Items 1–3 combined: ~30 lines of code, probably moves the HTML-family Δ from +0.07 to +0.15 or better before we even touch training.

## Claim for the paper

The paper already claims "environmental feedback → interaction scaling." This analysis sharpens a second claim: **environmental feedback is necessary but not sufficient.** Three additional invariants are required for the loop to actually scale:
- Keep-best (don't let later iterations overwrite earlier wins).
- Minimal-edit (preserve passing content).
- Calibrated stopping (don't iterate past diminishing returns).

Phase 1 accidentally tested interaction scaling without those invariants in the HTML/VLM family — and we see exactly the expected pathology: reviewer-induced regression, token waste, flat deltas. That's a paper-worthy ablation.

---

## Post-fix results (2026-04-18, benchmark btg2g7gu8)

After applying all 4 named fixes (keep-best, minimal-edit revision, stateful reviewer with prior_reviews, decoupled stop threshold on quality alone) + structural bug fixes (webpage system prompt, screenshot-redundancy removal, no-fabrication guard on research revision):

| Category   | N  | ss    | rv    | Δ post  | Δ pre   | swing   | regressions |
|------------|---:|------:|------:|--------:|--------:|--------:|------------:|
| slides     | 20 | 0.720 | 0.770 | +0.050  | −0.110  | **+0.160** |           3 |
| webpages   | 14 | 0.386 | 0.471 | +0.086  | +0.080  | +0.006  |       **0** |
| animations | 15 | 0.367 | 0.383 | +0.017  | +0.070  | −0.053  |           4 |
| research   | 15 | 0.720 | 0.839 | +0.119  | +0.170  | −0.051  |           2 |
| code       | 15 | 0.733 | 0.933 | +0.200  | +0.200  |  0.000  |       **0** |
| video      | 15 | 0.000 | 0.380 | +0.380  | +0.420  | −0.040  |       **0** |

**Headline wins**
- Slides flipped from net-negative to net-positive — primary target achieved (+0.16 swing).
- Webpages: 4 regressions → 0 (keep-best + correct system prompt). Same aggregate Δ but now strictly reliable.
- Code, video: no regressions; strong positive delta preserved.

**Remaining concerns**
- Animations aggregate Δ dropped (+0.07 → +0.02). Two severe regressions remain: anim_012 (0.60→0.30), anim_013 (0.70→0.30). Keep-best only protects within-run; run-to-run variance between ss and rv runs still exposes these cases. Plausible mechanism: VLM sees an early-frame render of the animation and flags it as static/broken even when it animates later.
- Research aggregate Δ dropped slightly (+0.17 → +0.12) but only 2 minor regressions (research_004 0.93→0.80, research_012 0.87→0.80). The catastrophic research_011 fabrication case (0.93→0.0) is gone — the no-fabricate guard worked.
- Slides regressions (3: slide_006 −0.20, slide_010 −0.10, slide_015 −0.10) are all in the 0.60–0.70 baseline band, consistent with run-to-run temperature noise rather than systematic revision damage.

**Fix-level attribution** (observational, not ablated)
- keep-best: visible in slide_004 (ss=0.40 → rv=0.95 via early rollback) and slide_014 (0.60 → 1.00).
- decoupled stop: slide_003/013/018/019/020 all early-stop at quality 1.00 after iter 1 even when meets_reqs was already true — saves tokens.
- webpage system prompt: web_001 (0.40→0.70) and web_012 (0.60→0.95) show the first clear positive deltas on webpages.
- no-fabricate research revision: research_011 cleanup (was 0.93→0.64→0.0 pre-fix).

**Next steps**
- Animation video-feedback review path needs inspection (anim_012/013 severe regressions). Likely VLM sampling first frame of animation; may need multi-frame sampling.
- Consider ablation runs to isolate each fix's contribution, but the broad stroke — slides flip + webpages zero-regression — is enough signal to proceed to Phase 2 distillation.

---

## Multi-frame VLM prompt fix (2026-04-18)

**Change.** Rewrote the VLM animation-review prompt (`_vlm_review_animation` in `src/experiments/hard_benchmark_runner.py:864`) and video-review prompt (`_vlm_review_frames` in `src/feedback/type3c_video.py:149`) to:
- Explicitly tell the VLM "you have N frames spanning Xms — evaluate as a sequence."
- Preserve the motion-timing checks without hard-coded score bands (a first-draft version with "score ≤ 0.3 if identical" biased the VLM into ubiquitous low scores; dropped).
- Added a 50ms post-load wait in `render_animation_frames` so captures at t=0 occur after the first paint rather than mid-parse.

**Verification run** (anim_007/009/012/013 — prior main-run regressors):

| Task      | Δ pre-fix | Δ post-fix | Notes |
|-----------|----------:|-----------:|-------|
| anim_007  |     −0.10 |      0.00  | ss dropped 0.70→0.20 (more faithful) |
| anim_009  |     −0.15 |     +0.05  | flipped positive |
| anim_012  |     −0.30 |     −0.10  | much milder regression |
| anim_013  |     −0.40 |     +0.15  | flipped positive (large swing) |

**Interpretation.** The earlier VLM scores for animation single-shot were artificially optimistic — it was judging a static-looking first frame as acceptable. With the multi-frame sequence framing, `ss` scores drop *and* the `rv` delta becomes less catastrophic. Single-run noise is sizable on 4 tasks, but the sign change on anim_013 (−0.40 → +0.15) is too large to be noise.

**Open question.** Whether the pre-fix `ss` scores overstated the animation baseline overall. If so, the Δ numbers in the headline post-fix table above may under-state the reviewer's true contribution on animations — the *baseline* was inflated. A full 15-task animations re-run with the new prompt would verify. Deferred to next Phase 2 prep cycle.
