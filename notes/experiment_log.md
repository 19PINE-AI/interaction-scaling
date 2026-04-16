# Experiment Log — Hard Benchmarks

## Status: ALL 6 CATEGORIES COMPLETE (2026-04-16, post-bugfix re-run)

### System Fixes Applied Before Re-run
13 bugs fixed across the benchmark system:
- **CRITICAL**: `python` → `sys.executable` (code never ran), fence parsing ValueError,
  budget allocation unused, exp2 missing parameter, best-of-N inconsistent eval,
  multiprocessing client init
- **MEDIUM**: hardcoded `/tmp`, fence stripping in video/factual, feedback_type mismatch,
  metrics doc, function extraction, truncate_output
- **Slide regression fix**: 3 bugs in review pipeline (suggestions discarded, revision
  was text-only without screenshot, feedback formatting broken)

### Summary Table (Final Results — ALL COMPLETE)

| Category | N | Feedback | Single-shot | Reviewed | Delta | Status |
|----------|--:|----------|:-----------:|:--------:|------:|--------|
| **Video** | 15 | Type 3c: Exec+Keyframe | 0.000 quality | **0.420** quality | **+0.420** | COMPLETE |
| **Code** | 15 | Type 3a: Execution | 73.3% pass | **93.3%** pass | **+20.0pp** | COMPLETE |
| **Research** | 15 | Type 3d: Factual | 0.676 accuracy | **0.849** accuracy | **+0.173** | COMPLETE |
| **Web pages** | 14 | Type 3b: Visual | 0.443 quality | **0.521** quality | **+0.079** | COMPLETE (1 errored) |
| **Animations** | 15 | Type 3c: Temporal | 0.463 quality | **0.530** quality | **+0.067** | COMPLETE |
| **Slides** | 20 | Type 3b: Visual | 0.750 quality | **0.775** quality | **+0.025** | COMPLETE |

### Completed Category Details

#### Code (15 tasks)
- Single-shot: 11/15 pass (73.3%)
- Reviewed: 14/15 pass (93.3%), delta = +20pp
- 3 tasks fixed by review: code_001, code_003, code_008 (2-3 iterations each)
- 1 task resists review: code_011 (CJK text wrapping, 5 iterations, still fails)
- **Previous result was INVALID**: `python` binary didn't exist, so tests never ran.

#### Video Editing (15 tasks)
- ALL single-shot attempts produce broken code (quality 0.00)
- Review fixes 5/15 to perfect (1.0), partially fixes 2/15
- Strongest delta of all categories (+0.420)
- Video feedback combines execution errors + keyframe VLM review — most actionable
- Tasks that worked: video_001 (2 iters), 008 (2 iters), 010 (2 iters), 013 (2 iters), 014 (3 iters)
- Partial fixes: video_004 (0.70), video_011 (0.60)

#### Slides (20 tasks)
- Regression FIXED: previous -0.11 is now +0.025
- Clear wins: slide_001 (+0.15), slide_002 (+0.15), slide_015 (+0.30)
- Minor regressions: slide_007 (-0.10), slide_010 (-0.10)
- Most tasks unchanged — review correctly identifies good slides and doesn't over-revise
- Fix validated: multimodal revision messages (screenshot + text) prevent destructive edits

#### Animations (15 tasks) — NEW
- Single-shot avg: 0.463, Reviewed avg: 0.530, delta = +0.067
- Notable wins: anim_004 (0.00 → 0.85), anim_012 (0.30 → 0.85), anim_009 (0.60 → 0.75)
- Notable regressions: anim_002 (0.75 → 0.70), anim_010 (0.85 → 0.70), anim_013 (0.60 → 0.30)
- Mixed results: review helps broken animations but can degrade working ones
- VLM temporal feedback (multi-frame) provides moderate signal

#### Web Pages (14/15 tasks, 1 errored) — NEW
- Single-shot avg: 0.443, Reviewed avg: 0.521, delta = +0.079
- Major win: web_012 (0.30 → 0.95)
- Moderate wins: web_006 (0.60 → 0.70), web_010 (0.40 → 0.60)
- Regressions: web_001 (0.40 → 0.30), web_003 (0.70 → 0.60), web_015 (0.75 → 0.70)
- web_011 errored (null results)
- web_007 perfect single-shot (0.95), review correctly left unchanged

#### Research (15 tasks) — COMPLETE
- Single-shot avg: 0.676, Reviewed avg: 0.849, delta = +0.173
- Strong delta — third highest after video and code
- 8 tasks improved by review, 4 unchanged, 3 regressed slightly
- Notable wins: research_003 (0.33 → 0.80), research_007 (0.27 → 0.80), research_009 (0.47 → 0.93)
- Best single-shot: research_005 (1.0, review left unchanged at 0.93)
- Regressions: research_006 (0.87 → 0.73), research_005 (1.00 → 0.93)
- Factual verification catches specific wrong claims and enables targeted corrections
- Slowest category: ~50 API calls per task (claim extraction + per-claim verification)

## Key Findings (Updated with All Categories)

### 1. Interaction Scaling Works — Magnitude Correlates with Feedback Actionability
```
Video exec+keyframe (+0.420) >> Code execution (+0.200) >> Research factual (+0.173)
>> Web visual (+0.079) >> Animation temporal (+0.067) >> Slides visual (+0.025)
```
The ranking confirms that **execution-based and verifiable feedback** dominates:
- Video: highest delta — every single-shot fails (code errors), giving clear signal
- Code: strong — test pass/fail is unambiguous and directly actionable
- Research: strong — factual verification catches specific wrong claims, enabling targeted fixes
- Web/Animation/Slides: visual review provides softer signal, harder to convert to fixes
- Clear separation: execution+factual feedback (top 3) >> visual-only feedback (bottom 3)

### 2. Video Editing Is the Strongest Demonstration
All 15 single-shot attempts produce quality 0.00 (broken code). The review loop
recovers 7/15 tasks to quality >0.5. This is the clearest evidence that
interaction scaling creates capabilities that don't exist at all in single-shot mode.

### 3. Visual Review Has Diminishing Returns
Slides (+0.025), Animations (+0.067), Web pages (+0.079) all show modest positive
deltas. The VLM reviewer identifies issues but the proposer struggles to fix them
— visual feedback is less actionable than execution errors or factual corrections.
Notably, review sometimes regresses working animations/webpages.

### 4. Previous Results Were Partially Invalid
The `python` binary issue meant code benchmarks never actually executed tests.
The `sys.executable` fix restored correct evaluation.

### 5. Slide Regression Was a System Bug
The -0.11 regression was caused by text-only revision messages. Fixed with
multimodal revision (screenshot + text).

## Training Data Status (Post-Fix, Final)

Training pipeline fully fixed and regenerated with all 6 categories:
- **SFT**: 48 examples (14 code, 3 slide, 4 webpage, 8 animation, 6 video, 13 research)
- **GRPO**: 84 examples (15 code, 10 slide, 14 webpage, 15 animation, 15 video, 15 research)
- All SFT examples validated as containing actual code/content (not template text)
- GRPO reward functions wired to actual grounded evaluation (not placeholders)
- Multi-task GRPO training supported across all 6 modalities

## Configuration
- Model: Claude Sonnet 4 (claude-sonnet-4-20250514), temperature 0.0
- Budget: 500K tokens per problem
- Max iterations: 5 (code), 3 (visual/video/animation), 2 (research)
- Visual review: VLM via Anthropic vision API (multimodal messages)
- Video: moviepy/ffmpeg + keyframe extraction + VLM
- Code: sandboxed subprocess execution via sys.executable
