# Closing the rubric program: hardened slides + the video modality (2026-05-31)

Two gaps remained in the rubric re-evaluation: the hardened **slides** suite was
built but never re-run, and **video** had never been brought into the rubric
program at all (only a holistic 0-1 score). Both are now closed, plus the video
suite was hardened in response to saturation.

## 1. Hardened slides SATURATE even after hardening
`slide_tasks_hard.json` (20 tasks, 10 binary reqs each, dense multi-region
layouts) re-run single-shot vs reviewed (3 seeds, 60 task-runs), scored two ways:

- **Binary hires rubric** (`checklist_score`): SS **0.955**, RV 0.967,
  lift +0.012, sign-test **p=0.58**, **45/60 single-shot perfect** -> still
  SATURATED. A frontier proposer (Sonnet) satisfies almost every requirement
  single-shot even on the hardened suite; the VLM rubric has no headroom.
- **DOM geometry on the VLM-feedback harness artifacts** (`geometric_checker`):
  SS mean 2.93 defects -> RV 0.35 (total 176 -> 21, -88% on the mean), BUT
  the reduction is carried by **3 catastrophic-overflow single-shots** (81, 51,
  30 clipped elements) that review happened to fix; meanwhile VLM-driven review
  **introduced** defects in 6 previously-clean slides (0->2,3,5,...). Net clean
  count flat (50/60 -> 50/60), per-slide sign-test n.s. (7 up / 6 down).
  This is the paper's thesis in miniature: VLM review is blind to off-canvas
  overflow, so it stumbles into fixing catastrophes while creating new defects.
- **DOM geometry on the geometric-FEEDBACK harness** (`run_geometric_harness`,
  the deterministic loop): mean 0.50 -> 0.05 defects (-90%), clean 16/20 ->
  19/20, 4 fixed / 1 introduced, p=0.375 (n.s. at N=20 with a low single-shot
  base rate). Cleaner than VLM feedback (4/1 vs 7/6 fix/introduce), consistent
  with the headline academic-figure result (-78%, p=0.008) where single-shot
  genuinely fails.

**Takeaway:** slides saturate for a frontier model even after hardening; the
measurable interaction-scaling headroom on fixed-canvas artifacts lives in the
dense academic figures, not slides. Confirms the earlier saturation finding.

## 2. Video = programmatic video EDITING (not generation)
The model writes a Python (ffmpeg/moviepy) script that transforms a source clip
to `/tmp/output.mp4`; the harness executes it and scores the result. Three
synthetic 1280x720 / 25fps / 5.0s sources (colors: red 0-2s, green 2-4s, blue
4-5s; countdown 5->1; gradient).

### Native full-video rubric (Gemini 3.1 Pro)
Replaced keyframe sampling with **Gemini 3.1 Pro native video understanding**
(`src/evaluation/gemini_video_judge.py`, Files API: upload whole mp4 -> judge
each requirement against full playback). This fixed the keyframe caveat: the
frame-sampler could not verify exact duration / codec validity / motion
smoothness and wrongly failed those reqs (e.g. video_001 0.75 -> 1.0 under the
native judge). The holistic score was wildly unreliable for video: mean
|holistic - rubric| = **0.575** (worse than slides' +-0.4); the old "+0.34"
video lift was largely holistic noise.

### Original video suite is SATURATED
SS **0.704**, RV 0.740, lift +0.036, **p=0.61**, **22/45 single-shot perfect**.
Single-shot editing is mostly already correct; the few real wins are
execution-failure recoveries (a script that produced no mp4), roughly balanced
by regressions.

### Hardened video suite (`video_tasks_hard.json`, 15 tasks)
Built to de-saturate: multi-step pipelines (speed+reverse+loop+concat with exact
color-at-time and duration checks), frame-accurate trims, PTS/timestamp traps
(reverse-then-retime, boomerang, freeze-frame at an exact digit), speed ramps,
2x2 grids with a reversed quadrant, time-mirrored split-screen, per-frame
rotated watermark, true-grayscale-in-yuv420p, crossfade chains. 9-10 observable
binary requirements each (color, order, blend, legibility, motion-not-frozen,
position) so the native video judge can verify them.

Result (Gemini native rubric, 3 seeds, 45 task-runs):
- **single-shot fully-correct collapsed 22/45 (49%) -> 5/45 (11%)** -> DE-SATURATED
- mean SS 0.706 (similar, because the rubric averages 9-10 reqs and the model
  gets most-but-not-all), median 0.80, per-task SS spread **0.44 -> 0.90**
- hardest: vhard_014 (frame-accurate trim+zoom 0.44), vhard_008 (speed ramp
  0.52), vhard_005 (2x2 grid 0.53), vhard_011 (strobe 0.56), vhard_012
  (mirrored split 0.60); still-easy: vhard_001/013/010/004 (~0.88-0.90)
- harness lift **+0.047, p=0.30** (10 up / 5 down) -- trends positive, larger
  than the saturated suite, not significant at single-seed N=45 (more seeds
  would be needed to call it).

**Takeaway:** the original video suite was saturated (49% perfect single-shot);
the hardened suite measures with real headroom (11% perfect, 0.44-0.90 spread).
Video editing single-shot remains strong, so the harness lift is small/n.s. --
consistent with the paper's reframing that interaction-scaling value
concentrates where single-shot genuinely fails (web; dense figures).

## 3. The original video headline was a missing-library artifact (paper reframed)
The paper's headline video result -- "+0.47 quality, 11/15 single-shot produce
0.0 because the script crashes, the strongest evidence for a distinct axis" --
turned out to be an **environment artifact**. The on-policy data that produced
it (SS holistic 0.12-0.14, 11/15 zeros, RV 0.51-0.66) is the same data I
rescored. Cross-tabulating the 34 single-shot task-runs the holistic judge
scored 0.0:
- 30 of 34 **execute fine** when re-run (only 4 genuinely fail to produce mp4);
- those "zeros" score mean **0.65** under the Gemini native rubric;
- **28 of the 30** are moviepy scripts.

Mechanism: `moviepy` was not installed in the generation environment (the
`pyproject.toml` dep was never `pip install`-ed there), so single-shot scripts
that imported moviepy crashed -> holistic 0.0; the reviewer saw the crash and
routed the proposer to `ffmpeg`, which worked -> the "+0.47 recovery." With
moviepy installed (this session), the same single-shot scripts run and score
~0.65; under proper measurement the modality is saturated (+0.036, p=0.61).

**Fix applied:** moviepy installed (pyproject already pinned `>=2.2.1`; the gap
was that deps were never installed in the run env -- `pip install -e .` fixes
it going forward).

**Paper reframe (done, compiles 30pp, 0 undefined refs):** removed video from
the abstract/intro/discussion execution-grounded headline; corrected the
`tab:harness-headline` video row to the native-rubric numbers (0.70->0.74,
+0.04, p=0.64, un-bolded); repurposed the "Capability emergence" case study from
video to **code** (genuine 66.7->100%, pytest-traceback mechanism); rested the
load-bearing tier claim on code + the in-modality feedback-type control
(tab:type-controls) rather than the cross-modality code+video pair; removed the
video bar from the Phase-1 actionability figure; added a short, de-emphasized
`sec:harness-video` paragraph (native rubric + hardened suite). Per user
direction: highlight strong results (code, web, animations, figures), do not
feature the now-null video lift.

## Artifacts
- `data/hard_benchmarks/video/video_tasks_hard.json` (new, 15 hard tasks)
- `src/evaluation/gemini_video_judge.py` (Gemini 3.1 Pro native video rubric)
- `scripts/rescore_video_rubric.py` (parameterized: --tasks/--run-prefix/--out)
- `scripts/rescore_slides_hard.py`, `scripts/analyze_rubric_rescore.py`
- results: `video_rubric_rescore.json` (+ `_frames.json` keyframe baseline),
  `video_hard_rubric_rescore.json`, `slides_hard_rubric_rescore.json`,
  `slides_hard_geom_run1.json`
