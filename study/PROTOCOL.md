# Human-preference study: does grounded geometric feedback produce *human-visible* quality gains?

## Why this study exists

The paper's grounded-geometry result has a built-in circularity: the deterministic
DOM instrument both **drives** the revision and **scores** it, and the harness keeps
the best-scoring iteration, so some defect reduction is mechanically guaranteed (see
the "Is the reduction circular?" paragraph in `sec_grounded_eval.tex`). Three
in-paper checks argue the effect is real (real single-shot defect rate; non-trivial
magnitude; the ungrounded-reviewer control that moves the *same* metric the wrong
way). What none of them establish is the final link: that a reduction in
DOM-measured defects corresponds to a layout a **human** judges as better.

This study closes that link with real raters. It is deliberately **blind** (raters
never see which side is single-shot vs. reviewed), **randomized** (side and order),
and **instrument-independent** (the analysis joins responses to a hidden key the
rating app never loads).

## Stimuli

For every (task, seed) in the four visual modalities backing Table 2
(`tab:geometric`) we render the single-shot and the reviewed artifact and present
them side by side as **A** vs **B**:

| Modality | n tasks × 3 seeds | how it is shown |
|---|---|---|
| Academic figures | 20 × 3 | one full-page capture (1920×1080, **full-page** so off-canvas overflow is visible) |
| Dense slides | 12 × 3 | one full-page capture |
| Web pages | 20 × 3 | two captures stacked: desktop (1920) + mobile (375), both full-page |
| Animations | 20 × 3 | horizontal strip of the scored frame times (left = earliest) |

**Faithfulness.** Artifacts are re-rendered from the exact final HTML that was
scored, at the same 1920×1080 design target the instrument used. The one
deliberate change from the scored screenshot is `full_page=True`: the scored
capture cropped overflow off-frame (the very blindness the paper attributes to the
VLM), so handing raters that same crop would rig the task toward "no difference."
Full-page rendering *reveals* the defect and makes the human task fair.

Pairs are labelled **decisive** (DOM scored the two sides differently) or **tie**
(DOM scored them equal). Tie pairs are retained as a control.

## Rater task

Open `study/index.html` (see "Running it" below), enter a short rater ID, and for
each pair choose one of:

- **Left is cleaner** (key `1`)
- **No visible difference** (key `2`)
- **Right is cleaner** (key `3`)

Instructions shown to raters: *judge only layout quality — overlapping text,
content cut off or spilling out of its box, misaligned elements. Ignore the topic,
wording, and color taste.* Click any image to open it full size. Progress is saved
in the browser; raters may stop and resume. At the end (or any time) the rater
clicks **Download results CSV** and returns the file.

Order is reshuffled per rater (seeded by rater ID); A/B side is randomized per pair
at build time, so position bias averages out.

## Recommended design

- **Raters:** ≥ 5 independent people, none involved in the project. Crowd platforms
  (Prolific/MTurk) or colleagues both work.
- **Coverage:** the full set is ~288 pairs (96 decisive + ~192 ties). Either have
  each rater do the whole set, or assign overlapping random blocks of ~80 pairs so
  every decisive pair gets ≥ 3 independent votes and some pairs are shared for
  inter-rater agreement.
- **Attention checks:** the high-defect decisive pairs (e.g. web single-shot with
  10+ defects) function as natural catch trials; a rater who calls those "no
  difference" or prefers the broken side throughout can be flagged.
- **Pre-registration (optional but recommended):** fix the primary endpoint and
  threshold below *before* collecting data.

## Endpoints and analysis

Run:

```bash
python3 scripts/analyze_human_study.py --responses responses_*.csv
```

It reports, joining each vote to `study/key.json`:

1. **PRIMARY — preference for the reviewed render among decisive pairs.** Of the
   decisive votes that expressed a preference, the fraction that picked the
   lower-defect (reviewed) side, overall and per modality, with a two-sided
   binomial sign test vs. 50% and a Wilson 95% CI. **The caveat is closed if this
   is significantly > 50%** (pre-registered threshold: lower CI bound > 50%).
2. **DOM concordance.** The same number read as agreement between human preference
   and the instrument's direction — the headline sentence for the paper.
3. **Tie control.** On DOM-equal pairs, human preference should be ≈ 50/50; a large
   skew would indicate the instrument misses a real quality axis (reported honestly
   either way).
4. **"No visible difference" rate** per modality (how often the defect is
   sub-perceptual even when present).
5. **Inter-rater agreement** on overlapping pairs.

## Running it

The app loads `manifest.js` via a `<script>` tag and images via relative paths, so
it works from `file://` with no server. If your browser restricts local file
access, serve the folder:

```bash
cd study && python3 -m http.server 8000   # then open http://localhost:8000/
```

## What to write in the paper

Replace the open caveat ("a quantitative human-preference study remains future
work") with the measured concordance, e.g.: *"In a blind, randomized study, N
raters preferred the reviewed render on X% (95% CI […]) of decisive pairs,
agreeing with the deterministic instrument and confirming the defect reductions are
human-visible; on DOM-equal pairs preference was Y% (no systematic skew)."* If the
result is null or weak, report that instead — the point of grounding is to measure
honestly.

## Files

- `index.html` — the rating app (self-contained).
- `manifest.js` — `window.MANIFEST`, the blind pair list (no ground truth).
- `key.json` — hidden condition/defect key (analysis only; do **not** put on the rating machine if you want to be strict about blinding).
- `stimuli/<modality>/…png` — rendered stimuli.
- built by `scripts/build_human_study.py`; analysed by `scripts/analyze_human_study.py`.
