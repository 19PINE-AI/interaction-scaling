# Human-preference study kit

A blind, randomized pairwise study testing whether the grounded-geometry defect
reductions (Table 2 in the paper) correspond to **human-perceived** layout quality
— closing the circularity caveat that the DOM instrument both drives and scores the
loop.

## Quick start

```bash
# 1. Build stimuli + blind manifest (re-renders single-shot vs reviewed artifacts
#    full-page from the saved HTML; ~25 min, mostly animation frame waits).
python3 scripts/build_human_study.py                 # all four modalities
python3 scripts/build_human_study.py --modalities figures slides   # subset, fast

# 2. Run the study (raters).
cd study && python3 -m http.server 8000   # open http://localhost:8000/
#    or just open study/index.html directly if your browser allows file:// access.

# 3. Collect each rater's responses_<id>.csv, then analyze.
python3 scripts/analyze_human_study.py --responses responses_*.csv
```

The stimuli, `manifest.js`, and `key.json` are **generated artifacts** (git-ignored);
rebuild them with step 1. The committed source of truth is the build script, the
rating app (`index.html`), the analyzer, and **`PROTOCOL.md`** — read that for the
full design, endpoints, and how to report the result in the paper.
