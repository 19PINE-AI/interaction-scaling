# Interaction Scaling — Visual Generations Gallery

Single-page React (Vite) companion site for the paper. It shows every visual
test case (academic figures, slides, web pages, animations) as a **single-shot
vs. reviewed** pair, with per-case scores and a side-by-side / draggable-slider
comparison.

## Run

```bash
cd website
npm install        # once
npm run dev        # dev server (hot reload)
# or
npm run build && npm run preview   # production build + static preview
```

Open the printed `localhost` URL.

## Contents

- `src/App.jsx` — the SPA (hero + summary stats, category tabs, card grid, comparison modal with side-by-side and slider modes).
- `src/styles.css` — styling.
- `public/data/manifest.json` — all 65 test cases with metadata (category, task description, scores, image paths).
- `public/images/<category>/<task>_{ss,rv}.png` — rendered single-shot / reviewed artifacts.

## Regenerating the data

From the repo root, after the experiments have produced
`results/hard_benchmarks/*.json`:

```bash
python -m scripts.export_gallery     # writes public/data/manifest.json + public/images/
```

## Metrics shown

- **Academic figures, Slides** — deterministic DOM-geometry defect count (lower is better; 0 = clean).
- **Web pages, Animations** — binary per-requirement rubric (fraction satisfied).

Summary lifts: figures −78 % defects (p=0.008), slides −91 %, web +0.063 (p=0.003), animations +0.192 (p=0.008).
