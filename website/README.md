# Grounding the Loop — interactive companion site

Single-page React (Vite) explainer for the paper, aimed at first-time readers.
Three sections:

1. **The idea** — a jargon-free walkthrough of the three test-time compute axes
   (reasoning, sampling, interaction), the internal ceiling, and the
   grounding-on-both-sides claim, ending with a real before/after case.
2. **Results** — eight figure cards built from the real experiment logs:
   scaling curves with seed spread, VLM-blindness, the reviewer-swap ablation,
   per-modality defect reductions with bootstrap CIs, cross-model replication,
   budget allocation, and distillation. Every chart has hover tooltips and most
   have a table view.
3. **Explorer / trajectory visualizer** — all 98 exported cases (figures,
   slides, web pages, animations, code) with the task spec, single-shot vs.
   reviewed comparison slider, per-seed instrument measurements, the full
   interaction trajectory (draft → instrument feedback → revision, including
   the model's raw output and thinking), the 4-strategy × 3-budget pass matrix
   for dev code tasks, cross-model final code, and a catalog of all 9 task
   suites (162 tasks) including the scoped video/research suites.

## Run

```bash
cd website
npm install
npm run dev            # dev server
npm run build && npm run preview   # production build + static preview
```

## Regenerating the data

From the repo root, after experiments have produced `results/**`:

```bash
python -m scripts.build_site_v2_data
```

This writes:

- `public/data/v2/site.json` — chart aggregates (computed from raw result JSON
  where available in-repo; a few headline stats transcribed from the paper are
  marked `"source": "paper"`) plus the explorer index and task catalog.
- `public/data/v2/cases/<modality>/<task_id>.json` — per-case dossiers
  (task spec, per-seed geometry measurements, final artifacts, trimmed
  interaction traces). Loaded lazily by the case view.

Before/after JPEGs in `public/images/` were exported from the same result
files by `scripts/build_site_data.py` (v1 pipeline) and are reused.

## Structure

- `src/App.jsx` — nav, hero, hash router (`#/case/<mod>/<id>` opens a dossier)
- `src/Idea.jsx` / `src/Results.jsx` / `src/Explorer.jsx` / `src/CaseView.jsx`
- `src/charts.jsx` — hand-rolled SVG charts (tooltips, seed whiskers, table twins)
- `src/bits.jsx` — loop diagram, compare slider, trace step, stat tiles
- `src/theme.css` — the design system (paper surface, Fraunces + IBM Plex,
  blue = grounded/external, orange = internal/ungrounded)
