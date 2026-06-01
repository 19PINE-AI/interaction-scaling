# Real-paper dense slides, alignment instrument, design-principle pipeline (2026-06-01)

Hardened the visual modalities the way real decks fail (dense text + figures,
aligned pillars), grounded in real papers; baked the four graphic-design
principles into the whole pipeline; and re-ran everything apples-to-apples.

## 1. Design principles in the pipeline (generation + feedback + judge)
The four core principles (Robin Williams: Proximity, Alignment, Repetition,
Contrast, + deliberate color) are now applied identically in: the slide/web
GENERATION prompt (`SLIDE_SYSTEM_PROMPT`, `WEBPAGE_SYSTEM_PROMPT`), the visual
REVIEWER feedback prompt (`VISUAL_REVIEW_PROMPT`), and the rubric JUDGE
(`CHECKLIST_SYSTEM`). Measured effect: the design-principle generation prompt
visibly improves single-shot layout quality (fewer geometric defects), which is
why suites had to be hardened to keep headroom -- per the rule "if single-shot
is too good, harden the tasks, do not weaken the evaluation."

## 2. Deterministic ALIGNMENT detection (new instrument axis)
`geometric_checker` previously caught overlap/overflow/clipping/scrollbar but
NOT misalignment (unequal-width pillars, misaligned edges, uneven gutters) --
exactly the "aligned boxes" failure. Added a deterministic detector: collect
bordered/filled HTML boxes, keep outermost cards, cluster into rows (shared top)
and columns (shared left), and within groups of >=3 flag unequal size,
misaligned far edges, and uneven gaps. Opt-in via `include_alignment` so the
already-published figure/slide geometry numbers are unchanged. Validated: a
deliberately-misaligned row scores 2 defects, an aligned row scores 0.

## 3. Real-paper dense-slide modality (`slide_tasks_hard2.json`, 12 tasks)
Each task PROVIDES real content (Transformer, ResNet, BERT, AlexNet, GAN,
word2vec, U-Net, Adam, Dropout, VGG, LSTM, DDPM) -- equations, tables, the
figure to draw -- and demands dense text+figure coexistence with aligned
equal-width pillars/panels. Hardened twice (after the design-principle prompt
made single-shot too clean): added an extra dense real region per task
(positional-encoding formula, ResNet bottleneck block, BERT token-embedding row,
GAN 3x3 sample grid, Adam convergence chart, VGG 19+ rows, LSTM unrolled chain,
DDPM noise schedule, ...).

**The instrument split is the headline finding:**
- VLM rubric: SATURATED (SS 0.97, 29/36 perfect). A screenshot judge rates dense
  slides near-perfect -- it cannot see the layout defects.
- Deterministic geometry (with alignment): NOT saturated -- single-shot clean
  only 16/36 (44%), mean 1.89 defects (text_overlap + misalignment dominate).
- VLM-feedback review makes geometry WORSE (1.89 -> 2.44 mean; clean 16 -> 13):
  the reviewer is blind to layout, so its edits break alignment as often as not.

**Geometric-FEEDBACK harness (deterministic reward + feedback, alignment-aware,
design-principle slide prompt), 3 seeds, n=36 task-runs -- the positive result:**
- mean defects **1.25 -> 0.33 (-73%)**, total 45 -> 12
- geometrically clean **18/36 (50%) -> 29/36 (81%)**
- **13 improved / 1 regressed / 22 tied**, two-sided sign-test **p = 0.0018**

Same shape as the academic-figure result (-78%, p=0.008) and original slides
(-91%), now on real-paper dense slides and with alignment folded in: grounded
geometric feedback reliably removes real layout defects; VLM feedback cannot.

## 4. Hard web -- same redesign as slides (`webpage_tasks_hard.json`, 20 tasks)
20 dense, real-grounded, responsive pages (dashboard, cloud pricing w/ 12-row
comparison table, 3-col docs, multi-track schedule, GitHub repo, status page,
PDP, newspaper grid, settings, arXiv results, kanban, email client, month
calendar, chat, invoice, social profile, checkout, phone spec-sheet, masonry
gallery, long SaaS landing). 10-12 precise requirements each (content +
alignment + responsive). Expanded from 10 -> 20 for representativeness/parity
with the other modalities.

**VLM rubric (new design-aware pipeline):** SS 0.667, RV 0.721, lift +0.054
(p=0.30 at n=30), 0/30 perfect -- positive direction (matches old +0.063) but
NOT significant. The rubric undersells the effect, exactly as on slides.

**Deterministic MULTI-WIDTH geometry instrument (the web analog of the slide
result).** Web is scrollable, so `geometric_checker(web=True)` ignores vertical
overflow/below-fold clipping and scores only text-overlap + container-overflow +
misalignment + HORIZONTAL overflow; `web_geometric_defects` sums this at desktop
(1920) AND mobile (375) -- the responsive failure (grids breaking / horizontal
scroll on mobile) is the dominant single-shot defect. Web geometric-feedback
harness (`run_geometric_harness --web --prompt web --include-alignment`),
**20 tasks x 3 seeds, n=60**:
- mean defects **16.1 -> 8.5 (-47%)**, total 965 -> 510
- geometrically clean **10/60 -> 16/60**
- **33 improved / 2 regressed / 25 tied**, two-sided sign-test **p = 3.7e-08**

Dense responsive pages are defect-RICH single-shot (~16 layout defects/page
across the two widths) -- unlike slides (near-clean). The harness cannot fully
clean them in 3 iters but removes ~half, with 33:2 odds. Same lesson as slides:
deterministic geometry reveals a large, highly-significant interaction-scaling
effect that the VLM rubric (+0.054, n.s.) is blind to.

## 5. Apples-to-apples regeneration under the new pipeline (old prompt -> new)
All slide/web suites regenerated with the design-principle prompts + judge; old
artifacts preserved in `backup/oldprompt_results/`.
- Web rubric lift: +0.063 (p=0.003, old) -> +0.034 (n.s., new). NOTE: not a clean
  single-variable change -- generation prompt AND judge both changed (stricter,
  design-aware). The hardened web suite above is the cleaner remeasurement.
- Slides (orig & hard): VLM rubric stays saturated; the design-principle prompt
  reduced single-shot geometric defects (slides_hard 2.93 -> 1.53 mean).

## Takeaways for the paper
- Add a real-paper dense-slide result to the geometry section: VLM judge AND VLM
  reviewer are blind to slide layout; deterministic geometry (incl. alignment) is
  the instrument, and geometric feedback gives -73% defects, p=0.0018, 13:1.
- Alignment is now a first-class deterministic defect axis.
- The design principles are part of the standard generation/feedback/judge
  pipeline; headroom is created by task difficulty, never by weakening the judge.

Artifacts: `data/hard_benchmarks/slides/slide_tasks_hard2.json`,
`data/hard_benchmarks/webpages/webpage_tasks_hard.json`,
`src/evaluation/geometric_checker.py` (alignment), `scripts/run_geometric_harness.py`
(`--prompt slide --include-alignment`), results `slides_hard2_geom_run{1,2,3}.json`,
`slides_hard2_rubric_rescore.json`, `web_hard_rubric_rescore.json`.
