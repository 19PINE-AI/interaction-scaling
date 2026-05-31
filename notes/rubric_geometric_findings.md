# Rubric + geometric re-evaluation of visual modalities (2026-05-30)

Triggered by a reviewer observation: AI-generated SVG/architecture figures for
papers routinely have overlapping text, clipped boxes, and uneven layouts, and a
render-and-review loop should fix them. Investigating this exposed a chain of
measurement problems in the original six-modality evaluation and produced a new,
objectively-grounded result.

## 1. The original holistic VLM score is noisy and saturating
Slides were scored with a single holistic VLM `quality_score in [0,1]`. Re-scoring
the 3 on-policy runs (60 task-runs):
- Noise ~±0.4: clean slides scored 0.30; 10/60 "regressions" under keep-best
  (impossible without judge noise).
- The reported +0.06 slides lift was mostly noise.

## 2. A binary per-requirement rubric de-noises but reveals task saturation
`src/evaluation/checklist_judge.py`: each task's requirements list is the rubric;
the judge returns per-requirement `satisfied:true|false` (structured JSON);
Python averages `n_met/n_total`. Multi-axis, binary, externally averaged.

Re-scored under the rubric (single-shot vs reviewed, 3 seeds):
| modality   | rubric single-shot | saturated? | harness lift | sign-test p |
|------------|--------------------|-----------|--------------|-------------|
| web        | 0.469              | no (3/45) | +0.063       | 0.003       |
| animations | 0.642              | no (4/32) | +0.192       | 0.008       |
| diagrams   | ~0.95              | mostly    | small        | --          |
| slides     | 0.948              | yes (46/60)| ~0          | 1.0         |
| research   | 0.945              | yes (39/45)| -0.062       | 0.065       |

The rubric SHARPENS real effects (web: p 0.020 -> 0.003 by cutting spurious
negatives 9 -> 4) and DEFLATES illusory ones (slides). Saturated suites
(slides, research, most diagrams) were hardened; web-verified harder research
facts in `research_tasks_hard.json`, denser `slide_tasks_hard.json`.

## 3. Even a binary VLM rubric is UNRELIABLE for geometric defects
New modality: 15 academic-paper architecture figures (Transformer, U-Net, ViT,
FPN, DETR, Mask R-CNN, latent-diffusion U-Net, CLIP, MoE, GAT, WaveNet, SE,
Perceiver-IO, Seq2Seq+attn, Evoformer) as self-contained HTML+SVG
(`data/hard_benchmarks/diagrams/paper_figure_tasks.json`).

The VLM rubric (even with native-resolution quadrant tiling) scored **14/15
single-shot figures perfect** — yet by eye the Evoformer figure has "MSA
Processing"/"MSA Representation" titles superimposed, and the ViT detail panel
has "Class Output" overlapping "Input". KEY MECHANISM: screenshots are captured
at 1080px, so any content overflowing the canvas is cropped OFF the image and
the VLM never sees it.

## 4. Deterministic DOM geometry is the reliable instrument
`src/evaluation/geometric_checker.py` loads the HTML in Playwright and, via
`page.evaluate`, extracts every element's bounding box to compute EXACTLY:
- text_overlap: non-nested text elements whose boxes intersect (>=6px)
- out_of_bounds: content leaves (text/SVG primitives) clipped at the viewport
- overflow: text past its container box
- scrollbar: document exceeds 1920x1080 by >16px (filters default margins)

On the 15 paper figures, single-shot: **only 3/15 are geometrically clean**
(mean 1.6 defects) vs the VLM's 14/15 "perfect". The VLM is blind to the very
defects that matter.

## 5. Objective interaction-scaling result (the payoff)
`scripts/run_geometric_harness.py`: proposer -> DOM geometric_defects() -> the
EXACT defects fed back ("text X overlaps text Y; move them apart"; "canvas is
40px too tall") -> revise; reward = defect count (deterministic, no VLM).

Paper figures (15, single-shot vs reviewed, max 3 iters):
- mean defects **1.20 -> 0.27 (-78%)**
- geometrically clean **4/15 -> 11/15**
- total defects 18 -> 4; 8 improved, **0 regressed**, sign-test **p=0.0078**

Cross-check on the ORIGINAL slides harness (VLM-feedback), scored objectively by
DOM geometry (60 task-runs): mean defects **2.63 -> 0.23 (-91%)**. The slides
harness was fixing real layout defects all along; the holistic/rubric VLM score
could not measure it. (Webpages: the fixed-canvas geometry metric does not apply
— pages are scrollable; text-overlap-only is noisy on dense webpage DOM, so the
binary VLM rubric remains the practical metric there.)

## Takeaways for the paper
1. Add a "diagram / academic figure" Type-3b modality.
2. Report it with a DETERMINISTIC DOM-geometry reward, not a VLM score.
3. State plainly: VLM-screenshot judging systematically misses geometric defects
   (overflow cropped off-frame; small overlaps below VLM acuity) — a measurement
   caveat that applies to the original visual-modality numbers.
4. The honest, well-scoped positive: on fixed-canvas visual artifacts (slides,
   figures), grounded geometric feedback reduces real layout defects 78-91%
   with zero regressions — interaction scaling on an objective metric.
5. Scope geometric scoring to fixed-canvas artifacts; keep the binary rubric for
   web/animations (with the VLM-reliability caveat).
