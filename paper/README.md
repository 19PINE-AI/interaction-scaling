# Paper source

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or `latexmk -pdf main.tex`.

## Files

- `main.tex` — single-file source (~26 pages compiled)
- `refs.bib` — 30 verified entries
- `arxiv.sty` — minimal arXiv-preprint style
- `main.pdf` — built output
- `_archive/` — superseded drafts (`main_distillation_version.tex`, `section4_harness.tex`) kept for provenance

## Title

**Interaction Scaling: A Third Test-Time Compute Axis Grounded in Environment Feedback**

Author: Bojie Li (Pine AI, `boj@19pine.ai`)

## Headline numbers

| Setup | Result |
|---|---|
| Code harness (Sonnet 4, 3-run mean) | 66.7 ± 6.7% → **100.0 ± 0.0%** (+33.3 pp) |
| Video editing harness (Sonnet 4, 3-run mean) | 0.13 ± 0.01 → **0.60 ± 0.07** (+0.47) |
| Web pages harness | 0.43 ± 0.03 → 0.56 ± 0.01 (+0.13) |
| Animations harness | 0.40 ± 0.03 → 0.55 ± 0.01 (+0.15) |
| Slides harness | 0.73 ± 0.01 → 0.79 ± 0.01 (+0.06) |
| Deep research harness | 0.63 ± 0.06 → 0.69 ± 0.06 (+0.06) |
| Reasoning-only at matched budget (code) | 73.3% → 86.7% (closes 2/3 of harness gap) |
| In-modality feedback-type controls (code) | Type 1 = Type 2 = 86.7%, Type 3a = 93.3% |
| **4-strategy scaling on hard code (B=20K, 3-seed mean)** | **R 73.3% / S 86.7% / L 97.8 ± 3.1pp / H 100.0 ± 0.0pp** |
| Among interaction strategies (B=20K) | H, L, IAD all reach ~100%; H wins on tokens (1,029 vs 1,431 L vs 1,416 IAD) |
| Cross-model (3 families) | Sonnet +33.3 pp / Qwen3-235B +26.7 pp / GPT-5 +13.3 pp (higher SS baseline) |
| Held-out (32 code tasks, zero overlap) | 90.6% → 100.0% (3/3 SS-fails fixed, 0 regressions) |
| Budget allocation simplex (B=10K) | 86.6 pp spread; propose-heavy (b1≥0.50) wins at 93.3% |
| Token ROI (code vs visual) | 37 vs 4,000–5,000 tokens per 0.01 quality (100× spread) |
| Markdown 8B SFT student (hard held-out, 18 tasks) | 44% pass@1, 56% pass@3 (0.50× teacher SS capability) |
| Markdown 8B student on 44 OOD gen tasks | 9/44 pass@1, 14/44 pass@2 (0.70× teacher SS) |
| RFT vs SFT pass@3 trade-off | RFT polishes consistency but regresses pass@3 by 17 pp |
| Cost per kept task (8B student) | ~10× cheaper than harness over 235B teacher |

## Outline

| § | Section |
|---|---|
| 1 | Introduction |
| 2 | Related Work |
| 3 | The Grounded Feedback Framework (taxonomy + DPI argument + predictions) |
| 4 | Method: The Proposer-Reviewer Harness |
| 5 | Interaction Scaling with External Scaffolding (6 modalities + Type-1/2/3a controls + reasoning baseline + 4-curve scaling + architectural separation + cross-model + held-out + allocation + ROI + early stopping) |
| 6 | Internalizing the Harness into a Small Student (8B markdown distillation + 3 recipe findings) |
| 7 | Discussion |
| 8 | Conclusion |
| App A | Phase 1 baseline (feedback actionability bar chart) |
| App B | Pipeline implementation |
| App C | Hyperparameters and SFT details |
| App D | Per-task pass@k breakdown for the 8B markdown student |

## Status

Reframed to lead with the interaction-scaling thesis (third test-time-compute axis). The §5.6 four-strategy scaling table (R/S/L/H) is the headline figure, now with 3-seed mean ± SD; the H-vs-L pass-rate gap is within seed noise (sign-test p>0.6) and §5.7 reframes the architectural separation as token efficiency + zero seed variance, not pass-rate ceiling. §5.4.5 adds in-modality Type-1/2/3a controls; §5.8 cross-model adds GPT-5 (third family); §5.7 adds an IAD baseline row. Distillation positives (§6) are retained; agentic regression and the "format > capacity" framing are dropped. Negative post-training results are not in the paper.
