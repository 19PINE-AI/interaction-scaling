# Finding 10: Hard Benchmark Summary — Interaction Scaling Value

## Comprehensive Results Across 3 Modalities

### Code Generation (15 tasks, execution feedback)
| Approach | Pass Rate | Avg Tokens |
|----------|-----------|------------|
| Single-shot | **60%** (9/15) | ~480 |
| With execution feedback | **93%** (14/15) | ~2,500 |
| **Improvement** | **+33pp** | 5x tokens |

**Best result.** Execution feedback provides clear, actionable error messages.
5 tasks fixed by seeing test failures. Only 1 task unfixable.

### Slide Generation (5 tasks, visual VLM review)
| Approach | Avg Quality | Meets Requirements |
|----------|------------|-------------------|
| Single-shot | **0.76** | **60%** |
| With visual review | **0.85** | **60%** |
| **Improvement** | **+0.09** | +0pp |

**Moderate result.** Visual review improved quality score on 2/5 tasks
(slide_002: 0.80→0.95, slide_005: 0.30→0.60). The challenging tasks
(slide_001, slide_005) remain below threshold even after review.

### Animation Generation (8 tasks, multi-frame VLM review)
| Approach | Avg Quality | Meets Requirements |
|----------|------------|-------------------|
| Single-shot | **0.29** | **25%** |
| With frame review | **0.37** | **12%** |
| **Improvement** | **+0.07** | -13pp |

**Weakest result.** Animations are genuinely very hard. Multi-frame VLM
feedback provides insufficient guidance for fixing complex timing and
physics bugs. Click-triggered animations (anim_003, 004) produce blank
frames in automated capture.

## Cross-Modal Analysis

### Where Grounded Feedback Works Best
1. **Code + execution** (+33pp): Error messages are precise, point to
   specific line/assertion, model can directly fix the bug
2. **Slides + visual** (+0.09 quality): VLM identifies overflow/overlap
   but fixing requires CSS expertise; partial improvements
3. **Animations + frames** (+0.07 quality): Frames show issues but fixing
   multi-step timing/physics requires deep algorithmic changes

### Feedback Actionability Spectrum
```
Most actionable                              Least actionable
|--------------------------------------------------|
Code execution     Slide visual     Animation frames
(precise errors)   (spatial issues)  (temporal issues)
+33pp              +0.09 quality     +0.07 quality
```

### Key Insight
The value of interaction scaling correlates with **feedback actionability**:
- Execution errors point to exact failures → easy to fix → large gains
- Visual screenshots show spatial problems → moderate fix difficulty
- Animation frames show temporal problems → hard to diagnose/fix from frames alone

## Implications for the Paper
1. **Code generation is the strongest demonstration** of interaction scaling
2. **Visual feedback provides genuine but moderate value** — larger
   improvements likely with more iterations or better VLM prompting
3. **Animation feedback needs work** — current frame-based approach is
   insufficient; may need more structured temporal analysis
4. **The feedback taxonomy predicts the results**: Type 3a (execution) >
   Type 3b (visual) > Type 3c (temporal) in terms of actionability
