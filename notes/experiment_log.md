# Experiment Log — Hard Benchmarks (Final)

## All Results Across 6 Categories

### Summary Table

| Category | Tasks | Feedback Type | Single-shot | Reviewed | Delta |
|----------|-------|---------------|-------------|----------|-------|
| **Code (SWE)** | 15 | Type 3a: Execution | 67% pass | **100%** pass | **+33pp** |
| **Research** | 5 | Type 3d: Factual | 0.76 accuracy | **0.92** accuracy | **+0.16** |
| **Web pages** | 2* | Type 3b: Visual | 0.57 quality | **0.72** quality | **+0.15** |
| **Slides** | 5 | Type 3b: Visual | 0.76 quality | **0.85** quality | **+0.09** |
| **Animations** | 8 | Type 3c: Temporal | 0.29 quality | **0.37** quality | **+0.07** |
| **Video editing** | 15 | Type 3c: Temporal | — | — | *not run* |

*Web pages: partial results due to API rate limiting

### Detailed Results

#### 1. Code — SWE-style Bug Fixing (15 tasks)
- **Single-shot: 67%** (10/15 bugs fixed correctly)
- **With execution feedback: 100%** (15/15, all fixed in ≤2 iterations)
- Fixed bugs: CSV parser, date ranges, markdown parser, CJK text, wildcard trie

#### 2. Deep Research (5 tasks)
- **Single-shot: 0.76** accuracy (model confabulates numbers/dates)
- **With fact-checking: 0.92** accuracy (4/5 tasks improved to 1.00)
- Unfixed: GDP rankings (too many interconnected facts to verify)

#### 3. Web Pages (2 tasks, partial)
- **Single-shot: 0.57** quality
- **With visual review: 0.72** quality
- web_001 (product landing page): 0.40 → 0.70 with review

#### 4. Slides (5 tasks)
- **Single-shot: 0.76** quality
- **With visual review: 0.85** quality
- slide_002 improved 0.80→0.95, slide_005 improved 0.30→0.60

#### 5. Animations (8 tasks)
- **Single-shot: 0.29** quality
- **With frame review: 0.37** quality
- Animation bugs are hardest to fix from frame screenshots alone

### Feedback Actionability Hierarchy (Validated)

```
Type 3a (execution)  >> Type 3d (factual)  >  Type 3b (visual)  >  Type 3c (temporal)
      +33pp                 +0.16                 +0.09-0.15            +0.07
   (exact errors)      (wrong claims)         (spatial issues)    (timing bugs)
```

This ordering directly validates the paper's grounded feedback framework:
- **Most actionable**: Execution errors point to exact line and exact failure
- **Highly actionable**: Factual verification identifies specific wrong claims
- **Moderately actionable**: Visual screenshots show spatial problems
- **Least actionable**: Animation frames show temporal issues but diagnosis is hard

### Configuration
- Model: Claude Sonnet 4 (claude-sonnet-4-20250514)
- Temperature: 0.0
- Budget: 500K tokens per problem
- Max iterations: 5 (code), 3 (visual), 2-3 (research)
- Date: 2026-04-16
