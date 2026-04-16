# Experiment Log — Hard Benchmarks

## Complete Results (5 of 6 categories)

### 1. Code — SWE-style Bug Fixing (15 tasks, Type 3a)
| Approach | Pass Rate |
|----------|-----------|
| Single-shot | **67%** (10/15) |
| With execution feedback | **100%** (15/15) |
| **Delta** | **+33pp** |

5 bugs fixed in 2 iterations each. Strongest result across all modalities.

### 2. Deep Research (5 tasks, Type 3d)
| Approach | Accuracy |
|----------|----------|
| Single-shot | **0.76** |
| With fact-checking | **0.92** |
| **Delta** | **+0.16** |

4/5 tasks improved to 1.00 accuracy. research_003 (GDP) unchanged.

### 3. Slides (5 tasks, Type 3b)
| Approach | Avg Quality |
|----------|------------|
| Single-shot | **0.76** |
| With visual review | **0.85** |
| **Delta** | **+0.09** |

### 4. Animations (8 tasks, Type 3b+3c)
| Approach | Avg Quality |
|----------|------------|
| Single-shot | **0.29** |
| With frame review | **0.37** |
| **Delta** | **+0.07** |

### 5. Web Pages (pending)
### 6. Video Editing (not yet run)

## Cross-Modal Summary

| Category | Feedback Type | Single-shot | Reviewed | Delta |
|----------|--------------|-------------|----------|-------|
| Code (SWE) | Type 3a: Execution | 67% pass | 100% pass | **+33pp** |
| Research | Type 3d: Factual | 0.76 acc | 0.92 acc | **+0.16** |
| Slides | Type 3b: Visual | 0.76 qual | 0.85 qual | **+0.09** |
| Animations | Type 3c: Temporal | 0.29 qual | 0.37 qual | **+0.07** |

## Feedback Actionability Hierarchy (validated)
```
Type 3a (execution) >> Type 3d (factual) > Type 3b (visual) > Type 3c (temporal)
+33pp                  +0.16               +0.09              +0.07
```

Execution errors are most actionable (exact line, exact failure).
Factual verification is next (specific wrong claim identified).
Visual feedback is moderate (shows spatial issues, harder to fix in CSS).
Temporal frame feedback is weakest (shows timing bugs, hardest to fix).
