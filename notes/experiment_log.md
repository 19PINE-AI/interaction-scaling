# Experiment Log — Hard Benchmarks

## 2026-04-16: Hard Benchmark Results

### Completed Results

#### 1. Code — SWE-style Bug Fixing (15 tasks, Type 3a execution feedback)
| Approach | Pass Rate |
|----------|-----------|
| Single-shot | **67%** (10/15) |
| With execution feedback | **100%** (15/15) |
| **Delta** | **+33pp** |

5 bugs fixed, all in exactly 2 iterations:
- code_001: CSV parser escaped quote state machine
- code_003: Date range DST boundary comparison
- code_008: Markdown table pipe-inside-backtick
- code_011: CJK text wrapping width calculation
- code_013: Wildcard trie early termination

**This is the strongest result.** Execution feedback provides precise,
actionable error messages that the model uses to fix subtle bugs.

#### 2. Slides (5 tasks, Type 3b visual feedback)
| Approach | Avg Quality | Meets Requirements |
|----------|------------|-------------------|
| Single-shot | **0.76** | **60%** |
| With visual review | **0.85** | **60%** |
| **Delta** | **+0.09** | **+0pp** |

slide_002 improved from 0.80→0.95, slide_005 from 0.30→0.60.
Moderate improvement — VLM catches overflow/alignment but fixing CSS is harder.

#### 3. Animations (8 tasks, Type 3b+3c multi-frame feedback)
| Approach | Avg Quality | Meets Requirements |
|----------|------------|-------------------|
| Single-shot | **0.29** | **25%** |
| With frame review | **0.37** | **12%** |
| **Delta** | **+0.07** | **-13pp** |

Weakest result. Animation bugs are hard to diagnose from frame screenshots.

### Pending Results
- Web page generation (5 tasks): running
- Deep research (5 tasks): running
- Video editing (15 tasks): not yet run

### Cross-Modal Summary
```
Feedback Actionability vs Improvement:

Code execution:   +33pp  (precise errors → direct fix)
Slide visual:     +0.09  (spatial issues → CSS fix)
Animation frames: +0.07  (temporal issues → hard to fix)
```

The value of interaction scaling is directly proportional to the
actionability of the feedback signal.
