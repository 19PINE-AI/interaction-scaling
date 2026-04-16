# Hard Benchmark Design and Results

## Six Benchmark Categories

| # | Category | Tasks | Feedback Type | Grounding Signal |
|---|----------|-------|---------------|------------------|
| 1 | Code (SWE-style) | 15 | Type 3a: Execution | Test pass/fail + error messages |
| 2 | Web pages | 15 | Type 3b: Visual | Browser screenshot + VLM review |
| 3 | Slides | 20 | Type 3b: Visual | Browser screenshot + VLM review |
| 4 | Animations | 15 | Type 3b+3c: Temporal | Multi-frame capture + VLM review |
| 5 | Video editing | 15 | Type 3c: Temporal | Keyframe extraction + VLM review |
| 6 | Deep research | 15 | Type 3d: Factual | Claim decomposition + verification |

**Total: 95 hard benchmark tasks across 4 feedback modalities.**

## Completed Results

### Code (SWE-style bug fixing) — 15 tasks
| Approach | Pass Rate |
|----------|-----------|
| Single-shot | **67%** (10/15) |
| With execution feedback | **100%** (15/15) |
| **Delta** | **+33pp** |

5 bugs fixed in 2 iterations each: CSV parser, date range DST,
markdown table, CJK text wrapping, wildcard trie.

### Slides — 5 tasks
| Approach | Avg Quality | Meets Reqs |
|----------|------------|------------|
| Single-shot | 0.76 | 60% |
| With visual review | 0.85 | 60% |
| **Delta** | +0.09 | +0pp |

### Animations — 8 tasks
| Approach | Avg Quality | Meets Reqs |
|----------|------------|------------|
| Single-shot | 0.29 | 25% |
| With frame review | 0.37 | 12% |
| **Delta** | +0.07 | -13pp |

### Web pages — 5 tasks (running)
### Deep research — 5 tasks (running)
### Video editing — 15 tasks (not yet run)

## Key Finding: Feedback Actionability Determines Value

The improvement from interaction scaling correlates directly with how
actionable the feedback signal is:

```
Execution errors:   +33pp   (precise: exact line, exact failure)
Visual screenshots: +0.09   (spatial: shows where, not why)
Animation frames:   +0.07   (temporal: shows when, harder to fix)
```

This supports the paper's framework: Type 3a (execution) >> Type 3b
(visual) > Type 3c (temporal) in terms of feedback actionability.
