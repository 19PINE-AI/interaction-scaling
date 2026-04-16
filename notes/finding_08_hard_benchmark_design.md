# Finding 08: Hard Benchmark Design Philosophy

## Problem with HumanEval+
Claude Sonnet 4 achieves 98.17% single-shot on HumanEval+. The remaining
1.83% improvement from interaction scaling, while real, is too small to be
compelling. The benchmark is saturated.

## Design Principles for Hard Benchmarks

### 1. Tasks must have failure modes invisible in code but visible in output
- **Slides**: Text overflow, overlap, and alignment are invisible when reading
  HTML/CSS but immediately obvious in a rendered screenshot
- **Animations**: Timing glitches, element escaping viewport, incorrect physics
  are invisible in JS code but obvious in rendered frames
- **Code**: Off-by-one errors, numerical precision issues, edge cases look
  correct in code review but fail specific test inputs

### 2. Single-shot accuracy must be genuinely low
- Complex layouts with dense content → high chance of overflow
- Multi-element animations → high chance of coordination bugs
- Subtle algorithmic edge cases → high chance of incorrect handling

### 3. Grounded feedback must be actionable
- Screenshot shows exactly where text overflows → proposer can fix CSS
- Frame sequence shows exactly when animation glitches → proposer can fix timing
- Test error shows exactly which input fails → proposer can fix logic

### 4. Self-review must be unable to help
- Cannot "see" text overflow by re-reading CSS (needs rendering)
- Cannot detect timing bugs by re-reading JS (needs execution + frames)
- Cannot find off-by-one by re-reading algorithm (needs test execution)

## Benchmark Statistics
- 20 slide tasks: dense layouts, complex typography, multi-column, mixed media
- 15 animation tasks: physics sims, UI interactions, coordinated motion
- 15 code tasks: boundary conditions, numerics, data structure edge cases
- Total: 50 hard tasks across 3 modalities
