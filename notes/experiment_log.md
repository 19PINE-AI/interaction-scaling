# Experiment Log

## 2026-04-16: Hard Benchmark Experiments

### Design Philosophy
All benchmarks are designed to be **genuinely hard** — tasks where single-shot
generation produces visible/testable problems that only grounded feedback
(visual rendering or code execution) can catch. Easy benchmarks like HumanEval+
(98%+ single-shot) are excluded because they don't demonstrate the value of
interaction scaling.

### Three Benchmark Modalities

#### 1. Slide Generation (20 tasks)
- Dense technical layouts, complex typography, multi-column designs
- Key failure modes: text overflow, element overlap, alignment issues
- Grounded feedback: browser rendering → screenshot → VLM review
- Expected single-shot quality: low (visual issues common with dense content)

#### 2. Animation Generation (15 tasks)
- Physics simulations, UI animations, coordinated multi-element motion
- Key failure modes: timing bugs, elements escaping viewport, incorrect physics
- Grounded feedback: multi-frame capture → VLM temporal analysis
- Expected single-shot quality: low (animation bugs invisible in code)

#### 3. Code with Subtle Bugs (15 tasks)
- Off-by-one, numerical precision, data structure edge cases, algorithm bugs
- Key failure modes: code looks correct but fails specific edge case tests
- Grounded feedback: execution with targeted test cases
- Expected single-shot quality: ~40-60% (subtle bugs that self-review misses)

### Experiments Running
- 10 slide tasks: single-shot vs 3-iteration visual review
- 8 animation tasks: single-shot vs 3-iteration frame review
- 15 code tasks: single-shot vs 5-iteration execution feedback

### Results
*Pending — experiments in progress*
