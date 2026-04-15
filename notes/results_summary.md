# Results Summary (In Progress)

## Collected Data Points

### HumanEval+ (50 problems, pre-prompt-fix)
| Condition | Pass@1 | Avg Tokens | Failed |
|-----------|--------|------------|--------|
| B1: Single-shot | 94.0% (47/50) | 368 | /1, /32, /38 |
| B5: Agentic loop | 98.0% (49/50) | 544 | /38 |
| Ours: PR-Fixed | 98.0% (49/50) | 804 | /38 |

### HumanEval+ (164 problems, pre-prompt-fix)
| Condition | Pass@1 | Avg Tokens | Failed |
|-----------|--------|------------|--------|
| B1: Single-shot | 95.73% (157/164) | 439 | /1,/32,/38,/50,/83,/130,/132 |
| B5: Agentic loop | 97.56% (160/164) | 590 | /32,/38,/50,/132 |
| Ours: PR-Fixed | *pending* | | |

### MBPP+ (50 problems)
| Condition | Pass@1 | Avg Tokens | Failed |
|-----------|--------|------------|--------|
| B1: Single-shot | 98.0% (49/50) | 302 | Mbpp/87 |
| B5: Agentic loop | 100.0% (50/50) | 324 | none |
| Ours: PR-Fixed | 100.0% (50/50) | 346 | none |

## Key Analysis

### Failure Breakdown (164 HumanEval+, pre-fix)
- **Evaluation artifacts** (4 problems): /1,/32,/38,/50 - NameErrors from
  missing prompt context
- **Genuine B1-only failures** (3): /83,/130,/132 - logic/assertion errors
- **B5 fixes /83 and /130** but not /132
- **With prompt fix**: B1 should be ~98.2%, B5 ~99.4%

### Interaction Scaling Value
On HumanEval+ (pre-fix):
- B5 vs B1: +1.83pp (fixes 3 problems B1 missed)
- B5 uses 1.3x tokens vs B1

On MBPP+:
- B5 vs B1: +2.0pp (fixes 1 problem B1 missed)
- B5 uses 1.1x tokens vs B1

### Proposer-Reviewer vs Agentic Loop
- On easy problems, B5 ≈ Ours in pass@1
- Ours uses ~1.5x tokens vs B5 (reviewer overhead)
- Separation value expected on harder problems / larger context

## Pending Experiments
- [x] B1 full 164 (old eval)
- [x] B5 full 164 (old eval)
- [ ] Ours full 164 (old eval)
- [ ] B1 full 164 (v2 with fix)
- [ ] B5 full 164 (v2 with fix)
- [ ] Ours full 164 (v2 with fix)
- [ ] B2 self-review 50
- [ ] Feedback type ablation (Exp 1)
