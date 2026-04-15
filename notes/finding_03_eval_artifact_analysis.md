# Finding 03: Evaluation Artifacts vs Genuine Failures

## B1 Full HumanEval+ Failure Analysis (pre-fix)

7 failures at 95.73% pass@1:

### Evaluation Artifacts (4 problems - NameErrors from missing prompt context)
1. **HumanEval/1**: `List` type annotation undefined - import is in the prompt
2. **HumanEval/32**: `poly()` helper undefined - defined in the prompt
3. **HumanEval/38**: `encode_cyclic()` helper undefined - defined in the prompt
4. **HumanEval/50**: `encode_shift()` helper undefined - defined in the prompt

### Genuine Model Failures (3 problems)
5. **HumanEval/83** (`starts_one_ends`): AssertionError - logic error
6. **HumanEval/130** (`tri`): IndexError - off-by-one or edge case
7. **HumanEval/132** (`is_nested`): AssertionError - logic error

## Expected Fixed Results
- B1 (with fix): ~161/164 = 98.17% (predicted)
- B5 (with fix): likely 163-164/164 = 99.4-100% (fixes logic errors via feedback)
- Ours (with fix): similar to B5

## Implications for the Paper
1. **The first 50 problems understated B1's weakness** because they didn't
   include many prompt-helper problems. The full 164 had more.
2. **Interaction scaling's value is on the 3 genuine failures** - problems
   where the model's initial reasoning is wrong and execution feedback
   provides the corrective signal.
3. **The gap between B1 and interaction approaches may be small** on
   HumanEval+ because Claude Sonnet 4 is already very strong. Harder
   benchmarks (SWE-bench, etc.) would show larger gaps.
4. **Evaluation methodology matters** - reporting results without prompt
   prefix would significantly understate model capability.
