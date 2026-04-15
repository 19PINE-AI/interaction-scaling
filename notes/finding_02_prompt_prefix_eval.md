# Finding 02: Prompt Prefix Required for Correct Evaluation

## Issue
HumanEval+ problems sometimes define helper functions in the prompt that tests
depend on. For example, HumanEval/38 defines `encode_cyclic` in the prompt,
and the test calls `encode_cyclic(str)` to verify `decode_cyclic`.

## Impact
Without including the prompt prefix, problems like HumanEval/38 always fail
with NameError even when the generated `decode_cyclic` is functionally correct.
The model was actually generating the right solution:

```python
groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
```

This correctly reverses the cyclic encoding. The "failure" was an evaluation
artifact, not a model failure.

## Fix Applied
Both the final evaluation in the runner AND the Type 3a execution feedback
now prepend `problem.prompt` to the generated code before execution. This
ensures helper definitions are available during testing.

## Lesson
This is a well-known issue in HumanEval evaluation — the standard approach
is to prepend the prompt. Important for reproducibility: always include
prompt-defined helper functions when evaluating solutions.

## Affected Problems
Any HumanEval problem where the prompt defines functions used by tests:
- HumanEval/38 (encode_cyclic / decode_cyclic)
- Potentially others with similar patterns
