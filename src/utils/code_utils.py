"""Utilities for extracting and processing code from LLM responses."""

import re


def extract_code(response: str, language: str = "python") -> str:
    """Extract code from an LLM response, handling markdown code blocks.

    Tries in order:
    1. Fenced code block with language tag (```python ... ```)
    2. Any fenced code block (``` ... ```)
    3. The raw response (assumed to be code)
    """
    # Try language-specific fenced block
    pattern = rf"```{language}\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try any fenced block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to raw response
    return response.strip()


def extract_function(code: str, function_name: str) -> str | None:
    """Extract a specific function definition from code."""
    pattern = rf"(def {function_name}\s*\(.*?\n(?:(?:    .*|)\n)*)"
    match = re.search(pattern, code)
    if match:
        return match.group(1).rstrip()
    return None


def combine_code_and_tests(solution: str, test_code: str) -> str:
    """Combine a solution with test code for execution."""
    return f"{solution}\n\n{test_code}"


def truncate_output(output: str, max_chars: int = 10_000) -> str:
    """Truncate output to a maximum number of characters."""
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n... [truncated {len(output) - max_chars} chars] ...\n\n"
        + output[-half:]
    )
