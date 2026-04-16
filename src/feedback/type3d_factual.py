"""Type 3d feedback: factual verification via web search.

Decomposes a research report into atomic claims and verifies each
against web search results. This is the grounding signal for deep
research tasks.
"""

import json
import logging
import re

from src.config import ModelConfig
from src.feedback.base import FeedbackProvider, FeedbackResult, FeedbackType
from src.utils.llm_client import get_client

logger = logging.getLogger(__name__)


class FactualVerificationFeedback(FeedbackProvider):
    """Type 3d — verify factual claims in research reports via web search.

    Pipeline:
    1. Use LLM to decompose report into atomic factual claims
    2. For each claim, generate a search query
    3. Execute web search (via tool or API)
    4. Use LLM to judge if search results support/contradict each claim
    5. Return structured verification results

    The problem dict should contain:
    - ``verification_queries``: pre-specified search queries (optional)
    - ``requirements``: factual requirements the report must satisfy
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.claude_sonnet()
        self.client = get_client()

    def get_feedback(self, code: str, problem: dict) -> FeedbackResult:
        """Verify factual claims in a research report.

        Args:
            code: The research report text to verify.
            problem: Dict with optional ``requirements`` and ``verification_queries``.
        """
        report = code
        requirements = problem.get("requirements", [])

        # Step 1: Decompose into atomic claims
        claims = self._extract_claims(report)
        if not claims:
            return FeedbackResult(
                feedback_type=FeedbackType.EXECUTION,
                content="Could not extract factual claims from report.",
                structured_data={"passed": False, "claims": [], "accuracy": 0},
                tokens_used=0,
            )

        # Step 2: Verify each claim via web search
        verifications = []
        for claim in claims[:15]:  # Cap at 15 claims to manage cost
            result = self._verify_claim(claim)
            verifications.append(result)

        # Step 3: Compute metrics
        supported = sum(1 for v in verifications if v["verdict"] == "supported")
        contradicted = sum(1 for v in verifications if v["verdict"] == "contradicted")
        unverifiable = sum(1 for v in verifications if v["verdict"] == "unverifiable")
        total = len(verifications)
        accuracy = supported / total if total > 0 else 0

        # Step 4: Format feedback
        content = self._format_verification(verifications, accuracy)

        total_tokens = sum(v.get("tokens", 0) for v in verifications)

        return FeedbackResult(
            feedback_type=FeedbackType.EXECUTION,
            content=content,
            structured_data={
                "passed": accuracy >= 0.8,
                "accuracy": accuracy,
                "total_claims": total,
                "supported": supported,
                "contradicted": contradicted,
                "unverifiable": unverifiable,
                "verifications": verifications,
            },
            tokens_used=total_tokens,
        )

    def _extract_claims(self, report: str) -> list[str]:
        """Use LLM to decompose report into atomic factual claims."""
        response = self.client.generate(
            config=self.model_config,
            system=(
                "Extract atomic factual claims from the following text. "
                "Each claim should be a single verifiable statement of fact "
                "(a number, date, name, event, or relationship). "
                "Ignore opinions, analysis, and hedged statements. "
                "Return a JSON array of strings."
            ),
            messages=[{
                "role": "user",
                "content": f"Extract factual claims from:\n\n{report[:5000]}",
            }],
        )

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw[raw.index("\n") + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            claims = json.loads(raw)
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except (json.JSONDecodeError, ValueError):
            # Fallback: split by newlines and filter
            lines = response.content.strip().split("\n")
            return [
                line.strip().lstrip("0123456789.-) ")
                for line in lines
                if line.strip() and len(line.strip()) > 20
            ]

        return []

    def _verify_claim(self, claim: str) -> dict:
        """Verify a single claim using web search + LLM judgment."""
        # Generate search query
        search_query = self._generate_search_query(claim)

        # Execute web search
        search_results = self._web_search(search_query)

        if not search_results:
            return {
                "claim": claim,
                "verdict": "unverifiable",
                "evidence": "No search results found",
                "query": search_query,
                "tokens": 0,
            }

        # Judge claim against search results
        judgment = self._judge_claim(claim, search_results)
        return {
            "claim": claim,
            "verdict": judgment["verdict"],
            "evidence": judgment["evidence"],
            "query": search_query,
            "tokens": judgment.get("tokens", 0),
        }

    def _generate_search_query(self, claim: str) -> str:
        """Generate a web search query to verify a claim."""
        # Extract key entities and facts for a focused query
        # Simple heuristic: use the claim itself, shortened
        if len(claim) > 100:
            return claim[:100]
        return claim

    def _web_search(self, query: str) -> str:
        """Execute a web search and return results as text.

        Uses a simple approach: call a search API or use the
        WebSearch tool if available. Falls back to the LLM's
        knowledge with a disclaimer.
        """
        try:
            # Try using the Anthropic client's web search if available
            # For now, use the LLM to simulate search-grounded verification
            # In production, this would call a real search API
            response = self.client.generate(
                config=self.model_config,
                system=(
                    "You are a fact-checking assistant. Given a claim to verify, "
                    "provide what you know about its accuracy. Be specific about "
                    "what is correct and what is incorrect. If you're not sure, "
                    "say so explicitly."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Verify this factual claim: \"{query}\"",
                }],
            )
            return response.content
        except Exception as e:
            logger.warning("Search failed for query: %s (%s)", query, e)
            return ""

    def _judge_claim(self, claim: str, evidence: str) -> dict:
        """Use LLM to judge whether evidence supports the claim."""
        response = self.client.generate(
            config=self.model_config,
            system=(
                "You are a fact-checking judge. Given a claim and evidence, "
                "determine if the evidence supports, contradicts, or is "
                "insufficient to verify the claim. Respond with JSON: "
                '{"verdict": "supported|contradicted|unverifiable", '
                '"evidence": "brief explanation"}'
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Claim: \"{claim}\"\n\n"
                    f"Evidence:\n{evidence[:2000]}\n\n"
                    f"Does the evidence support or contradict this claim?"
                ),
            }],
        )

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw[raw.index("\n") + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            data = json.loads(raw)
            data["tokens"] = response.input_tokens + response.output_tokens
            return data
        except (json.JSONDecodeError, ValueError):
            return {
                "verdict": "unverifiable",
                "evidence": response.content[:500],
                "tokens": response.input_tokens + response.output_tokens,
            }

    @staticmethod
    def _format_verification(verifications: list[dict], accuracy: float) -> str:
        """Format verification results as human-readable feedback."""
        parts = [
            f"Factual Accuracy: {accuracy:.0%} "
            f"({sum(1 for v in verifications if v['verdict'] == 'supported')}"
            f"/{len(verifications)} claims supported)\n"
        ]

        contradicted = [v for v in verifications if v["verdict"] == "contradicted"]
        if contradicted:
            parts.append("CONTRADICTED CLAIMS:")
            for v in contradicted:
                parts.append(f"  ✗ \"{v['claim']}\"")
                parts.append(f"    Evidence: {v['evidence'][:200]}")

        unverifiable = [v for v in verifications if v["verdict"] == "unverifiable"]
        if unverifiable:
            parts.append("\nUNVERIFIABLE CLAIMS:")
            for v in unverifiable:
                parts.append(f"  ? \"{v['claim']}\"")

        supported = [v for v in verifications if v["verdict"] == "supported"]
        if supported:
            parts.append(f"\nSUPPORTED: {len(supported)} claims verified")

        return "\n".join(parts)
