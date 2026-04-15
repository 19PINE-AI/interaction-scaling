"""Global configuration for the interaction scaling experiments."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.0

    @staticmethod
    def claude_sonnet() -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            max_tokens=8192,
        )

    @staticmethod
    def claude_haiku() -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-haiku-4-5-20251001",
            max_tokens=4096,
        )

    @staticmethod
    def gpt4() -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4.1-2025-04-14",
            max_tokens=8192,
        )


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    benchmark: str  # "humaneval" or "mbpp"
    num_problems: int | None = None  # None = all problems
    budget_tokens: int = 200_000
    max_iterations: int = 10
    proposer_model: ModelConfig = field(default_factory=ModelConfig.claude_sonnet)
    reviewer_model: ModelConfig = field(default_factory=ModelConfig.claude_sonnet)
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTES_DIR = PROJECT_ROOT / "notes"

# Execution sandbox settings
EXECUTION_TIMEOUT_SECONDS = 30
MAX_OUTPUT_LENGTH = 10_000  # chars of stdout/stderr to capture
