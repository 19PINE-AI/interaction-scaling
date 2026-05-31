"""Global configuration for the interaction scaling experiments."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.0
    use_thinking: bool = False
    thinking_budget: int = 10000

    @staticmethod
    def claude_sonnet() -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            max_tokens=8192,
        )

    @staticmethod
    def claude_sonnet_thinking() -> "ModelConfig":
        """Claude Sonnet 4.6 with extended thinking for SFT distillation."""
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-4-6",
            max_tokens=16000,
            use_thinking=True,
            thinking_budget=10000,
        )

    @staticmethod
    def claude_opus() -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-opus-4-6",
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

    @staticmethod
    def qwen3_8b(model_path: str = "Qwen/Qwen3-8B") -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.LOCAL,
            model_id=model_path,
            max_tokens=4096,
            temperature=0.7,
        )

    @staticmethod
    def gemma4_31b(model_path: str = "google/gemma-4-31B-it") -> "ModelConfig":
        return ModelConfig(
            provider=ModelProvider.LOCAL,
            model_id=model_path,
            max_tokens=8192,
            temperature=0.7,
        )

    @staticmethod
    def qwen3_235b() -> "ModelConfig":
        """Qwen3-235B-Instruct-2507 via OpenRouter — bare `qwen3-235b-a22b`
        has Alibaba-only routing that is not enabled on this account; the
        2507 instruct variant is the working ID and matches the Phase-5
        teacher model."""
        return ModelConfig(
            provider=ModelProvider.OPENROUTER,
            model_id="qwen/qwen3-235b-a22b-2507",
            max_tokens=0,  # 0 = no limit, let model decide
            temperature=0.7,
        )

    @staticmethod
    def deepseek_r1() -> "ModelConfig":
        """DeepSeek R1 via OpenRouter — strong reasoning with <think> traces."""
        return ModelConfig(
            provider=ModelProvider.OPENROUTER,
            model_id="deepseek/deepseek-r1-0528",
            max_tokens=0,
            temperature=0.7,
        )

    @staticmethod
    def gpt5() -> "ModelConfig":
        """GPT-5 via OpenRouter — second non-Anthropic / non-Qwen
        cross-family replication for the harness lift."""
        return ModelConfig(
            provider=ModelProvider.OPENROUTER,
            model_id="openai/gpt-5",
            max_tokens=0,  # let model decide
            temperature=0.7,
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
