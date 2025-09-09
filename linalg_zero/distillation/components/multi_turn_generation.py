from typing import TYPE_CHECKING, Any

from distilabel.steps.tasks.base import GeneratorTask
from pydantic import Field

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase

if TYPE_CHECKING:
    from distilabel.typing import GeneratorStepOutput, StepColumns


class MultiTurnWithToolUseGenerator(GeneratorTask, MultiTurnWithToolUseBase):
    """Simplified multi-turn generator that combines planning, execution, and summarization."""

    dataset: list[dict[str, Any]] = Field(description="Linear algebra problems to process")

    @property
    def outputs(self) -> "StepColumns":
        """Define what data this task produces for downstream steps."""
        return ["messages", "model_name", "final_answer", "is_correct"]

    def format_output(
        self,
        output: str | None,
        input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Does nothing."""
        return {}

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Generate multi-turn conversations from the source dataset.

        Args:
            offset: Starting index for generation (for resumable generation)

        Yields:
            Tuple of (batch_data, is_last_batch)
        """
        generated = offset
        dataset_size = len(self.dataset)

        while generated < dataset_size:
            # Send batch_size problems to the LLM at once
            batch_size = getattr(self, "batch_size", 8)
            rows_to_generate = min(batch_size, dataset_size - generated)

            # Get the next batch of problems
            batch_problems = self.dataset[generated : generated + rows_to_generate]

            # Simulate the conversation with the LLM
            batch_conversations = self._generate_with_pre_query_template(batch_problems)

            generated += rows_to_generate
            is_last_batch = generated >= dataset_size

            yield (batch_conversations, is_last_batch)
