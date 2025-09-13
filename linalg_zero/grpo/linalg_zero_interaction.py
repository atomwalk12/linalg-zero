import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from linalg_zero.grpo.compute_score import (
    get_interaction_reward,
)
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.lib import get_lib_fn_names
from linalg_zero.shared.types import LibTypes

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LinalgZeroInteraction(BaseInteraction):
    """Evaluates tool-assistant interactions and guides the model to reach the correct result."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, Any] = {}

        # The expected completion output is <think> AND (<tool_call> OR <answer>)
        self.parser = XMLParser()
        self._tool_names: list[str] = get_lib_fn_names()

    async def start_interaction(
        self, instance_id: str | None = None, ground_truth: str | None = None, **kwargs: dict
    ) -> str:
        # Unique identifier for this class instance
        if instance_id is None:
            instance_id = str(uuid4())

        if ground_truth is None:
            raise ValueError("Ground truth is required for interaction creation")

        # Store state in a dictionary
        self._instance_dict[instance_id] = {"messages": [], "ground_truth": json.loads(ground_truth)}
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs: dict
    ) -> tuple[bool, str, float, dict]:
        """Generate response to the model during rollout."""
        self._instance_dict[instance_id]["messages"] = messages

        # Calculate reward for current user-assistant turn
        reward = await self.calc_reward(instance_id)
        metadata = self._instance_dict[instance_id]["metadata"]

        # Use metadata for intelligent decision making
        answer_correct = metadata["reward_final_answer"] is True
        format_good = metadata["reward_response_format"] is True

        # Response lookup table: (answer_correct, format_good) -> (response, terminate)
        # TODO: Make the feedback more detailed ("Step 3 matrix multiplication is incorrect")
        responses = {
            (True, True): ("Excellent work! Your solution is correct and well-formatted.", True),
            (True, False): ("Good job! Your answer is correct, but please improve your formatting next time.", True),
            (False, True): ("Your formatting looks good, but please check your calculations.", False),
            (False, False): ("Please check your calculations and formatting.", False),
        }

        response, should_terminate_sequence = responses[(answer_correct, format_good)]

        return should_terminate_sequence, response, reward, metadata

    async def calc_reward(self, instance_id: str, **kwargs: dict) -> float:
        """Calculate reward based on final answer and response formatting."""
        # Retrieve state
        instance_data = self._instance_dict[instance_id]
        messages = instance_data["messages"]
        ground_truth: LibTypes = instance_data["ground_truth"]

        # Compute reward and store metrics for analysis
        reward, metadata = get_interaction_reward(parser=self.parser, ground_truth=ground_truth, completion=messages)
        self._instance_dict[instance_id]["metadata"] = metadata

        # Notice that this function is not used for step-wise progress. It is called
        # upon trajectory completion to assert the correctness of the final tool call.
        return reward

    def _diagnose(self, messages: list[dict[str, Any]]) -> str:
        """Produce a short message stored in metadata for diagnostic purposes."""
        if not messages:
            return "empty conversation"

        assistant_msg: str | None = self.parser.get_last_message(messages, role="assistant")

        if assistant_msg is None or not assistant_msg.strip():
            return "empty generation"

        try:
            analysis = self.parser.analyze_message_in_context(
                context=messages,
                message=assistant_msg,
                tool_names=self._tool_names,
            )
        except Exception:
            return "unable to analyze output"

        return self.parser.get_analysis_failure_reason(analysis, tool_names=self._tool_names)

    async def finalize_interaction(self, instance_id: str, **kwargs: dict) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        del self._instance_dict[instance_id]
