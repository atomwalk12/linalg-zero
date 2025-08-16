import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

from linalg_zero.grpo.compute_score import get_tool_reward
from linalg_zero.shared.lib import get_lib

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LinalgZeroTool(BaseTool):
    """
    Linear algebra calculation tool for GRPO training. Provides access to
    mathematical operations defined in shared/lib.py.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, Any] = {}
        self.lib = get_lib()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, ground_truth: str | None = None, **kwargs: dict) -> str:
        # Unique identifier for this class instance
        if instance_id is None:
            instance_id = str(uuid4())

        if ground_truth is None:
            raise ValueError("Ground truth is required for tool creation")

        # Store state in a dictionary
        self._instance_dict[instance_id] = {
            "tool_result": None,
            "ground_truth": json.loads(ground_truth),
            "reward": 0.0,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: dict) -> tuple[str, float, dict]:
        # breakpoint()
        """Execute a linear algebra calculation operation."""

        # This object is instantiated for a specific function schema
        function_name = self.tool_schema.function.name

        try:
            # Tool execution
            func = self.lib[function_name]
            tool_result = func(**parameters)

            # Reward calculation
            self._instance_dict[instance_id]["tool_result"] = tool_result
            reward = await self.calc_reward(instance_id)

            # The reward is saved for the final reward calculation upon interaction completion
            self._instance_dict[instance_id]["reward"] = reward

        except Exception as e:
            # This happens when the tool is not called with the correct args.
            error_msg = f"Error executing {function_name}: {e!s}"
            logger.exception(error_msg)

            reward = -0.1
            metadata = {"error": str(e), "function": function_name, "success": False}

            return error_msg, reward, metadata
        else:
            # Prepare response
            result = f"Executed {function_name}({parameters}) = {tool_result}"

            # Step-wise function execution is not awarded/penalized.
            reward = 0.0

            invocation_metadata = self._instance_dict[instance_id]["metadata"]
            metadata = {"function": function_name, "success": True, **invocation_metadata}
            return result, reward, metadata

    async def calc_reward(self, instance_id: str, **kwargs: dict) -> float:
        """Calculate reward based on tool execution success."""

        # Prepare the current execution result and ground truth value
        instance_data = self._instance_dict[instance_id]
        result = instance_data["tool_result"]
        ground_truth = instance_data["ground_truth"]

        reward, metadata = get_tool_reward(result, ground_truth)

        # Used to restore statistics after each tool invocation
        self._instance_dict[instance_id]["metadata"] = metadata

        # Notice that this function is used for step-wise progress as well as
        # final reward computation. It validates all tool calls.
        return reward

    async def release(self, instance_id: str, **kwargs: dict) -> None:
        del self._instance_dict[instance_id]
