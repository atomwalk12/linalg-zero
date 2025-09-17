import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

from linalg_zero.grpo.compute_score import get_tool_reward
from linalg_zero.grpo.verify import verify_answers
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
        stepwise_ground_truth = kwargs.get("stepwise_ground_truth")
        assert isinstance(stepwise_ground_truth, str)  # noqa: S101
        # Store state in a dictionary
        self._instance_dict[instance_id] = {
            "tool_result": None,
            "ground_truth": json.loads(ground_truth),
            "stepwise_ground_truth": json.loads(stepwise_ground_truth),
            "reward": 0.0,
            "step_index": 0,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: dict) -> tuple[str, float, dict]:
        """Execute a linear algebra calculation operation."""

        # This object is instantiated for a specific function schema
        function_name = self.tool_schema.function.name

        try:
            # Tool execution
            if function_name not in self.lib:
                raise ValueError(f"Function {function_name} not found in library")  # noqa: TRY301
            func = self.lib[function_name]
            tool_result = func(**parameters)

            # Update state and calculate rewards
            self._instance_dict[instance_id]["tool_result"] = tool_result
            current_step = self._instance_dict[instance_id]["step_index"]
            reward = await self.calc_reward(instance_id, step_index=current_step, is_executing=True)

            # Increment step counter for next execution
            self._instance_dict[instance_id]["step_index"] = current_step + 1
            self._instance_dict[instance_id]["reward"] = reward

        except Exception as e:
            # This happens when the tool is not called with the correct args.
            error_msg = f"Error executing {function_name}: {e!s}"
            logger.exception(error_msg)

            reward = -0.1
            metadata = self._create_error_metadata(e, function_name, instance_id)
            return error_msg, reward, metadata
        else:
            # Prepare response
            result = f"Executed {function_name}({parameters}) = {tool_result}"

            # Step-wise function execution is not awarded/penalized.
            reward = 0.0

            invocation_metadata = self._instance_dict[instance_id]["metadata"]
            metadata = {"function": function_name, "success": True, **invocation_metadata}
            return result, reward, metadata

    async def calc_reward(
        self, instance_id: str, step_index: int | None = None, is_executing: bool = False, **kwargs: dict
    ) -> float:
        """Calculate reward based on tool execution success.

        Args:
            instance_id: Instance identifier
            step_index: Current step index for stepwise validation
            is_executing: Whether this is called during tool execution (enables stepwise metrics)

        Returns:
            final-answer reward (against ground_truth)
            Records stepwise metrics (against stepwise_ground_truth) for logging only
        """
        instance_data = self._instance_dict[instance_id]
        result = instance_data["tool_result"]
        ground_truth = instance_data["ground_truth"]

        # Always calculate final reward
        final_reward, final_meta = get_tool_reward(ground_truth=ground_truth, tool_output=result)
        is_final_correct = verify_answers(ground_truth, result)

        # Calculate stepwise metrics if executing
        stepwise_metrics = self._calculate_stepwise_metrics(
            instance_data, result, step_index=step_index, is_executing=is_executing
        )

        # Aggregate metadata for logging/metrics
        metadata = {
            "final_reward": final_reward,
            "is_final_correct": is_final_correct,
            "final_meta": final_meta,
            **stepwise_metrics,
        }

        self._instance_dict[instance_id]["metadata"] = metadata
        return final_reward

    def _calculate_stepwise_metrics(
        self, instance_data: dict, result: Any, step_index: int | None, is_executing: bool
    ) -> dict:
        """Extract stepwise metrics calculation into separate method."""
        stepwise_ground_truth = instance_data.get("stepwise_ground_truth")
        assert stepwise_ground_truth is not None  # noqa: S101

        # Early return if not executing or missing stepwise data
        if not is_executing or step_index is None:
            return {
                "stepwise_reward": 0.0,
                "is_stepwise_correct": False,
                "stepwise_meta": {},
            }

        if not (0 <= step_index < len(stepwise_ground_truth)):
            return {
                "stepwise_reward": 0.0,
                "is_stepwise_correct": False,
                "stepwise_meta": {},
            }

        # Calculate metrics for current step
        stepwise_expected = stepwise_ground_truth[step_index]
        stepwise_reward, stepwise_meta = get_tool_reward(ground_truth=stepwise_expected, tool_output=result)
        is_stepwise_correct = verify_answers(stepwise_expected, result)

        return {
            "stepwise_reward": stepwise_reward,
            "is_stepwise_correct": is_stepwise_correct,
            "stepwise_meta": stepwise_meta,
            "step_index": step_index,
        }

    def _create_error_metadata(self, error: Exception, function_name: str, instance_id: str) -> dict:
        """Create standardized error metadata for failed tool executions."""
        current_step = self._instance_dict.get(instance_id, {}).get("step_index", 0)

        return {
            "error": str(error),
            "error_type": error.__class__.__name__,
            "function": function_name,
            "success": False,
            "step_index": current_step,
            "final_reward": 0.0,
            "is_final_correct": False,
            "stepwise_reward": 0.0,
            "is_stepwise_correct": False,
        }

    async def release(self, instance_id: str, **kwargs: dict) -> None:
        del self._instance_dict[instance_id]
