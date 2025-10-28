"""
Linear Algebra Environment for GRPO training.

This module implements an environment for training language models
on linear algebra problems with tool calling capabilities.
"""

import json
import random
import uuid
from typing import Any

from linalg_zero.grpo.openpipe_art.data_types import LinearAlgebraTrainingConfig, RunConfig
from linalg_zero.shared.utils import get_logger

from .base_env import Env, Tool
from .base_types import (
    RESPOND_ACTION_NAME,
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardActionInfo,
    RewardResult,
    Task,
)

logger = get_logger(__name__)


class LinAlgTask(Task):
    """
    Linear algebra task compatible with HuggingFace dataset format.

    This class works directly with the dataset entries from atomwalk12/linalgzero
    and atomwalk12/linalgzero-distilled without requiring translation overhead.
    """

    # Core dataset fields (from HuggingFace datasets)
    query: str  # Problem statement
    ground_truth: str  # Expected answer (JSON string)
    stepwise_ground_truths: str  # Intermediate steps (JSON string)
    tools: list[dict] | None = None  # Available tools

    # Optional metadata fields
    difficulty_level: int = 1
    problem_type: str = "general"

    @classmethod
    def from_dataset_entry(cls, entry: dict[str, Any]) -> "LinAlgTask":
        """
        Create LinAlgTask from HuggingFace dataset entry.

        Args:
            entry: Dataset entry with fields like query, ground_truth, etc.

        Returns:
            LinAlgTask instance
        """
        return cls(
            user_id=str(entry.get("id", entry.get("idx", 0))),
            instruction=entry["query"],
            actions=[],  # Will be populated during episode
            outputs=[entry["ground_truth"]],
            query=entry["query"],
            ground_truth=entry["ground_truth"],
            stepwise_ground_truths=entry["stepwise_ground_truths"],
            tools=entry.get("tools"),
            difficulty_level=entry.get("difficulty_level", len(json.loads(entry["stepwise_ground_truths"]))),
            problem_type=entry.get("problem_type", "general"),
        )

    def get_ground_truth_parsed(self) -> Any:
        """Parse ground truth from JSON string."""
        import json

        return json.loads(self.ground_truth)

    def get_stepwise_ground_truths_parsed(self) -> list[dict[str, Any]]:
        """Parse stepwise ground truths from JSON string."""
        import json

        return json.loads(self.stepwise_ground_truths)


class LinAlgEnvironment(Env):
    """
    Linear algebra environment extending tau-bench Env base class.

    This environment manages linear algebra problem solving sessions,
    providing tools for matrix operations and evaluating agent performance.
    """

    def __init__(
        self,
        tools: list[type[Tool]],
        tasks: list[LinAlgTask],
        config: RunConfig,
        wiki: str = "",
        rules: list[str] | None = None,
    ):
        """
        Initialize the LinAlgEnvironment.

        Args:
            tools: List of tool classes for mathematical operations
            tasks: List of linear algebra tasks
            config: Run configuration
            wiki: Wiki information (optional)
            rules: List of environment rules (optional)
        """

        # Default rules if none provided
        if rules is None:
            rules = [
                "Use the available mathematical tools to solve linear algebra problems",
                "Provide your final answer in the specified format",
                "Show your reasoning step by step",
            ]

        # Initialize parent class
        super().__init__(
            tools=tools,
            tasks=tasks,
            wiki=wiki,
            rules=rules,
        )

        self.config = config

        # Environment state management
        self.current_task: LinAlgTask | None = None
        self.tool_call_history: list[Action] = []
        self.intermediate_results: dict[str, Any] = {}
        self.session_id: str | None = None
        self.episode_step_count: int = 0
        self.max_steps: int = getattr(config, "max_num_steps", 30)

    async def reset(self, task_index: int | None = None) -> EnvResetResponse:
        """
        Reset the environment with a new task.

        Args:
            task_index: Index of task to use, random if None

        Returns:
            EnvResetResponse with initial observation and task info
        """
        # Select task index
        if task_index is None:
            task_index = random.randint(0, len(self.tasks) - 1)

        self.task_index = task_index
        self.current_task = self.tasks[task_index]
        assert self.current_task is not None  # noqa: S101

        # Reset all environment state
        self._reset_episode_state()

        # Generate session ID for tracking
        self.session_id = str(uuid.uuid4())

        # Get initial observation from user simulator
        initial_observation = self.user.reset(instruction=self.current_task.instruction)

        return EnvResetResponse(observation=initial_observation, info=EnvInfo(task=self.current_task, source="user"))

    def _reset_episode_state(self) -> None:
        """Reset all episode-specific state variables."""
        self.actions = []
        self.tool_call_history = []
        self.intermediate_results = {}
        self.episode_step_count = 0
        self.session_id = None

    async def step(self, action: Action) -> EnvResponse:
        """
        Process an agent action and return the environment response.

        Args:
            action: Action taken by the agent

        Returns:
            EnvResponse with observation, reward, done flag, and info
        """
        # Update step count and state
        self.episode_step_count += 1
        self.actions.append(action)

        info = EnvInfo(task=self.current_task)
        reward = 0
        done = False

        # Check for maximum steps exceeded
        if self.episode_step_count >= self.max_steps:
            done = True
            observation = f"Maximum steps ({self.max_steps}) exceeded. Episode terminated."
            info.source = "environment"

        elif action.name == RESPOND_ACTION_NAME:
            # Agent is providing a final response
            observation = self.user.step(action.kwargs["content"])
            info.source = "user"
            done = "###STOP###" in observation

        elif action.name in self.tools_map:
            # Agent is calling a mathematical tool
            try:
                observation = self.tools_map[action.name].invoke(**action.kwargs)
                # Store intermediate result for potential use in reward calculation
                self._store_intermediate_result(action.name, observation, action.kwargs)
                self.tool_call_history.append(action)

            except Exception as e:
                observation = f"Error: {e}"

            info.source = action.name

            # Check if this tool terminates the episode
            if action.name in self.terminate_tools:
                done = True

        else:
            # Unknown action
            observation = f"Unknown action {action.name}"
            info.source = action.name

        # Calculate reward if episode is done
        if done:
            reward_res = self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def _store_intermediate_result(self, tool_name: str, result: Any, kwargs: dict[str, Any]) -> None:
        """
        Store intermediate result from a tool call with metadata.

        Args:
            tool_name: Name of the tool that was called
            result: Result returned by the tool
            kwargs: Arguments passed to the tool
        """
        self.intermediate_results[f"{tool_name}_{self.episode_step_count}"] = {
            "tool": tool_name,
            "result": result,
            "kwargs": kwargs,
            "step": self.episode_step_count,
        }

        # Also store the latest result for each tool type
        self.intermediate_results[f"latest_{tool_name}"] = result

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        Get information about available mathematical tools.

        Returns:
            List of tool information dictionaries
        """
        return self.tools_info

    def get_current_matrices(self) -> dict[str, list[list[float]]]:
        """
        Get the current matrices available in the environment.

        Note: This method returns empty dict because matrices are passed
        directly to tools through their arguments, not stored in environment.

        Returns:
            Empty dictionary (matrices are passed directly to tools)
        """
        return {}

    def get_tool_call_history(self) -> list[Action]:
        """
        Get the history of tool calls made during the current episode.

        Returns:
            List of tool call actions
        """
        return self.tool_call_history.copy()

    def get_intermediate_results(self) -> dict[str, Any]:
        """
        Get intermediate results from tool calls.

        Returns:
            Dictionary mapping tool names to their results
        """
        return self.intermediate_results.copy()

    def get_environment_state(self) -> dict[str, Any]:
        """
        Get complete environment state for debugging and monitoring.

        Returns:
            Dictionary containing all environment state information
        """
        return {
            "session_id": self.session_id,
            "task_index": getattr(self, "task_index", None),
            "current_task_id": self.current_task.user_id if self.current_task else None,
            "episode_step_count": self.episode_step_count,
            "max_steps": self.max_steps,
            "num_actions": len(self.actions),
            "num_tool_calls": len(self.tool_call_history),
            "available_matrices": self.get_current_matrices(),
            "intermediate_results_count": len(self.intermediate_results),
            "ground_truth": self.current_task.ground_truth if self.current_task else None,
            "stepwise_ground_truths": self.current_task.stepwise_ground_truths if self.current_task else None,
        }

    def is_episode_done(self) -> bool:
        """
        Check if the current episode should be terminated.

        Returns:
            True if episode should end, False otherwise
        """
        return self.episode_step_count >= self.max_steps

    def get_step_count(self) -> int:
        """
        Get the current step count for this episode.

        Returns:
            Number of steps taken in current episode
        """
        return self.episode_step_count

    def set_max_steps(self, max_steps: int) -> None:
        """
        Set the maximum number of steps allowed per episode.

        Args:
            max_steps: Maximum number of steps
        """
        self.max_steps = max_steps

    def get_task_info(self) -> dict[str, Any]:
        """
        Get information about the current task.

        Returns:
            Dictionary with current task information
        """
        if not self.current_task:
            return {}

        return {
            "user_id": self.current_task.user_id,
            "instruction": self.current_task.instruction,
            "problem_type": getattr(self.current_task, "problem_type", "unknown"),
            "difficulty_level": getattr(self.current_task, "difficulty_level", 1),
            "expected_outputs": self.current_task.outputs,
            "matrix_data_keys": [],  # Matrices passed directly to tools, not stored in env
            "ground_truth": self.current_task.ground_truth,
            "stepwise_ground_truths": self.current_task.stepwise_ground_truths,
        }

    def calculate_reward(self) -> RewardResult:
        """
        Calculate reward using existing dual reward system (format + accuracy).

        This method leverages the existing reward functions from compute_score.py
        to evaluate both XML format compliance and mathematical accuracy.

        Returns:
            RewardResult with composite reward and metadata
        """
        if not self.current_task:
            # No task available, return zero reward
            info = RewardActionInfo(r_actions=0.0, gt_data_hash=self.get_data_hash())
            return RewardResult(reward=0.0, info=info, actions=self.actions)

        # Get the agent's messages from the last solve session
        # We need to reconstruct the conversation from the actions taken
        messages = self._reconstruct_conversation_messages()

        if not messages:
            # No messages available, return zero reward
            info = RewardActionInfo(r_actions=0.0, gt_data_hash=self.get_data_hash())
            return RewardResult(reward=0.0, info=info, actions=self.actions)

        try:
            # Import reward calculation dependencies only when needed
            # This avoids import errors when distillation dependencies are not available
            from linalg_zero.grpo.compute_score import get_interaction_reward
            from linalg_zero.grpo.verifiers.xml_parser import XMLParser

            # Parse ground truth from JSON string
            ground_truth = json.loads(self.current_task.ground_truth)

            # Use existing reward calculation system
            parser = XMLParser()
            reward, metadata = get_interaction_reward(parser, ground_truth=ground_truth, completion=messages)

            logger.debug(f"Calculated reward: {reward}, metadata: {metadata}")

        except ImportError as e:
            logger.warning(f"Reward calculation dependencies not available: {e}")
            # Fallback to basic reward calculation
            reward = 1.0 if self.actions else 0.0
            metadata = {"fallback": "basic_reward", "import_error": str(e)}

        except Exception as e:
            logger.exception("Error calculating reward")
            reward = 0.0
            metadata = {"error": str(e)}

        # Return result in expected format
        info = RewardActionInfo(r_actions=reward, gt_data_hash=self.get_data_hash())
        return RewardResult(reward=reward, info=info, actions=self.actions)

    def _reconstruct_conversation_messages(self) -> list[dict[str, Any]]:
        """
        Reconstruct conversation messages from actions and responses.

        This method creates the message format expected by existing reward functions
        by reconstructing the conversation from the environment's action history.

        Returns:
            List of messages in OpenAI chat format
        """
        messages = []

        # Add system message if we have one
        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add initial user message with the task
        if self.current_task:
            messages.append({"role": "user", "content": self.current_task.instruction})

        # Reconstruct conversation from actions
        for action in self.actions:
            if action.name == RESPOND_ACTION_NAME:
                # Agent provided a final response
                content = action.kwargs.get("content", "")
                messages.append({"role": "assistant", "content": content})
            else:
                # Agent made a tool call - we need to simulate the assistant message
                # and the tool response based on our stored intermediate results

                # Create assistant message with tool call
                tool_call_message = {
                    "role": "assistant",
                    "content": f"I'll use the {action.name} tool to solve this.",
                    "tool_calls": [
                        {
                            "id": f"call_{len(messages)}",
                            "type": "function",
                            "function": {"name": action.name, "arguments": json.dumps(action.kwargs)},
                        }
                    ],
                }
                messages.append(tool_call_message)

                # Add tool response if we have it stored
                tool_result = self.intermediate_results.get(f"latest_{action.name}", "Tool executed")
                tool_response = {
                    "role": "tool",
                    "tool_call_id": f"call_{len(messages) - 1}",
                    "name": action.name,
                    "content": str(tool_result),
                }
                messages.append(tool_response)

        return messages


def load_linalg_tasks_from_hub(
    dataset_name: str = "atomwalk12/linalgzero-grpo", split: str = "validation", max_tasks: int | None = None
) -> list[LinAlgTask]:
    """
    Load linear algebra tasks from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load ("train", "validation", "test")
        max_tasks: Maximum number of tasks to load (None for all)

    Returns:
        List of LinAlgTask instances
    """
    try:
        from datasets import load_dataset

        logger.info(f"Loading tasks from {dataset_name}, split: {split}")
        dataset = load_dataset(dataset_name, split=split)

        # Limit number of tasks if specified
        if max_tasks is not None:
            dataset = dataset.select(range(min(max_tasks, len(dataset))))

        # Convert dataset entries to LinAlgTask instances
        tasks = []
        for i, entry in enumerate(dataset):
            try:
                task = LinAlgTask.from_dataset_entry(entry)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to create task from entry {i}: {e}")
                continue

    except Exception:
        logger.exception(f"Failed to load dataset {dataset_name}")
        logger.info("Falling back to empty task list - no sample tasks available")
        return []
    else:
        logger.info(f"Successfully loaded {len(tasks)} tasks from {dataset_name}")
        return tasks


def create_linalg_environment(
    config: RunConfig,
    dataset_name: str | None = None,
    max_tasks: int | None = None,
) -> LinAlgEnvironment:
    """
    Create a LinAlgEnvironment with all necessary tools and tasks.

    Args:
        config: Run configuration
        dataset_name: HuggingFace dataset name to load tasks from
        max_tasks: Maximum number of tasks to load from dataset

    Returns:
        Configured LinAlgEnvironment instance
    """
    # Import here to avoid circular imports
    from linalg_zero.grpo.openpipe_art.linalg_tools import get_linalg_tools

    # Use task_split from config to determine which dataset split to load
    # Following tau-bench patterns: "train" for training, "validation" for evaluation
    task_split = getattr(config, "task_split", "train")
    dataset_name = dataset_name or "atomwalk12/linalgzero-grpo"

    logger.info(f"Loading tasks from {dataset_name}, split: {task_split}")
    tasks = load_linalg_tasks_from_hub(dataset_name=dataset_name, split=task_split, max_tasks=max_tasks)

    # Get all available tools
    tools = get_linalg_tools()

    # Create and return environment
    return LinAlgEnvironment(
        tools=tools,
        tasks=tasks,
        config=config,
    )


def create_linalg_eval_environment(
    config: RunConfig,
    training_config: LinearAlgebraTrainingConfig | None = None,
    dataset_name: str | None = None,
    max_tasks: int | None = None,
) -> LinAlgEnvironment:
    """
    Create a LinAlgEnvironment for evaluation using validation split.

    Following tau-bench patterns, this creates an environment specifically
    for evaluation using the validation dataset split.

    Args:
        config: Run configuration
        training_config: Training configuration with eval settings
        dataset_name: HuggingFace dataset name to load tasks from
        max_tasks: Maximum number of tasks to load from dataset

    Returns:
        Configured LinAlgEnvironment instance for evaluation
    """
    # Import here to avoid circular imports
    from linalg_zero.grpo.openpipe_art.linalg_tools import get_linalg_tools

    # Determine evaluation split from training config or default to validation
    eval_split = "validation"
    if training_config and hasattr(training_config, "eval_dataset_split"):
        eval_split = training_config.eval_dataset_split

    dataset_name = dataset_name or "atomwalk12/linalgzero-grpo"

    logger.info(f"Loading evaluation tasks from {dataset_name}, split: {eval_split}")
    tasks = load_linalg_tasks_from_hub(dataset_name=dataset_name, split=eval_split, max_tasks=max_tasks)

    # Get all available tools
    tools = get_linalg_tools()

    # Create evaluation environment
    return LinAlgEnvironment(
        tools=tools,
        tasks=tasks,
        config=config,
    )
