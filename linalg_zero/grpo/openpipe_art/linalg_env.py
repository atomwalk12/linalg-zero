"""
Linear Algebra Environment for GRPO training.

This module implements an environment for training language models
on linear algebra problems with tool calling capabilities.
"""

import json
import random
import uuid
from collections.abc import Callable
from typing import Any

from linalg_zero.grpo.openpipe_art.data_types import RunConfig
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
            difficulty_level=entry.get("difficulty_level", 1),
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

    def extract_matrix_data(self) -> dict[str, list[list[float]]]:
        """
        Extract matrix data from the query string.

        This method parses the query to find matrix definitions like:
        "matrix A = [[1, 2], [3, 4]]"

        Returns:
            Dictionary mapping matrix names to their values
        """
        import re

        matrix_data = {}

        # Pattern to match matrix definitions in queries
        # Matches: "matrix A = [[1, 2], [3, 4]]" or "A = [[1, 2], [3, 4]]"
        matrix_pattern = r"(?:matrix\s+)?([A-Z])\s*=\s*(\[\[.*?\]\])"

        matches = re.findall(matrix_pattern, self.query)

        for matrix_name, matrix_str in matches:
            try:
                # Safely evaluate the matrix string
                import ast

                matrix_value = ast.literal_eval(matrix_str)
                matrix_data[matrix_name] = matrix_value
            except (ValueError, SyntaxError):
                # If parsing fails, skip this matrix
                continue

        return matrix_data

    def validate(self) -> tuple[bool, list[str]]:  # noqa:C901
        """
        Validate the task structure and data consistency.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        if not self.query or not self.query.strip():
            errors.append("Query field is empty or missing")

        if not self.ground_truth:
            errors.append("Ground truth field is missing")

        if not self.stepwise_ground_truths:
            errors.append("Stepwise ground truths field is missing")

        # Validate JSON fields
        try:
            self.get_ground_truth_parsed()
        except Exception as e:
            errors.append(f"Invalid ground truth JSON: {e}")

        try:
            stepwise = self.get_stepwise_ground_truths_parsed()
            if not isinstance(stepwise, list):
                errors.append("Stepwise ground truths must be a list")
            elif len(stepwise) == 0:
                errors.append("Stepwise ground truths cannot be empty")
        except Exception as e:
            errors.append(f"Invalid stepwise ground truths JSON: {e}")

        # Validate matrix data extraction
        try:
            matrix_data = self.extract_matrix_data()
            for matrix_name, matrix_value in matrix_data.items():
                if not self._is_valid_matrix(matrix_value):
                    errors.append(f"Invalid matrix format for {matrix_name}: {matrix_value}")
        except Exception as e:
            errors.append(f"Error extracting matrix data: {e}")

        # Validate tools if present
        if self.tools is not None:
            if not isinstance(self.tools, list):
                errors.append("Tools field must be a list")
            else:
                for i, tool in enumerate(self.tools):
                    if not isinstance(tool, dict):
                        errors.append(f"Tool {i} must be a dictionary")
                    elif "function" not in tool:
                        errors.append(f"Tool {i} missing 'function' field")

        return len(errors) == 0, errors

    def _is_valid_matrix(self, matrix: Any) -> bool:
        """
        Check if a value represents a valid matrix.

        Args:
            matrix: Value to check

        Returns:
            True if valid matrix format
        """
        if not isinstance(matrix, list):
            return False

        if len(matrix) == 0:
            return False

        # Check if all rows are lists and have same length
        first_row_length = None
        for row in matrix:
            if not isinstance(row, list):
                return False
            if first_row_length is None:
                first_row_length = len(row)
            elif len(row) != first_row_length:
                return False

            # Check if all elements are numbers
            for element in row:
                if not isinstance(element, (int, float)):
                    return False

        return True


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
        data_load_func: Callable[[], dict[str, Any]] | None = None,
        wiki: str = "",
        rules: list[str] | None = None,
    ):
        """
        Initialize the LinAlgEnvironment.

        Args:
            tools: List of tool classes for mathematical operations
            tasks: List of linear algebra tasks
            config: Run configuration
            data_load_func: Function to load environment data
            wiki: Wiki information (optional)
            rules: List of environment rules (optional)
        """
        # Default data loader if none provided
        if data_load_func is None:
            data_load_func = lambda: {}

        # Default rules if none provided
        if rules is None:
            rules = [
                "Use the available mathematical tools to solve linear algebra problems",
                "Provide your final answer in the specified format",
                "Show your reasoning step by step",
            ]

        # Initialize parent class
        super().__init__(
            data_load_func=data_load_func,
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

    def reset(self, task_index: int | None = None) -> EnvResetResponse:
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

        # Load fresh data
        self.data = self.data_load_func()

        # Add task-specific matrix data to environment data
        if self.current_task:
            matrix_data = self.current_task.extract_matrix_data()
            self.data.update(matrix_data)

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

    def step(self, action: Action) -> EnvResponse:
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
                observation = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
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

        Returns:
            Dictionary mapping matrix names to their values
        """
        if self.current_task:
            return self.current_task.extract_matrix_data()
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
            "matrix_data_keys": list(self.current_task.extract_matrix_data().keys()),
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

        # Validate tasks
        valid_tasks, validation_errors = validate_task_dataset(tasks)

        if validation_errors:
            logger.warning(f"Dataset validation found {len(validation_errors)} errors")
            for error in validation_errors[:5]:  # Show first 5 errors
                logger.warning(f"  {error}")
            if len(validation_errors) > 5:
                logger.warning(f"  ... and {len(validation_errors) - 5} more errors")

        # Analyze dataset compatibility
        compatibility = validate_dataset_compatibility(valid_tasks)
        logger.info(
            f"Dataset analysis: {compatibility['total_tasks']} tasks, "
            f"{len(compatibility['problem_types'])} problem types, "
            f"{len(compatibility['compatibility_issues'])} compatibility issues"
        )
    except Exception:
        logger.exception(f"Failed to load dataset {dataset_name}")
        logger.info("Falling back to sample tasks for testing")
        return create_sample_tasks()
    else:
        logger.info(f"Successfully loaded {len(valid_tasks)} valid tasks from {dataset_name}")
        return valid_tasks


def load_linalg_tasks_from_prepared_datasets(
    train_dataset: str = "atomwalk12/linalgzero-distilled",
    validation_dataset: str = "atomwalk12/linalgzero",
    use_train_split: bool = False,
    max_tasks: int | None = None,
) -> list[LinAlgTask]:
    """
    Load tasks directly from the original datasets (before GRPO preparation).

    This function replicates the logic from prepare_dataset.py to load and process
    the original datasets on-the-fly. Useful for development and testing.

    Args:
        train_dataset: Training dataset name (has solutions)
        validation_dataset: Validation dataset name (no solutions)
        use_train_split: Whether to use training data (with solutions)
        max_tasks: Maximum number of tasks to load

    Returns:
        List of LinAlgTask instances
    """
    try:
        import json

        from datasets import load_dataset

        from linalg_zero.shared.lib import get_tools

        if use_train_split:
            logger.info(f"Loading training tasks from {train_dataset}")
            dataset = load_dataset(train_dataset, split="train")

            # Process training data (has messages field)
            def process_train_entry(entry: dict[str, Any]) -> dict[str, Any]:
                # Parse messages from JSON string
                messages = json.loads(entry["messages"])

                return {
                    "query": entry["query"],
                    "ground_truth": entry["ground_truth"],
                    "stepwise_ground_truths": entry["stepwise_ground_truths"],
                    "tools": get_tools(),
                    "messages": messages,  # Keep original messages for reference
                    "id": entry.get("id", entry.get("idx", 0)),
                }

            processed_entries = [process_train_entry(entry) for entry in dataset]

        else:
            logger.info(f"Loading validation tasks from {validation_dataset}")
            dataset = load_dataset(validation_dataset, split="validation")

            # Process validation data (no messages, need to create them)
            def process_validation_entry(entry: dict[str, Any]) -> dict[str, Any]:
                return {
                    "query": entry["query"],
                    "ground_truth": entry["ground_truth"],
                    "stepwise_ground_truths": entry["stepwise_ground_truths"],
                    "tools": get_tools(),
                    "id": entry.get("id", entry.get("idx", 0)),
                }

            processed_entries = [process_validation_entry(entry) for entry in dataset]

        # Limit number of tasks if specified
        if max_tasks is not None:
            processed_entries = processed_entries[:max_tasks]

        # Convert to LinAlgTask instances
        tasks = []
        for i, entry in enumerate(processed_entries):
            try:
                task = LinAlgTask.from_dataset_entry(entry)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to create task from entry {i}: {e}")
                continue

        # Validate tasks
        valid_tasks, validation_errors = validate_task_dataset(tasks)

        if validation_errors:
            logger.warning(f"Dataset validation found {len(validation_errors)} errors")

        # Analyze dataset compatibility
        compatibility = validate_dataset_compatibility(valid_tasks)
        logger.info(
            f"Dataset analysis: {compatibility['total_tasks']} tasks, "
            f"{len(compatibility['problem_types'])} problem types"
        )

    except Exception:
        logger.exception("Failed to load datasets")
        logger.info("Falling back to sample tasks for testing")
        return create_sample_tasks()
    else:
        logger.info(f"Successfully loaded {len(valid_tasks)} valid tasks")
        return valid_tasks


def validate_task_dataset(tasks: list[LinAlgTask]) -> tuple[list[LinAlgTask], list[str]]:
    """
    Validate a list of tasks and filter out invalid ones.

    Args:
        tasks: List of tasks to validate

    Returns:
        Tuple of (valid_tasks, validation_errors)
    """
    valid_tasks = []
    all_errors = []

    for i, task in enumerate(tasks):
        is_valid, errors = task.validate()
        if is_valid:
            valid_tasks.append(task)
        else:
            task_errors = [f"Task {i} ({task.user_id}): {error}" for error in errors]
            all_errors.extend(task_errors)
            logger.warning(f"Task {i} failed validation: {errors}")

    logger.info(f"Validation complete: {len(valid_tasks)}/{len(tasks)} tasks are valid")
    if all_errors:
        logger.warning(f"Found {len(all_errors)} validation errors")

    return valid_tasks, all_errors


def validate_dataset_compatibility(tasks: list[LinAlgTask]) -> dict[str, Any]:
    """
    Analyze dataset compatibility with existing generator components.

    Args:
        tasks: List of tasks to analyze

    Returns:
        Dictionary with compatibility analysis
    """
    from linalg_zero.shared.lib import get_lib_fn_names

    analysis = {
        "total_tasks": len(tasks),
        "problem_types": {},
        "difficulty_levels": {},
        "matrix_operations": {},
        "tool_usage": {},
        "compatibility_issues": [],
    }

    lib_functions = set(get_lib_fn_names())

    for task in tasks:
        # Analyze problem types
        problem_type = task.problem_type
        analysis["problem_types"][problem_type] = analysis["problem_types"].get(problem_type, 0) + 1

        # Analyze difficulty levels
        difficulty = task.difficulty_level
        analysis["difficulty_levels"][difficulty] = analysis["difficulty_levels"].get(difficulty, 0) + 1

        # Analyze matrix operations in stepwise ground truths
        try:
            stepwise = task.get_stepwise_ground_truths_parsed()
            for step in stepwise:
                if isinstance(step, dict):
                    for operation in step:
                        analysis["matrix_operations"][operation] = analysis["matrix_operations"].get(operation, 0) + 1

                        # Check compatibility with lib functions
                        if operation not in lib_functions:
                            analysis["compatibility_issues"].append(
                                f"Task {task.user_id}: Unknown operation '{operation}' not in lib functions"
                            )
        except Exception as e:
            analysis["compatibility_issues"].append(
                f"Task {task.user_id}: Failed to parse stepwise ground truths: {e}"
            )

        # Analyze tool usage
        if task.tools:
            for tool in task.tools:
                if isinstance(tool, dict) and "function" in tool:
                    tool_name = tool["function"].get("name", "unknown")
                    analysis["tool_usage"][tool_name] = analysis["tool_usage"].get(tool_name, 0) + 1

    return analysis


def create_sample_tasks() -> list[LinAlgTask]:
    """
    Create sample linear algebra tasks for testing.

    Returns:
        List of sample LinAlgTask instances
    """
    import json

    tasks = []

    # Sample determinant task
    task1 = LinAlgTask(
        user_id="sample_1",
        instruction="Calculate the determinant of matrix A = [[1, 2], [3, 4]]",
        actions=[
            Action(name="determinant", kwargs={"matrix": [[1, 2], [3, 4]]}),
            Action(name="respond", kwargs={"content": "The determinant is -2.0"}),
        ],
        outputs=["-2.0"],
        query="Calculate the determinant of matrix A = [[1, 2], [3, 4]]",
        ground_truth=json.dumps(-2.0),
        stepwise_ground_truths=json.dumps([{"determinant": -2.0}]),
        tools=None,
        difficulty_level=1,
        problem_type="determinant",
    )
    tasks.append(task1)

    # Sample transpose task
    task2 = LinAlgTask(
        user_id="sample_2",
        instruction="Find the transpose of matrix B = [[1, 2, 3], [4, 5, 6]]",
        actions=[
            Action(name="matrix_transpose", kwargs={"matrix": [[1, 2, 3], [4, 5, 6]]}),
            Action(name="respond", kwargs={"content": "The transpose is [[1, 4], [2, 5], [3, 6]]"}),
        ],
        outputs=["[[1, 4], [2, 5], [3, 6]]"],
        query="Find the transpose of matrix B = [[1, 2, 3], [4, 5, 6]]",
        ground_truth=json.dumps([[1, 4], [2, 5], [3, 6]]),
        stepwise_ground_truths=json.dumps([{"matrix_transpose": [[1, 4], [2, 5], [3, 6]]}]),
        tools=None,
        difficulty_level=1,
        problem_type="transpose",
    )
    tasks.append(task2)

    return tasks


def create_linalg_environment(
    config: RunConfig,
    tasks: list[LinAlgTask] | None = None,
    dataset_name: str | None = None,
    max_tasks: int | None = None,
    use_sample_tasks: bool = False,
) -> LinAlgEnvironment:
    """
    Create a LinAlgEnvironment with all necessary tools and tasks.

    Args:
        config: Run configuration
        tasks: List of tasks, if provided will use these directly
        dataset_name: HuggingFace dataset name to load tasks from
        max_tasks: Maximum number of tasks to load from dataset
        use_sample_tasks: Force use of sample tasks for testing

    Returns:
        Configured LinAlgEnvironment instance
    """
    # Import here to avoid circular imports
    from linalg_zero.grpo.openpipe_art.linalg_tools import get_linalg_tools

    # Load tasks based on priority: provided tasks > dataset > sample tasks
    if tasks is not None:
        logger.info(f"Using {len(tasks)} provided tasks")
    elif use_sample_tasks:
        logger.info("Using sample tasks for testing")
        tasks = create_sample_tasks()
    elif dataset_name:
        logger.info(f"Loading tasks from dataset: {dataset_name}")
        tasks = load_linalg_tasks_from_hub(
            dataset_name=dataset_name,
            split="validation",  # Use validation split for GRPO training
            max_tasks=max_tasks,
        )
    else:
        # Try to load from default GRPO dataset, fall back to prepared datasets
        logger.info("Loading tasks from default datasets")
        try:
            tasks = load_linalg_tasks_from_hub(
                dataset_name="atomwalk12/linalgzero-grpo", split="validation", max_tasks=max_tasks
            )
        except Exception:
            logger.warning("Failed to load from GRPO dataset, trying prepared datasets")
            tasks = load_linalg_tasks_from_prepared_datasets(
                use_train_split=False,  # Use validation split
                max_tasks=max_tasks,
            )

    # Get all available tools
    tools = get_linalg_tools()

    # Create and return environment
    return LinAlgEnvironment(
        tools=tools,
        tasks=tasks,
        config=config,
    )
