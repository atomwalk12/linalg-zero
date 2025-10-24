"""
Linear Algebra Environment for GRPO training.

This module implements an environment for training language models
on linear algebra problems with tool calling capabilities.
"""

import random
import uuid
from collections.abc import Callable
from typing import Any

from linalg_zero.grpo.openpipe_art.data_types import RunConfig

from .base_env import Env, Tool
from .base_types import (
    RESPOND_ACTION_NAME,
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    Task,
)


class LinAlgTask(Task):
    """Linear algebra task with problem statement and expected solution."""

    matrix_data: dict[str, list[list[float]]]
    expected_result: float | int | list[list[float]]
    difficulty_level: int = 1
    problem_type: str = "general"


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

        # Reset all environment state
        self._reset_episode_state()

        # Load fresh data
        self.data = self.data_load_func()

        # Add task-specific matrix data to environment data
        if hasattr(self.current_task, "matrix_data"):
            self.data.update(self.current_task.matrix_data)

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
        if self.current_task and hasattr(self.current_task, "matrix_data"):
            return self.current_task.matrix_data
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
            "matrix_data_keys": list(self.current_task.matrix_data.keys())
            if hasattr(self.current_task, "matrix_data")
            else [],
        }


def create_sample_tasks() -> list[LinAlgTask]:
    """
    Create sample linear algebra tasks for testing.

    Returns:
        List of sample LinAlgTask instances
    """
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
        matrix_data={"A": [[1, 2], [3, 4]]},
        expected_result=-2.0,
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
        matrix_data={"B": [[1, 2, 3], [4, 5, 6]]},
        expected_result=[[1, 4], [2, 5], [3, 6]],
        difficulty_level=1,
        problem_type="transpose",
    )
    tasks.append(task2)

    return tasks


def create_linalg_environment(config: RunConfig, tasks: list[LinAlgTask] | None = None) -> LinAlgEnvironment:
    """
    Create a LinAlgEnvironment with all necessary tools and tasks.

    Args:
        config: Run configuration
        tasks: List of tasks, creates sample tasks if None

    Returns:
        Configured LinAlgEnvironment instance
    """
    # Import here to avoid circular imports
    from linalg_zero.grpo.openpipe_art.linalg_tools import get_linalg_tools

    # Use sample tasks if none provided
    if tasks is None:
        tasks = create_sample_tasks()

    # Get all available tools
    tools = get_linalg_tools()

    # Create and return environment
    return LinAlgEnvironment(
        tools=tools,
        tasks=tasks,
        config=config,
    )
