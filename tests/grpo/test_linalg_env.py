"""
Pytest tests for LinAlgEnvironment implementation.
"""

import pytest

from linalg_zero.grpo.openpipe_art.base_types import Action
from linalg_zero.grpo.openpipe_art.data_types import RunConfig
from linalg_zero.grpo.openpipe_art.linalg_env import LinAlgEnvironment, create_linalg_environment


@pytest.fixture
def config() -> RunConfig:
    """Create a test configuration."""
    return RunConfig(
        model_provider="test",
        user_model_provider="test",
        model="test-model",
        user_model="test-user-model",
        env="linalg",
        user_strategy="llm",
    )


@pytest.fixture
def env(config: RunConfig) -> LinAlgEnvironment:
    """Create a test environment."""
    return create_linalg_environment(config)


def test_environment_creation(env: LinAlgEnvironment) -> None:
    """Test that we can create the environment successfully."""
    assert env is not None
    assert len(env.tasks) > 0
    assert len(env.tools_map) > 0

    # Check that expected tools are available
    expected_tools = [
        "determinant",
        "matrix_transpose",
        "matrix_cofactor",
        "frobenius_norm",
        "matrix_rank",
        "matrix_trace",
    ]
    for tool_name in expected_tools:
        assert tool_name in env.tools_map

    # Check that tasks have the new structure
    first_task = env.tasks[0]
    assert hasattr(first_task, "query")
    assert hasattr(first_task, "ground_truth")
    assert hasattr(first_task, "stepwise_ground_truths")


def test_environment_reset(env: LinAlgEnvironment) -> None:
    """Test environment reset functionality."""
    # Reset with first task
    reset_response = env.reset(task_index=0)

    assert reset_response is not None
    assert env.current_task is not None
    assert env.session_id is not None
    assert env.get_step_count() == 0
    assert len(env.actions) == 0
    assert len(env.tool_call_history) == 0


def test_tool_execution(env: LinAlgEnvironment) -> None:
    """Test tool execution."""
    # Reset environment first
    env.reset(task_index=0)

    # Test determinant tool
    action = Action(name="determinant", kwargs={"matrix": [[1, 2], [3, 4]]})

    response = env.step(action)

    assert response is not None
    assert response.observation == "-2.0"  # Expected determinant result
    assert not response.done  # Should not be done after one tool call
    assert env.get_step_count() == 1
    assert len(env.tool_call_history) == 1


def test_matrix_transpose_tool(env: LinAlgEnvironment) -> None:
    """Test matrix transpose tool specifically."""
    env.reset(task_index=0)

    action = Action(name="matrix_transpose", kwargs={"matrix": [[1, 2, 3], [4, 5, 6]]})

    response = env.step(action)

    assert response is not None
    assert "[[1, 4], [2, 5], [3, 6]]" in response.observation
    assert env.get_step_count() == 1


def test_state_management(env: LinAlgEnvironment) -> None:
    """Test state management functionality."""
    env.reset(task_index=0)

    # Get environment state
    state = env.get_environment_state()
    task_info = env.get_task_info()
    matrices = env.get_current_matrices()

    # Check state structure
    assert isinstance(state, dict)
    assert "session_id" in state
    assert "episode_step_count" in state
    assert "max_steps" in state

    # Check task info
    assert isinstance(task_info, dict)
    assert "user_id" in task_info
    assert "instruction" in task_info

    # Check matrices
    assert isinstance(matrices, dict)


def test_max_steps_limit(env: LinAlgEnvironment) -> None:
    """Test that environment respects max steps limit."""
    env.reset(task_index=0)
    env.set_max_steps(2)  # Set very low limit for testing

    # Take first step
    action1 = Action(name="determinant", kwargs={"matrix": [[1, 2], [3, 4]]})
    response1 = env.step(action1)
    assert not response1.done

    # Take second step - should hit limit
    action2 = Action(name="matrix_transpose", kwargs={"matrix": [[1, 2], [3, 4]]})
    response2 = env.step(action2)
    assert response2.done
    assert "Maximum steps" in response2.observation


def test_invalid_tool_call(env: LinAlgEnvironment) -> None:
    """Test handling of invalid tool calls."""
    env.reset(task_index=0)

    # Call non-existent tool
    action = Action(name="nonexistent_tool", kwargs={})
    response = env.step(action)

    assert "Unknown action" in response.observation
    assert not response.done


def test_tool_error_handling(env: LinAlgEnvironment) -> None:
    """Test handling of tool execution errors."""
    env.reset(task_index=0)

    # Call determinant with invalid matrix (non-square)
    action = Action(
        name="determinant",
        kwargs={"matrix": [[1, 2, 3], [4, 5, 6]]},  # 2x3 matrix, not square
    )

    response = env.step(action)

    assert "Error:" in response.observation
    assert not response.done


def test_intermediate_results_storage(env: LinAlgEnvironment) -> None:
    """Test that intermediate results are stored correctly."""
    env.reset(task_index=0)

    # Execute a tool
    action = Action(name="determinant", kwargs={"matrix": [[1, 2], [3, 4]]})
    env.step(action)

    # Check intermediate results
    results = env.get_intermediate_results()
    assert len(results) > 0
    assert "latest_determinant" in results
    assert results["latest_determinant"] == "-2.0"


def test_multiple_tool_calls(env: LinAlgEnvironment) -> None:
    """Test multiple sequential tool calls."""
    env.reset(task_index=0)

    # First tool call
    action1 = Action(name="determinant", kwargs={"matrix": [[1, 2], [3, 4]]})
    response1 = env.step(action1)
    assert not response1.done

    # Second tool call
    action2 = Action(name="matrix_transpose", kwargs={"matrix": [[1, 2], [3, 4]]})
    response2 = env.step(action2)
    assert not response2.done

    # Check state
    assert env.get_step_count() == 2
    assert len(env.tool_call_history) == 2

    # Check both results are stored
    results = env.get_intermediate_results()
    assert "latest_determinant" in results
    assert "latest_matrix_transpose" in results


def test_task_loading_and_validation() -> None:
    """Test task loading and validation functionality."""
    from linalg_zero.grpo.openpipe_art.linalg_env import create_sample_tasks, validate_task_dataset

    # Test sample task creation
    tasks = create_sample_tasks()
    assert len(tasks) > 0

    # Test task validation
    valid_tasks, errors = validate_task_dataset(tasks)
    assert len(valid_tasks) == len(tasks)  # Sample tasks should be valid
    assert len(errors) == 0

    # Test task structure
    task = tasks[0]
    assert hasattr(task, "query")
    assert hasattr(task, "ground_truth")
    assert hasattr(task, "stepwise_ground_truths")

    # Test validation methods
    is_valid, task_errors = task.validate()
    assert is_valid
    assert len(task_errors) == 0

    # Test matrix extraction
    matrix_data = task.extract_matrix_data()
    assert isinstance(matrix_data, dict)


def test_environment_with_sample_tasks() -> None:
    """Test environment creation with sample tasks."""
    from linalg_zero.grpo.openpipe_art.data_types import RunConfig
    from linalg_zero.grpo.openpipe_art.linalg_env import create_linalg_environment

    config = RunConfig(
        model_provider="test",
        user_model_provider="test",
        model="test-model",
        user_model="test-user-model",
        env="linalg",
        user_strategy="llm",
    )

    # Test with sample tasks
    env = create_linalg_environment(config, use_sample_tasks=True)
    assert env is not None
    assert len(env.tasks) > 0

    # Test environment functionality
    reset_response = env.reset(task_index=0)
    assert reset_response is not None
    assert env.current_task is not None

    # Test matrix data extraction
    matrices = env.get_current_matrices()
    assert isinstance(matrices, dict)
