import json
from types import TracebackType
from typing import Any

import sympy

from linalg_zero.generator.models import ComponentResult, DifficultyCategory
from linalg_zero.shared.types import LibTypes


class ProblemContext:
    """
    Context manager for state information around the resolution process.
    """

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, step_counter: int):
        self.entropy = entropy
        self.difficulty_level = difficulty_level
        self.used_entropy = 0.0
        self.tool_calls_count = 0
        self.stepwise_results: list[dict[str, Any]] = []
        self.golden_result: dict[str, str] = {}
        self._step_counter = step_counter
        self.constraints: dict[str, Any] = {}

    def __enter__(self) -> "ProblemContext":
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: TracebackType) -> None:
        pass

    def record_entropy_usage(self, amount: float) -> None:
        """
        Record entropy usage for tracking problem complexity.
        """
        self.used_entropy += amount

    def _prepare_verification_data(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare verification data by extracting dependencies and input fields."""
        verification = {}

        # Handle dependencies
        dependent_on = input_data.pop("dependent_on", None)
        if dependent_on is not None:
            verification["dependent_on"] = dependent_on

        # Extract and JSON-encode all input_* fields
        for key in list(input_data.keys()):
            if key.startswith("input_"):
                verification[key] = json.dumps(input_data.pop(key))

        # Add generator type and remaining input data
        verification["generator_type"] = input_data.pop("generator_type")
        verification["input"] = json.dumps(input_data)

        return verification

    def record_tool_call(
        self,
        function_name: str,
        result: LibTypes,
        input_data: dict[str, Any],
        is_final: bool = False,
    ) -> str:
        """
        Record a tool call with its result. It tracks the dependencies between
        steps which will later be used to verify correctness during GRPO.
        """
        self.tool_calls_count += 1
        step_id = str(self._step_counter)

        if result is not None:
            result_json = json.dumps(result)
            step_data = {
                "tool": function_name,
                "result": result_json,
                "step_id": step_id,
                "verification": self._prepare_verification_data(input_data),
            }

            if is_final:
                self.golden_result = {"final_answer": result_json, "from_step_id": step_id}

            self.stepwise_results.append(step_data)

        self._step_counter += 1
        return step_id


class CompositionContext(ProblemContext):
    """
    Extends the base ProblemContext to support shared state and global variables
    across composed problem components.
    """

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, step_counter: int):
        super().__init__(entropy, difficulty_level, step_counter)
        self.shared_state: dict[str, Any] = {}
        self.component_results: list[ComponentResult] = []
        self.global_variables: dict[str, sympy.Symbol] = {}

    def share_variable(self, name: str, symbol: sympy.Symbol) -> None:
        """Make a variable available to other components."""
        self.global_variables[name] = symbol

    def get_shared_variable(self, name: str) -> sympy.Symbol | None:
        """Retrieve a shared variable from another component."""
        return self.global_variables.get(name)

    def update_shared_state(self, key: str, value: Any) -> None:
        """Update shared state accessible by all components."""
        self.shared_state[key] = value

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared state value."""
        return self.shared_state.get(key, default)

    def record_component_result(self, result: ComponentResult) -> None:
        """Record the result of a component execution."""
        self.component_results.append(result)

        # Update entropy usage and validate budget
        self.used_entropy += result.entropy_consumed
        if self.used_entropy > self.entropy + 1e-12:
            raise ValueError(f"Entropy budget exceeded: used {self.used_entropy:.3f}, available {self.entropy:.3f}")

        self.tool_calls_count += result.tool_calls_used

        # Update shared state
        for key, value in result.context_updates.items():
            self.update_shared_state(key, value)

        # Register shared variables
        for var in result.shared_variables:
            self.share_variable(var, var)
