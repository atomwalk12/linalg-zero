from abc import abstractmethod
from typing import Any

import sympy
from sympy import Float, Integer, Rational

from linalg_zero.generator.composition.composition import (
    ComponentResult,
    CompositionContext,
    ProblemComponent,
)
from linalg_zero.generator.models import Task, Topic
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate, SympyProblemGenerator
from linalg_zero.generator.sympy.generators.determinant_generator import (
    DeterminantGenerator,
    DeterminantGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import (
    FrobeniusNormGenerator,
    FrobeniusNormGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.linear_system_generator import (
    LinearSystemGenerator,
    LinearSystemGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_rank_generator import (
    MatrixRankGenerator,
    MatrixRankGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_trace_generator import (
    MatrixTraceGenerator,
    MatrixTraceGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import (
    MatrixTransposeGenerator,
    MatrixTransposeGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixMatrixMultiplicationGeneratorDependent,
    MatrixVectorMultiplicationGenerator,
    MatrixVectorMultiplicationGeneratorDependent,
)


class SympyGeneratorWrapperComponent(ProblemComponent):
    """Generic base class for wrapping sympy generators in the composition system."""

    def __init__(
        self,
        name: Task,
        generator_class: type[SympyProblemGenerator],
        component_type: Task,
        topic: Topic,
        context_update_mapping: dict[str, str],
        constraints: dict[str, Any],
        gen_constraints: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.constraints = constraints
        self.gen_constraints = gen_constraints or {}
        self.is_independent = self.constraints.pop("is_independent")
        self.generator_class = generator_class
        self.component_type = component_type
        self.topic = topic
        self.context_update_mapping = context_update_mapping

    def get_generator_params(self, context: CompositionContext, input_names: list[str]) -> dict[str, Any]:
        """Extract previous component results to use as inputs."""
        if not self.is_independent:
            params = {}
            input_indices = self.constraints["input_indices"]

            # Validate we have indices for all input names
            for input_name in input_names:
                if input_name not in input_indices:
                    raise ValueError(f"Missing input_index for input '{input_name}'")

                input_index = input_indices[input_name]
                previous_result = context.component_results[input_index]
                if not hasattr(previous_result.template, "sympy_solution"):
                    raise ValueError(f"Previous component result has no sympy_solution: {previous_result}")

                # Get the result from previous computation
                previous_sol = previous_result.template.sympy_solution
                self._validate_dependent_input(previous_sol)

                # Add to params
                params[input_name] = previous_sol
                params[f"{input_name}_index"] = input_index

            return params
        return {}

    def _get_input_validation_spec(self) -> dict[str, bool]:
        """Subclasses may override to declare constraints for dependent input.

        Supported flags:
        - require_matrix: input must be a sympy.Matrix
        - non_empty: rows > 0 and cols > 0
        - column_vector: cols == 1
        - square: rows == cols
        - numeric_only: all elements are numeric (Integer, Float, Rational)
        """
        return {}

    def _validate_dependent_input(self, value: Any) -> None:
        """Validate dependent input according to subclass spec."""

        spec = self._get_input_validation_spec()
        if not spec:
            return

        is_matrix = isinstance(value, sympy.Matrix)
        if spec.get("require_matrix", False) and not is_matrix:
            raise TypeError(f"Expected dependent input to be a sympy Matrix, got {type(value)}")
        if not is_matrix:
            raise TypeError(f"Dependent input must be a sympy Matrix-like with shape, got {type(value)}")

        rows, cols = value.shape

        if spec.get("non_empty", False) and (rows == 0 or cols == 0):
            raise ValueError(f"Dependent input matrix cannot be empty, got shape {value.shape}")

        if spec.get("column_vector", False) and cols != 1:
            raise ValueError(f"Dependent input must be a column vector with shape (n, 1), got shape {value.shape}")

        if spec.get("square", False) and rows != cols:
            raise ValueError(f"Dependent input must be square, got shape {value.shape}")

        if spec.get("numeric_only", False) and not all(
            isinstance(element, (Integer, Float, Rational)) for element in value
        ):
            raise ValueError("Dependent input must contain only numeric elements")

    @abstractmethod
    def get_input_name(self) -> list[str]:
        pass

    def generate(self, context: CompositionContext) -> ComponentResult:
        # This context is used for communication and state tracking
        problem_context = ProblemContext(
            entropy=context.entropy, difficulty_level=context.difficulty_level, step_counter=context._step_counter
        )

        # Get any additional parameters for parameterized generation
        additional_params = self.get_generator_params(context, self.get_input_name())
        additional_params["constraints"] = self.constraints
        additional_params["gen_constraints"] = self.gen_constraints or {}

        # Now, we perform the 3 key steps involved in component generation
        generator: SympyProblemGenerator = self.generator_class(
            difficulty_level=context.difficulty_level,
            problem_type=self.component_type,
            topic=self.topic,
            entropy=problem_context.entropy,
            **additional_params,
        )
        template: ProblemTemplate = generator.generate_mathematical_content(problem_context)
        generator.verify_problem(template)

        # Transfer the state of the problem context to the new problem template
        formatted_template = ProblemTemplate(
            expression=template.expression,
            variables=template.variables,
            sympy_solution=template.sympy_solution,
            lib_result=template.lib_result,
            question_templates=template.question_templates,
            context_info={
                **template.context_info,
            },
            difficulty_markers=template.difficulty_markers,
            difficulty=template.difficulty,
        )

        # Build context updates using the mapping
        context_updates = {}
        for update_key, template_key in self.context_update_mapping.items():
            if template_key == "sympy_solution":
                context_updates[f"{self.name}_{update_key}"] = template.sympy_solution
            else:
                context_updates[f"{self.name}_{update_key}"] = template.context_info[template_key]

        context.stepwise_results.extend(problem_context.stepwise_results)
        context.golden_result.update(problem_context.golden_result)
        context._step_counter = problem_context._step_counter

        return ComponentResult(
            template=formatted_template,
            context_updates=context_updates,
            entropy_consumed=problem_context.used_entropy,
            tool_calls_used=problem_context.tool_calls_count,
            generator=generator,
        )


class MatrixVectorMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixVectorMultiplicationGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = (
            MatrixVectorMultiplicationGenerator if is_independent else MatrixVectorMultiplicationGeneratorDependent
        )
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.MATRIX_VECTOR_MULTIPLICATION,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "vector": "vector", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_vector_b"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "column_vector": True}


class MatrixMatrixMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixMatrixMultiplicationGeneratorDependent."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = (
            MatrixVectorMultiplicationGenerator if is_independent else MatrixMatrixMultiplicationGeneratorDependent
        )
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.MATRIX_VECTOR_MULTIPLICATION,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "vector": "vector", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix_A", "input_matrix_B"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}


class LinearSystemSolverWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the LinearSystemGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = LinearSystemGenerator if is_independent else LinearSystemGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.LINEAR_SYSTEM_SOLVER,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={
                "matrix_A": "matrix_A",
                "x_symbols": "x_symbols",
                "target_b": "target_b",
                "solution": "sympy_solution",
            },
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_vector_b"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "column_vector": True}


class FrobeniusNormWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the FrobeniusNormGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = FrobeniusNormGenerator if is_independent else FrobeniusNormGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.FROBENIUS_NORM,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}


class DeterminantWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the DeterminantGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = DeterminantGenerator if is_independent else DeterminantGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True}


class RankWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixRankGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixRankGenerator if is_independent else MatrixRankGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.MATRIX_RANK,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "numeric_only": True}


class TransposeWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixTransposeGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixTransposeGenerator if is_independent else MatrixTransposeGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.MATRIX_TRANSPOSE,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True}


class TraceWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixTraceGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        constraints = kwargs["constraints"]
        is_independent = constraints["is_independent"]
        generator_cls = MatrixTraceGenerator if is_independent else MatrixTraceGeneratorDependent
        super().__init__(
            name=name,
            generator_class=generator_cls,
            component_type=Task.MATRIX_TRACE,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "result": "sympy_solution"},
            **kwargs,
        )

    def get_input_name(self) -> list[str]:
        return ["input_matrix"]

    def _get_input_validation_spec(self) -> dict[str, bool]:
        return {"require_matrix": True, "non_empty": True, "square": True}
