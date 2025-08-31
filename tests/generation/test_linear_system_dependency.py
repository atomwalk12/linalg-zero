import json

import numpy as np
import pytest

from linalg_zero.generator.composition.components import (
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
)
from linalg_zero.generator.composition.composition import SequentialComposition
from linalg_zero.generator.generator_factories import create_linear_system_generator
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.registry import (
    FactoryRegistry,
)


def create_registry_for_composite_linear_system_dependency() -> FactoryRegistry:
    """Create a registry that only registers the linear system dependency composite.

    This avoids relying on the evolving default registry and hard-codes the
    exact composite problem type required by tests.
    """
    registry = FactoryRegistry()

    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
            ),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={"is_independent": False, "input_index": 0},
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={"is_independent": False, "input_index": 1},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.MEDIUM,
    )

    return registry


def create_registry_for_atomic_linear_system_solver() -> FactoryRegistry:
    """Create a registry with only the atomic linear system solver task registered."""
    registry = FactoryRegistry()
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.LINEAR_SYSTEM_SOLVER,
        create_linear_system_generator(difficulty=DifficultyCategory.MEDIUM),
    )
    return registry


class TestLinearSystemDependency:
    """Test the three-step Linear System Dependency composite problem."""

    @pytest.fixture
    def registry_composite(self):
        """Hard-typed registry with only the linear system dependency composite."""
        return create_registry_for_composite_linear_system_dependency()

    @pytest.fixture
    def registry_atomic(self):
        """Hard-typed registry with only the atomic linear system solver."""
        return create_registry_for_atomic_linear_system_solver()

    @pytest.fixture
    def question(self, registry_composite: FactoryRegistry):
        """Generate a Linear System Dependency question (composite)."""
        factory = registry_composite.get_factory(Topic.LINEAR_ALGEBRA, Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY)
        return factory()

    def test_composite_problem_generation(self, question):
        """Test that the composite problem generates successfully."""
        assert isinstance(question, Question)
        assert question.problem_type == Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY
        assert question.topic == Topic.LINEAR_ALGEBRA
        assert isinstance(question.question, str)
        assert isinstance(question.answer, str)

    def test_dependency_chain_structure(self, question):
        """Test that the problem has the correct three-step structure."""
        # The question should mention all three operations in sequence
        question_text = question.question.lower()

        # Should contain linear system solving language
        assert any(phrase in question_text for phrase in ["what is", "determine", "calculate"]), (
            f"Question text: {question_text}"
        )

        # Should contain matrix multiplication language
        assert "*" in question_text, f"Question text: {question_text}"

        # Should contain Frobenius norm language
        assert any(word in question_text for word in ["frobenius", "norm", "||"]), f"Question text: {question_text}"

    def test_answer_format(self, question):
        """Test that the answer has three tool results."""
        import json

        answer = json.loads(question.answer)

        # Should have exactly 3 tool results
        assert "tool_1" in answer  # Linear system solution
        assert "tool_2" in answer  # Matrix multiplication result
        assert "tool_3" in answer  # Frobenius norm result
        assert len([k for k in answer if k.startswith("tool_")]) == 3

    def test_mathematical_dependency_chain(self, question):
        """Test that the mathematical dependency chain is correct."""
        import json

        answer = json.loads(question.answer)

        step1_result = answer["tool_1"]  # Should be solution vector
        step2_result = answer["tool_2"]  # Should be matrix multiplication result
        step3_result = answer["tool_3"]  # Should be Frobenius norm (scalar)

        # Step 1: Should produce a vector (list of lists with single column)
        assert isinstance(step1_result, list)
        assert all(isinstance(row, list) and len(row) == 1 for row in step1_result)

        # Step 2: Should produce a vector (result of matrix * vector)
        assert isinstance(step2_result, list)
        assert all(isinstance(row, list) and len(row) == 1 for row in step2_result)

        # Step 3: Should produce a scalar (Frobenius norm)
        assert isinstance(step3_result, (int, float))
        assert step3_result >= 0  # Norms are non-negative

    def test_step_dependencies_mathematical_consistency(self, question):
        """Test that steps are mathematically consistent with each other."""
        import json

        answer = json.loads(question.answer)

        step2_result = np.array(answer["tool_2"])
        step3_result = answer["tool_3"]

        # Step 3 should be the Frobenius norm of Step 2's result
        calculated_norm = np.linalg.norm(step2_result, "fro")

        # Allow for small floating point differences
        assert abs(calculated_norm - step3_result) < 1e-6, (
            f"Step 3 result {step3_result} should equal ||Step 2 result||_F = {calculated_norm}"
        )

    def test_multiple_generations_are_different(self, registry_composite: FactoryRegistry):
        """Test that multiple generations produce different problems."""
        factory = registry_composite.get_factory(Topic.LINEAR_ALGEBRA, Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY)

        question1 = factory()
        question2 = factory()

        # Different problems should have different questions and answers
        assert question1.question != question2.question
        assert question1.answer != question2.answer

    def test_registry_configuration(self, registry_composite: FactoryRegistry):
        """Test that the registry correctly registers the composite problem."""
        topics = registry_composite.list_topics()
        assert Topic.LINEAR_ALGEBRA in topics

        problem_types = registry_composite.list_problem_types(Topic.LINEAR_ALGEBRA)
        assert Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY in problem_types

    def test_tool_order_and_names(self, question: Question):
        """Ensure the exact tool call order and names are correct for the composite."""
        tool_names = [step["tool"] for step in question.stepwise]
        assert tool_names == [
            "solve_linear_system",
            "multiply_matrices",
            "frobenius_norm",
        ]

    def test_atomic_linear_system_generation(self, registry_atomic: FactoryRegistry):
        """Atomic linear system solver generates a single-step valid question."""
        factory = registry_atomic.get_factory(Topic.LINEAR_ALGEBRA, Task.LINEAR_SYSTEM_SOLVER)
        q = factory()
        assert isinstance(q, Question)
        assert q.problem_type == Task.LINEAR_SYSTEM_SOLVER
        assert q.topic == Topic.LINEAR_ALGEBRA
        # Atomic tasks use exactly one tool call
        assert q.tool_calls_required == 1

        ans = json.loads(q.answer)
        assert isinstance(ans, list) and all(isinstance(row, list) and len(row) == 1 for row in ans)
