import json

import numpy as np
import pytest

from linalg_zero.generator.composition.components import (
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
)
from linalg_zero.generator.composition.composition import CompositeProblem, SequentialComposition
from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.generator_factories import create_linear_system_generator
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.template_engine import TemplateEngine


def create_composite_generator():
    def generator() -> Question:
        difficulty_level = DifficultyCategory.TWO_TOOL_CALLS
        problem_type = Task.COMPOSITE_SYSTEM_NORM
        topic = Topic.LINEAR_ALGEBRA

        components = [
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
            ),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_vector_b": 0},
                    "sources": {"input_vector_b": "result"},
                },
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
            ),
        ]

        config = get_problem_config(difficulty_level, topic, problem_type)
        sample_args = config.create_sample_args_for_composition(len(components))

        composite_problem = CompositeProblem(
            components=components,
            composition_strategy=SequentialComposition(),
            sample_args=sample_args,
            template_engine=TemplateEngine(),
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )

        return composite_problem.generate()

    return generator


def create_simple_atomic_generator():
    return create_linear_system_generator(difficulty=DifficultyCategory.TWO_TOOL_CALLS)


class TestLinearSystemDependency:
    """Test the three-step Linear System Dependency composite problem."""

    @pytest.fixture
    def composite_generator(self):
        """Composite question generator."""
        return create_composite_generator()

    @pytest.fixture
    def atomic_generator(self):
        """Atomic question generator."""
        return create_simple_atomic_generator()

    @pytest.fixture
    def question(self, composite_generator):
        """Generate a Linear System Dependency question (composite)."""
        return composite_generator()

    def test_composite_problem_generation(self, question):
        """Test that the composite problem generates successfully."""
        assert isinstance(question, Question)
        assert question.problem_type == Task.COMPOSITE_SYSTEM_NORM
        assert question.topic == Topic.LINEAR_ALGEBRA
        assert isinstance(question.question, str)
        assert isinstance(question.answer, str)

    def test_dependency_chain_structure(self, question):
        """Test that the problem has the correct three-step structure."""
        # The question should mention all three operations in sequence
        question_text = question.question.lower()

        # Should contain linear system solving language
        assert any(phrase in question_text for phrase in ["step 1", "step 2", "step 3"]), (
            f"Question text: {question_text}"
        )

        # Should contain matrix multiplication language
        assert any(phrase in question_text for phrase in ["matrix", "product", "*"]), f"Question text: {question_text}"

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

        # Allow for rounding differences (answers are typically rounded to 2 decimal places)
        # The verify_answers function is too strict for this case since it expects exact equality
        assert abs(calculated_norm - step3_result) < 1e-2, (
            f"Step 3 result {step3_result} should equal ||Step 2 result||_F = {calculated_norm}"
        )

    def test_multiple_generations_are_different(self, composite_generator):
        """Test that multiple generations produce different problems."""
        question1 = composite_generator()
        question2 = composite_generator()

        # Different problems should have different questions and answers
        assert question1.question != question2.question
        assert question1.answer != question2.answer

    def test_generator_configuration(self, composite_generator):
        """Test that the composite generator produces the expected problem type."""
        question = composite_generator()
        assert question.topic == Topic.LINEAR_ALGEBRA
        assert question.problem_type == Task.COMPOSITE_SYSTEM_NORM

    def test_tool_order_and_names(self, question: Question):
        """Ensure the exact tool call order and names are correct for the composite."""
        tool_names = [step["tool"] for step in question.stepwise]
        assert tool_names == [
            "solve_linear_system",
            "multiply_matrices",
            "frobenius_norm",
        ]

    def test_atomic_linear_system_generation(self, atomic_generator):
        """Atomic linear system solver generates a single-step valid question."""
        q = atomic_generator()
        assert isinstance(q, Question)
        assert q.problem_type == Task.LINEAR_SYSTEM_SOLVER
        assert q.topic == Topic.LINEAR_ALGEBRA
        # Atomic tasks use exactly one tool call
        assert q.tool_calls_required == 1

        ans = json.loads(q.answer)
        assert isinstance(ans, list) and all(isinstance(row, list) and len(row) == 1 for row in ans)
