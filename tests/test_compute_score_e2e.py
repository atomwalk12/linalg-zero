"""End-to-end tests for compute_score.py without mocks."""

from linalg_zero.grpo.compute_score import calc_reward, get_interaction_reward, get_tool_reward
from linalg_zero.grpo.verifiers.xml_parser import XMLParser


class TestGetToolRewardE2E:
    """End-to-end tests for get_tool_reward function."""

    def test_matching_float_values(self):
        """Test reward when float values match exactly."""
        score, metadata = get_tool_reward(3.14, 3.14)
        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_non_matching_float_values(self):
        """Test reward when float values don't match."""
        score, metadata = get_tool_reward(3.14, 2.71)
        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_matching_integer_values(self):
        """Test reward when integer values match."""
        score, metadata = get_tool_reward(42, 42)
        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_non_matching_integer_values(self):
        """Test reward when integer values don't match."""
        score, metadata = get_tool_reward(42, 24)
        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_matching_list_values(self):
        """Test reward when list values match."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        score, metadata = get_tool_reward(matrix, matrix)
        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_non_matching_list_values(self):
        """Test reward when list values don't match."""
        matrix1 = [[1.0, 2.0], [3.0, 4.0]]
        matrix2 = [[5.0, 6.0], [7.0, 8.0]]
        score, metadata = get_tool_reward(matrix1, matrix2)
        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_string_representations_of_numbers(self):
        """Test with string representations that should match numerically."""
        score, metadata = get_tool_reward("42", "42.0")
        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_mathematical_expressions(self):
        """Test with mathematical expressions that evaluate to the same value."""
        score, metadata = get_tool_reward("2 + 2", "4")
        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_invalid_mathematical_expressions(self):
        """Test with invalid mathematical expressions."""
        score, metadata = get_tool_reward("invalid_expr", "42")
        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_empty_values(self):
        """Test with empty values."""
        score, metadata = get_tool_reward("", "")
        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_none_values(self):
        """Test with None values - should handle gracefully."""
        score, metadata = get_tool_reward(None, None)

        assert score == 1.0
        assert metadata["reward_tool_output"] is True


class TestGetInteractionRewardE2E:
    """End-to-end tests for get_interaction_reward function."""

    def setup_method(self):
        """Setup test data for each test."""
        self.parser = XMLParser()

    def test_perfect_response_list_format(self):
        """Test with perfect response in list format."""
        completion = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "<think>I need to add 2 + 2</think><answer>4</answer>"},
        ]
        ground_truth = "4"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_perfect_response_string_format(self):
        """Test with perfect response in string format."""
        completion = "<think>I need to calculate this</think><answer>42</answer>"
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_correct_answer_wrong_format(self):
        """Test with correct answer but wrong format."""
        completion = [{"role": "assistant", "content": "The answer is 42"}]
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.0
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_wrong_answer_correct_format(self):
        """Test with wrong answer but correct format."""
        completion = [{"role": "assistant", "content": "<think>Let me calculate</think><answer>41</answer>"}]
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_partial_format_with_think_only(self):
        """Test with only think tags, no answer tags."""
        completion = [{"role": "assistant", "content": "<think>I'm thinking about this problem</think>"}]
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.0
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_multiple_assistant_messages(self):
        """Test with multiple assistant messages - should use the last one."""
        completion = [
            {"role": "user", "content": "Solve this"},
            {"role": "assistant", "content": "<think>First attempt</think><answer>wrong</answer>"},
            {"role": "user", "content": "Try again"},
            {"role": "assistant", "content": "<think>Second attempt</think><answer>42</answer>"},
        ]
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_no_assistant_messages(self):
        """Test with no assistant messages in completion."""
        completion = [{"role": "user", "content": "What is 2 + 2?"}, {"role": "tool", "content": "Tool response"}]
        ground_truth = "4"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.0
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_empty_completion_list(self):
        """Test with empty completion list."""
        completion = []
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.0
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_malformed_completion_structure(self):
        """Test with malformed completion structure."""
        completion = [{"invalid_key": "value"}, {"role": "assistant"}]
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 0.0
        assert metadata["reward_final_answer"] is False
        assert metadata["reward_response_format"] is False

    def test_mathematical_expression_in_answer(self):
        """Test with mathematical expression that should be evaluated."""
        completion = [{"role": "assistant", "content": "<think>Let me calculate</think><answer>2 + 2</answer>"}]
        ground_truth = "4"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_truncated_response(self):
        """Test with truncated response (missing closing tags)."""
        completion = "<think>I need to calculate</think><answer>42"
        ground_truth = "42"

        score, metadata = get_interaction_reward(self.parser, completion, ground_truth)

        expected_total = 0.0

        assert score == expected_total
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True


class TestCalcRewardE2E:
    """End-to-end tests for calc_reward function."""

    def test_basic_functionality(self):
        """Test basic calc_reward functionality."""
        solution_str = [{"role": "assistant", "content": "<think>Let me solve this</think><answer>42</answer>"}]
        ground_truth = "42"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser)

        assert isinstance(score, (int, float))  # noqa: UP038
        assert score == 1.2

    def test_with_additional_kwargs(self):
        """Test that additional kwargs are handled gracefully."""
        solution_str = [{"role": "assistant", "content": "<think>Calculating</think><answer>100</answer>"}]
        ground_truth = "100"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser, extra_param="test", another_param=123)

        assert score == 1.2

    def test_string_format_input(self):
        """Test with string format input instead of list."""
        solution_str = "<think>Let me work on this</think><answer>3.14</answer>"
        ground_truth = "3.14"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser)

        assert score == 1.2

    def test_zero_score_scenario(self):
        """Test scenario that should result in zero score."""
        solution_str = [{"role": "assistant", "content": "Just plain text without proper format"}]
        ground_truth = "42"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser)

        assert score == 0.0

    def test_partial_score_scenario(self):
        """Test scenario with partial score (format only)."""
        solution_str = [{"role": "assistant", "content": "<think>Thinking</think><answer>wrong_answer</answer>"}]
        ground_truth = "42"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser)

        assert score == 0.2

    def test_with_correct_answer(self):
        """Test scenario with correct answer."""
        solution_str = [{"role": "assistant", "content": "Thinking</think><answer>42</answer>"}]
        ground_truth = "42"
        parser = XMLParser()

        score = calc_reward(solution_str, ground_truth, parser)

        assert score == 1.0


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_linear_algebra_problem_solving(self):
        """Test with realistic linear algebra problem solving scenario."""

        completion = [
            {"role": "user", "content": "Calculate the determinant of [[1, 2], [3, 4]]"},
            {"role": "assistant", "content": "I need to calculate this step by step."},
            {"role": "tool", "content": "determinant([[1, 2], [3, 4]]) = -2"},
            {
                "role": "assistant",
                "content": "<think>The tool calculated the determinant as -2. Let me verify: det([[1,2],[3,4]]) = 1*4 - 2*3 = 4 - 6 = -2. That's correct.</think><answer>-2</answer>",
            },
        ]
        ground_truth = "-2"
        parser = XMLParser()

        score, metadata = get_interaction_reward(parser, completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_tool_calculation_verification(self):
        """Test tool output verification scenario."""

        ground_truth_result = [[4.0, 6.0], [10.0, 12.0]]
        tool_output = [[4.0, 6.0], [10.0, 12.0]]

        score, metadata = get_tool_reward(ground_truth_result, tool_output)

        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_wrong_tool_calculation(self):
        """Test scenario with wrong tool calculation."""
        ground_truth_result = [[4.0, 6.0], [10.0, 12.0]]
        wrong_tool_output = [[1.0, 2.0], [3.0, 4.0]]

        score, metadata = get_tool_reward(ground_truth_result, wrong_tool_output)

        assert score == 0.0
        assert metadata["reward_tool_output"] is True

    def test_complete_problem_solving_flow(self):
        """Test complete problem solving flow with calc_reward."""

        solution = [
            {"role": "user", "content": "What is 5 factorial?"},
            {"role": "assistant", "content": "Let me calculate this."},
            {"role": "tool", "content": "factorial(5) = 120"},
            {
                "role": "assistant",
                "content": "<think>The factorial of 5 is 5 * 4 * 3 * 2 * 1 = 120. The tool gave the correct answer.</think><answer>120</answer>",
            },
        ]
        ground_truth = "120"

        final_score = calc_reward(solution, ground_truth, XMLParser())

        assert final_score == 1.2

    def test_mathematical_equivalence(self):
        """Test mathematical equivalence in answers."""
        completion = [{"role": "assistant", "content": "<think>Converting to decimal</think><answer>0.5</answer>"}]
        ground_truth = "1/2"

        score, metadata = get_interaction_reward(XMLParser(), completion, ground_truth)

        assert score == 1.2
        assert metadata["reward_final_answer"] is True
        assert metadata["reward_response_format"] is True

    def test_complex_nested_data_structures(self):
        """Test with complex nested data structures (3D matrices)."""
        ground_truth = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tool_output = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

        score, metadata = get_tool_reward(ground_truth, tool_output)

        assert score == 1.0
        assert metadata["reward_tool_output"] is True

    def test_floating_point_precision_issues(self):
        """Test handling of floating point precision issues."""

        ground_truth = 0.1 + 0.2
        tool_output = 0.3

        score, metadata = get_tool_reward(ground_truth, tool_output)

        assert isinstance(score, (float))
        assert metadata["reward_tool_output"] is True
