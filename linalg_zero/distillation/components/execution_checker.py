"""
Math-verify based semantic checker that works with the distillation pipeline output.
Uses the math-verify package to compare generated results against ground truth from dataset.
"""

import re
from typing import Any

from distilabel.pipeline import Pipeline
from distilabel.steps import StepInput, StepOutput
from distilabel.steps.base import Step
from distilabel.typing import StepColumns
from math_verify import parse, verify
from typing_extensions import override


class MathVerifySemanticChecker(Step):
    """
    Semantic checker using math-verify package for verification.

    Takes ground truth from the dataset and uses math-verify to compare
    generated results against the expected answers.

    Input columns:
        - messages (list[dict]): Conversation messages from pipeline
        - ground_truth (str): Expected correct result from dataset

    Output columns:
        - final_result_correct (bool): Whether final result matches ground truth
        - keep_row_after_semantic_check (bool): Whether to keep this row
        - verification_details (str): Detailed verification information
        - extracted_result (str): What was extracted from the generated response
    """

    @property
    def inputs(self) -> StepColumns:
        return {
            "messages": True,
            "ground_truth": True,
        }

    @property
    def outputs(self) -> StepColumns:
        return [
            "final_result_correct",
            "keep_row_after_semantic_check",
            "verification_details",
            "extracted_result",
        ]

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """Process batch of message sequences for verification."""
        # Since this step has only one previous step, inputs[0] contains our data
        input_batch = inputs[0]
        outputs = []

        for input_data in input_batch:
            # Skip verification if marked by upstream filter
            if input_data.get("skip_downstream_processing", False):
                skip_output = input_data.copy()
                skip_output.update({
                    "final_result_correct": False,
                    "keep_row_after_semantic_check": False,
                    "verification_details": "Skipped due to upstream failure",
                    "extracted_result": None,
                })
                outputs.append(skip_output)
                continue

            try:
                result = self._verify_with_math_verify(input_data)
                outputs.append(result)
            except Exception as e:
                # Handle errors gracefully
                error_output = input_data.copy()
                error_output.update({
                    "final_result_correct": False,
                    "keep_row_after_semantic_check": False,
                    "verification_details": f"Verification error: {e!s}",
                    "extracted_result": None,
                })
                outputs.append(error_output)

        yield outputs

    def _verify_with_math_verify(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Verify generated result against ground truth using math-verify.


        """
        messages = input_data["messages"]
        ground_truth = input_data["ground_truth"]

        # Extract result from <RESULT></RESULT> tags in messages
        extracted_result = self._extract_result_from_messages(messages)

        if extracted_result is None:
            # No result found in messages
            result_correct = False
            details = "No result found in <RESULT></RESULT> tags"
        else:
            # Use math-verify to compare
            try:
                # Parse both results
                ground_truth_parsed = parse(ground_truth)
                extracted_parsed = parse(extracted_result)

                # Verify using math-verify
                result_correct = verify(ground_truth_parsed, extracted_parsed)

                if result_correct:
                    details = (
                        f"Math-verify: PASS - Generated '{extracted_result}' matches ground truth '{ground_truth}'"
                    )
                else:
                    details = f"Math-verify: FAIL - Generated '{extracted_result}' != ground truth '{ground_truth}'"

            except Exception as e:
                # Math-verify parsing/comparison failed
                result_correct = False
                details = f"Math-verify error: {e!s} - Generated: '{extracted_result}', Ground truth: '{ground_truth}'"

        # Update input with results
        output_data = input_data.copy()
        output_data.update({
            "final_result_correct": result_correct,
            "keep_row_after_semantic_check": result_correct,
            "verification_details": details,
            "extracted_result": extracted_result,
        })

        return output_data

    def _extract_result_from_messages(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract final result from <RESULT></RESULT> tags in last assistant message."""
        # Look for the last assistant message with content
        for message in reversed(messages):
            if message.get("role") == "assistant" and "content" in message:
                content = message["content"]
                if content:
                    # Extract from <RESULT> tags
                    match = re.search(r"<RESULT>(.*?)</RESULT>", content, re.DOTALL)
                    if match:
                        return match.group(1).strip()

        return None


if __name__ == "__main__":
    with Pipeline() as pipeline:
        checker = MathVerifySemanticChecker()

        # Test with sample data
        test_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "The answer is **4**.\n\n<RESULT>4</RESULT>"},
                ],
                "ground_truth": "4",
            },
            {
                "messages": [
                    {"role": "user", "content": "What is the determinant of [[2, 1], [3, 4]]?"},
                    {"role": "assistant", "content": "The determinant is **5**.\n\n<RESULT>5</RESULT>"},
                ],
                "ground_truth": "5",
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 1/2 + 1/3?"},
                    {"role": "assistant", "content": "The result is **5/6**.\n\n<RESULT>5/6</RESULT>"},
                ],
                "ground_truth": "5/6",
            },
        ]

        results = list(checker.process(test_data))

    print("Math-Verify Semantic Checker Results:")
    print("=" * 60)
    for batch in results:
        for i, result in enumerate(batch):
            print(f"\nTest {i + 1}:")
            print(f"  Extracted: '{result.get('extracted_result')}'")
            print(f"  Ground Truth: '{result.get('ground_truth')}'")
            print(f"  Correct: {result['final_result_correct']}")
            print(f"  Keep Row: {result['keep_row_after_semantic_check']}")
            print(f"  Details: {result['verification_details']}")
            print("-" * 40)
