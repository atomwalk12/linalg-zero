import json
from typing import Any

from distilabel.steps.base import StepInput
from distilabel.steps.tasks import APIGenExecutionChecker
from distilabel.steps.tasks.apigen.utils import (
    execute_from_response,
)
from distilabel.typing import StepColumns, StepOutput
from typing_extensions import override


class LinAlgZeroExecutionChecker(APIGenExecutionChecker):
    def _parse_arguments(self, arguments: dict[str, Any], previous_results: list[Any]) -> dict[str, Any]:
        """Parse arguments, handling result references."""
        parsed_args = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("[result_of_call_"):
                index = int(value.split("_")[-1][:-1])
                parsed_args[key] = previous_results[index]
            else:
                parsed_args[key] = value
        return parsed_args

    @property
    @override
    def inputs(self) -> StepColumns:
        """The inputs for the task are those found in the original dataset."""
        return ["generation"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        """Checks the answer to see if it can be executed.
        Captures the possible errors and returns them.

        If a single example is provided, it is copied to avoid raising an error.

        Args:
            inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """

        # TODO: this function is fairly long, need to evaluate whether a refactor is necessary.

        for _input in inputs:
            output = []
            if _input["generation"]:
                try:
                    answers = json.loads(_input["generation"])
                except json.JSONDecodeError:
                    self._logger.exception(f"Answers are not valid JSON: {_input['generation']}")
                    _input.update(**{
                        "keep_row_after_execution_check": False,
                        "execution_result": [f"Answers are not valid JSON: {_input['generation']}"],
                    })
                    continue
            else:
                _input.update(**{
                    "keep_row_after_execution_check": False,
                    "execution_result": ["No answers were provided."],
                })
                continue

            tool_results: list[dict[str, Any]] = []
            previous_results: list[Any] = []
            for wrapper in answers:
                answer = wrapper["function"]
                if answer is None:
                    output.append({
                        "keep": False,
                        "execution_result": "Nothing was generated for this answer.",
                    })
                    continue

                function_name = answer.get("name", None)
                arguments = answer.get("arguments", None)

                self._logger.debug(f"Executing function '{function_name}' with arguments: {arguments}")
                function = self._get_function(function_name)

                if self.check_is_dangerous and function is not None and self._is_dangerous(function):
                    function = None

                if function is None:
                    output.append({
                        "keep": False,
                        "execution_result": f"Function '{function_name}' not found.",
                    })
                else:
                    try:
                        args_dict = json.loads(arguments)
                        parsed_args = self._parse_arguments(args_dict, previous_results)
                        execution = execute_from_response(function, parsed_args)
                        previous_results.append(execution["execution_result"])

                        output.append({
                            "keep": execution["keep"],
                            "execution_result": execution["execution_result"],
                        })
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": wrapper["id"],
                            "content": execution["execution_result"],
                        })
                    except Exception:
                        self._logger.exception("Error parsing arguments")

                        output.append({
                            "keep": False,
                            "execution_result": f"Error parsing arguments: {arguments}",
                        })
            # Check if no function calls were executed
            if not tool_results:
                _input.update(**{
                    "keep_row_after_execution_check": False,
                    "execution_result": [],
                })
            else:
                # We only consider a good response if all the answers were executed successfully,
                # but keep the reasons for further review if needed.
                _input.update(**{
                    "keep_row_after_execution_check": all(o["keep"] is True for o in output),
                    "execution_result": [o["execution_result"] for o in output],
                })
            _input["messages"].extend(tool_results)

        yield inputs
