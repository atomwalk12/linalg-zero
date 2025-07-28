import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from pydantic import Field
from typing_extensions import override

from linalg_zero.distillation.utils import is_openai_format

if TYPE_CHECKING:
    from distilabel.typing import ChatType


class ChatGeneration(Task):
    """
    Generates text based on a conversation. This class customises the default `ChatGeneration` class.
    It prepares the output for the subsequent steps and dynamically adjusts the system prompt.
    """

    # TODO: these variables look fairly ugly, are they really necessary?
    system_prompt: str | None = Field(default=None, description="The system prompt to use in the generation.")
    tool_calls: bool = Field(default=False, description="Whether the generation contains tool calls.")
    initialized: bool = Field(
        default=False, description="Whether the component has been initialized with the system prompt."
    )
    thinking_mode: Literal["/think", "/no_think"] = Field(
        default="/think", description="Whether to enable thinking mode."
    )

    @property
    @override
    def inputs(self) -> list[str]:
        """The input for the task are the `messages`."""
        return ["messages"]

    @override
    def format_input(self, input: dict[str, Any]) -> "ChatType":
        """
        The input is formatted as a `ChatType` assuming that the messages provided
        are already formatted that way i.e. following the OpenAI chat format.
        """

        if not is_openai_format(input["messages"]):
            raise DistilabelUserError((
                "Input `messages` must be an OpenAI chat-like format conversation.",
                f" Got: {input['messages']}.",
            ))

        if self.system_prompt and not self.initialized:
            # TODO: this is somewhat a hack that is necessary to make local development with LlamaCPP work.
            # Currently, the library doesn't support dynamically switching between thinking and no-thinking modes.
            # To work around this, I am setting the mode inside the message itself. This is not needed for vLLM.
            if self.thinking_mode == "/no_think":
                input["messages"][1]["content"] = input["messages"][1]["content"] + " /no_think"

            # The difference between the default `ChatGeneration` class and this one is that
            # we use customised logic to handle the system prompt.
            if input["messages"][0]["role"] != "system":
                # No system prompt is present, so we simply add it to the beginning of the conversation.
                input["messages"].insert(0, {"role": "system", "content": self.system_prompt})
            else:
                # An existing system prompt is present, so we need to replace it.
                input["messages"].pop(0)
                input["messages"].insert(0, {"role": "system", "content": self.system_prompt})
            self.initialized = True

        return input["messages"]

    @property
    @override
    def outputs(self) -> list[str]:
        """The output for the task is the `generation`, `model_name` and the updated `messages`."""
        # The state that is changed upon task completion. The messages represent the chat history,
        # to be reused in the subsequent steps.
        return ["generation", "model_name", "messages"]

    @override
    def format_output(self, output: str | None, input: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically. The messages are updated to include the assistant's response.
        """
        # TODO: this function is really convoluted. Need to improve flow.
        if input is None:
            raise DistilabelUserError("Input is required to format the output.")

        input_copy = deepcopy(input)
        if output is None:
            return self._default_error(input_copy)

        result = []
        try:
            if self.tool_calls:
                # Currently, instructor doesn't support root fields in the model output, and because
                # of this, we need to recur to this hack to obtain the generation in the correct format.
                parsed_output = json.loads(output)
                for i, tool_call in enumerate(parsed_output["tool_calls"]):
                    # Ensure tool_call is a dict and arguments is properly formatted
                    if isinstance(tool_call, dict):
                        tool_call_copy = tool_call.copy()
                        if "arguments" in tool_call_copy:
                            tool_call_copy["arguments"] = json.dumps(tool_call_copy["arguments"])
                        result.append({"type": "function", "id": f"tool_call_{i}", "function": tool_call_copy})
                    output = parsed_output["thinking"]
                input_copy["messages"].append({"role": "assistant", "tool_calls": result, "content": output})
                input_copy.update({"generation": json.dumps(result)})
            else:
                input_copy["messages"].append({"role": "assistant", "content": output})
                input_copy.update({"generation": output})
        except json.JSONDecodeError:
            return self._default_error(input_copy)
        else:
            return input_copy

    def _default_error(self, _input: dict[str, Any]) -> dict[str, Any]:
        """Returns a default error output, to fill the responses in case of failure."""
        _input.update(**{"generation": None, "messages": None})
        return _input
