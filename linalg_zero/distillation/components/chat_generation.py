from copy import deepcopy
from typing import TYPE_CHECKING, Any

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.utils.chat import is_openai_format
from pydantic import Field
from typing_extensions import override

if TYPE_CHECKING:
    from distilabel.typing import ChatType


class ChatGeneration(Task):
    """
    Generates text based on a conversation. This class customises the default `ChatGeneration` class.
    It prepares the output for the subsequent steps and dynamically adjusts the system prompt.
    """

    system_prompt: str | None = Field(default=None, description="The system prompt to use in the generation.")
    initialized: bool = Field(
        default=False, description="Whether the component has been initialized with the system prompt."
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
        if input is None:
            raise DistilabelUserError("Input is required to format the output.")

        input_copy = deepcopy(input)
        if output is None:
            return self._default_error(input_copy)

        input_copy["messages"].append({"role": "assistant", "content": output})
        input_copy.update({"generation": output})
        return input_copy

    def _default_error(self, _input: dict[str, Any]) -> dict[str, Any]:
        """Returns a default error output, to fill the responses in case of failure."""
        _input.update(**{"generation": None, "messages": None})
        return _input
