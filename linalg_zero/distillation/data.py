from typing import Any

from pydantic import BaseModel, Field


class FunctionInvocationInfo(BaseModel):
    name: str = Field(..., description="The name of the function to call.")
    arguments: dict[str, Any] = Field(..., description="The arguments for the function call.")


class AssistantMessage(BaseModel):
    """The structured output of the assistant generated during the planning phase."""

    thinking: str = Field(..., description="The thinking process for selecting tools.")
    tool_calls: list[FunctionInvocationInfo] = Field(..., description="List of function calls to answer the query.")
