from typing import Any

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str = Field(..., description="The name of the function to call.")
    arguments: dict[str, Any] = Field(..., description="The arguments for the function call.")


class QueryAnswer(BaseModel):
    answers: list[FunctionCall] = Field(..., description="List of function calls to answer the query.")
