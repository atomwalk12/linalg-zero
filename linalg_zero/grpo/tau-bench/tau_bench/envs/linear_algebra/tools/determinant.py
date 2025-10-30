from typing import Any

from tau_bench.envs.tool import Tool
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.shared.lib import determinant


class Determinant(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], expression: str) -> str:
        try:
            return str(determinant(**data))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def get_info() -> dict[str, Any]:
        return get_json_schema(determinant)
