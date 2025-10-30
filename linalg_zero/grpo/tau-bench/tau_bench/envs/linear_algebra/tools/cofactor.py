from typing import Any

from tau_bench.envs.tool import Tool
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.shared.lib import matrix_cofactor


class Cofactor(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], expression: str) -> str:
        try:
            return str(matrix_cofactor(**data))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def get_info() -> dict[str, Any]:
        return get_json_schema(matrix_cofactor)
