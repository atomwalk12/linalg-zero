from typing import TYPE_CHECKING, Any, Final

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.math_shepherd.utils import (
    parse_json_response,
    split_solution_steps,
)
from distilabel.typing import ChatItem
from jinja2 import Template
from pydantic import PositiveInt
from typing_extensions import override

if TYPE_CHECKING:
    from distilabel.typing import ChatType, StepColumns


SYSTEM_PROMPT = """\
You are a math tutor that helps students solve math problems by breaking them down into clear, logical steps. Follow these guidelines:

# For each step:
- Clearly explain the reasoning
- Show the calculated result for any arithmetic calculation
- Present intermediate calculations clearly
- Use clear, concise language to explain the mathematical reasoning

# Format requirements:
- Number each step starting with "Step 1:"
- The final step should clearly state "The answer is: [result]"
- Keep explanations clear and concise

{{ extra_rules }}{{ few_shots }}{{ structured_prompt }}"""


SYSTEM_PROMPT_STRUCTURED: Final[str] = """
Your answer must adhere to the following format, with each step by step solution in a separate object:
```
[
    {
        "solution": "Step 1: Your first step\nStep 2: Your second step\n...\nThe answer is: [Your final answer]",
    },
    ... (more solutions as required)
]
```
"""


RULES_GSM8K: Final[str] = """\
# Rules:
- All calculations must be shown within <<>> brackets
- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction
- Write the full calculation and result, e.g., <<5*10=50>>50
"""

FEW_SHOTS_GSM8K: Final[str] = """
# Examples:
## Instruction
A store sells notebooks for $3 each. If you buy 5 or more, you get a 20% discount. How much would you pay for 6 notebooks?

## Solution
Step 1: Calculate the regular price for 6 notebooks: 6 * $3 = <<63=18>>18 dollars
Step 2: Calculate the 20% discount amount: 18 * 20/100 = <<1820/100=3.6>>3.6 dollars
Step 3: Subtract the discount from the regular price: 18 - 3.6 = <<18-3.6=14.4>>14.4 dollars. The answer is: 14.4

## Instruction
A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?

## Solution
Step 1: Find out how many cups of flour are needed per cookie: 2.5 ÷ 12 = <<2.5/12=0.208333>>0.208333 cups
Step 2: Calculate the flour needed for 30 cookies: 0.208333 * 30 = <<0.208333*30=6.25>>6.25 cups. The answer is: 6.25
"""

RULES_MATH: Final[str] = """\
# Rules:
- Always wrap mathematical expressions in $ symbols
- Use LaTeX-style math notation with $ symbols for mathematical expressions
- Format operations and equations properly using LaTeX notation within $ symbols
- Keep explanations precise and mathematically rigorous
- Use $\boxed{}$ notation only in the final step
"""

FEW_SHOTS_MATH: Final[str] = """
# Examples
## Input
Find the sum of the first three perfect squares greater than 50.

## Output
Step 1: The first perfect square greater than 50 is $8^2 = 64$.
Step 2: The second perfect square is $9^2 = 81$.
Step 3: The third perfect square is $10^2 = 100$.
Step 4: The sum is $64 + 81 + 100 = 245$.
Step 5: Therefore, the answer is $\boxed{245}$. The answer is: 245

## Input
What is the value of $2^5 + 3^3$?

## Output
Step 1: Calculate $2^5 = 32$.
Step 2: Calculate $3^3 = 27$.
Step 3: Add the results: $32 + 27 = 59$.
Step 4: Therefore, the answer is $\boxed{59}$. The answer is: 59
"""

TEMPLATE: str = """{% if M %}Generate {{ M }} example solutions to the following problem, separated by a single `---`. This is your problem:{% endif %}
{{ instruction }}"""

TEMPLATE_STRUCTURED: str = """{% if M %}Generate {{ M }} diverse solutions, even if they are incorrect. This is the problem:{% endif %}
{{ instruction }}"""


class ChainOfThoughtGenerator(Task):
    system_prompt: str | None = SYSTEM_PROMPT
    extra_rules: str | None = RULES_GSM8K
    few_shots: str | None = FEW_SHOTS_GSM8K
    M: PositiveInt | None = None

    @override
    def load(self) -> None:
        super().load()
        if self.system_prompt is not None:
            self.system_prompt = Template(self.system_prompt).render(
                extra_rules=self.extra_rules or "",
                few_shots=self.few_shots or "",
                structured_prompt=SYSTEM_PROMPT_STRUCTURED if self.use_default_structured_output else "",
            )
        if self.use_default_structured_output:
            self._template: Template = Template(TEMPLATE_STRUCTURED)
        else:
            self._template = Template(TEMPLATE)

    @property
    @override
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    @property
    @override
    def outputs(self) -> "StepColumns":
        if self.M:
            return ["solutions", "model_name"]
        return ["golden_solution", "model_name"]

    @override
    def format_input(self, input: dict[str, Any]) -> "ChatType":
        messages = [
            {
                "role": "user",
                "content": self._template.render(
                    instruction=input["instruction"],
                    M=self.M,
                ),
            }
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        result: list[ChatItem] = []
        for message in messages:
            result.append(ChatItem(role=message["role"], content=message["content"]))
        return result

    @override
    def format_output(self, output: str | None, input: dict[str, Any] | None = None) -> dict[str, Any]:
        output_name = "solutions" if self.M else "golden_solution"
        inp = input or {}

        if output is None:
            inp.update(**{output_name: None})
            return inp

        if self.M:
            output_parsed = (
                self._format_structured_output(output) if self.use_default_structured_output else output.split("---")
            )
            solutions = [split_solution_steps(o) for o in output_parsed]
        else:
            output_parsed = self._format_structured_output(output)[0] if self.use_default_structured_output else output
            solutions = split_solution_steps(output_parsed)

        inp.update(**{output_name: solutions})
        return inp

    @override
    def get_structured_output(self) -> dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        The schema corresponds to the following:

        ```python
        from pydantic import BaseModel, Field

        class Solution(BaseModel):
            solution: str = Field(..., description="Step by step solution leading to the final answer")

        class MathShepherdGenerator(BaseModel):
            solutions: list[Solution] = Field(..., description="List of solutions")

        MathShepherdGenerator.model_json_schema()
        ```

        Returns:
            JSON Schema of the response to enforce.
        """
        """
        [{
            'instruction': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            'solutions': [[], []],
            'distilabel_metadata': {'raw_output_solution_generator': '{"solutions":[{"solution":"Step 1: Calculate the number of eggs Janet\'s ducks lay per day."}]}',
            'raw_input_solution_generator': [
                {
                    'role': 'system',
                    'content': 'You are a math tutor that helps students solve math problems by breaking them down into clear, logical steps. Follow these guidelines:\n\n# For each step:\n- Clearly explain the reasoning\n- Show the calculated result for any arithmetic calculation\n- Present intermediate calculations clearly\n- Use clear, concise language to explain the mathematical reasoning\n\n# Format requirements:\n- Number each step starting with "Step 1:"\n- The final step should clearly state "The answer is: [result]"\n- Keep explanations clear and concise\n\n# Rules:\n- All calculations must be shown within <<>> brackets\n- Basic operations: use * for multiplication, / for division, + for addition, - for subtraction\n- Write the full calculation and result, e.g., <<5*10=50>>50\n\n# Examples:\n## Instruction\nA store sells notebooks for $3 each. If you buy 5 or more, you get a 20% discount. How much would you pay for 6 notebooks?\n\n## Solution\nStep 1: Calculate the regular price for 6 notebooks: 6 * $3 = <<63=18>>18 dollars\nStep 2: Calculate the 20% discount amount: 18 * 20/100 = <<1820/100=3.6>>3.6 dollars\nStep 3: Subtract the discount from the regular price: 18 - 3.6 = <<18-3.6=14.4>>14.4 dollars. The answer is: 14.4\n\n## Instruction\nA recipe calls for 2.5 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?\n\n## Solution\nStep 1: Find out how many cups of flour are needed per cookie: 2.5 ÷ 12 = <<2.5/12=0.208333>>0.208333 cups\nStep 2: Calculate the flour needed for 30 cookies: 0.208333 * 30 = <<0.208333*30=6.25>>6.25 cups. The answer is: 6.25\n\nYour answer must adhere to the following format, with each step by step solution in a separate object:\n
                    ```
                    [
                        {
                            "solution": "Step 1: Your first step\nStep 2: Your second step\n...\nThe answer is: [Your final answer]",
                        },
                        ... (more solutions as required)
                    ]
                    ```
                {
                    'role': 'user',
                    'content': "Generate 2 diverse solutions, even if they are incorrect. This is the problem:\nJanet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market?"}
                ],
                'statistics_solution_generator':
                {
                    'input_tokens': 946,
                    'output_tokens': 30
                }
            },
            'model_name': 'https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf/resolve/main/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf'
        }]
        """

        return {
            "$defs": {
                "Solution": {
                    "properties": {
                        "solution": {
                            "description": "Step by step solution leading to the final answer",
                            "title": "Solution",
                            "type": "string",
                        }
                    },
                    "required": ["solution"],
                    "title": "Solution",
                    "type": "object",
                }
            },
            "properties": {
                "solutions": {
                    "description": "List of solutions",
                    "items": {"$ref": "#/$defs/Solution"},
                    "title": "Solutions",
                    "type": "array",
                }
            },
            "required": ["solutions"],
            "title": "MathShepherdGenerator",
            "type": "object",
        }

    def _format_structured_output(self, output: str) -> list[str]:
        default_output = [""] * self.M if self.M else [""]
        if parsed_output := parse_json_response(output):
            solutions = parsed_output["solutions"]
            extracted_solutions = [o["solution"] for o in solutions]
            if len(extracted_solutions) != self.M:
                extracted_solutions = default_output
            return extracted_solutions
        return default_output
