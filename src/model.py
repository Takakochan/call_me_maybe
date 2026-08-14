from pydantic import BaseModel, Field, field_validator, RootModel
from typing import TypeAlias, Union


class PromptWrite(BaseModel):
    """One entry of ``function_calling_tests.json``: a single user prompt.
    Attributes:
        prompt: The natural-language request to translate into a function
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def is_empty(cls, prompt: str) -> str:
        """Reject a prompt that is empty or contains only whitespace.
        Args:
            prompt: The raw prompt string to validate.
        Returns:
            The unchanged prompt string, if valid.
        Raises:
            ValueError: If ``prompt`` is empty or whitespace-only.
        """
        if not prompt.strip():
            raise ValueError("Prompt should not be empty")
        return prompt


class ParameterSchema(BaseModel):
    """Schema for one function parameter (or return value).
    Self-referential via ``properties`` so that ``type: "object"``
    parameters can nest arbitrarily deep.
    Attributes:
        type: The JSON-ish type name, e.g. "string", "number", "integer",
        properties: For ``type == "object"``, the nested parameter schemas
            keyed by name; ``None`` for scalar types.
    """
    type: str
    properties: dict[str, "ParameterSchema"] | None = None


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``: a callable function.
    Attributes:
        name: The function's name, e.g. "fn_add_numbers".
        description: A natural-language description of what the function
            does, used to build the function-selection prompt.
        parameters: The function's parameters, keyed by name.
        returns: The schema of the function's return value.
    """
    name: str
    description: str
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: ParameterSchema


ParameterValue: TypeAlias = RootModel[
    Union[str | float | int | bool | dict[str, "ParameterValue"]]
]
ParameterValue.model_rebuild()


class ParameterFetch(BaseModel):
    """Accumulator for one prompt's generated answer.
    Filled in incrementally while ``engine()`` processes a prompt, then
    dumped as one entry of the output JSON via ``model_dump()``.
    Attributes:
        prompt: The original user prompt, copied verbatim.
        name: The name of the function chosen by the model, or "fn_none"
        parameters: The generated arguments, keyed by parameter name.
    """
    prompt: str = ""
    name: str = ""
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
