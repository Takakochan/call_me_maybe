from pydantic import BaseModel, Field, field_validator, RootModel
from typing import TypeAlias, Union


class PromptWrite(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def is_empty(cls, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt should not be empty")
        return prompt


class ParameterSchema(BaseModel):
    """Using itself BaseModel enable self-referential"""

    type: str
    properties: dict[str, "ParameterSchema"] | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


ParameterValue: TypeAlias = RootModel[
    Union[str | float | int | bool | dict[str, "ParameterValue"]]
]
ParameterValue.model_rebuild()


class ParameterFetch(BaseModel):
    prompt: str = ""
    name: str = ""
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
