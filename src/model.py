from pydantic import BaseModel, field_validator
from typing import Literal


class PromptWrite(BaseModel):
    prompt: str

    @field_validator('prompt')
    @classmethod
    def is_empty(cls, prompt: str):
        if not prompt.strip():
            raise ValueError("Prompt should not be empty")
        return prompt


class ParameterSchema(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDifinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


