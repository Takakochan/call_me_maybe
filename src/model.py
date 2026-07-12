from pydantic import BaseModel
from typing import Literal


class PromptWrite(BaseModel):
    prompt: str


class ParameterSchema(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDifinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema
