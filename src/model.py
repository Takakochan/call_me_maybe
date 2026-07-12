from pydantic import BaseModel
from typing import Literal


class PromptWrite(BaseModel):
    prompt: str


class Parameterschema(BaseModel):
    type: Literal["number", "string", "boolean"]


class FunctionDifinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameterschema]
    returns: Parameterschema
