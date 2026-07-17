from pydantic import BaseModel, Field, field_validator, model_validator
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


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    # required_para: dict[str, ParameterSchema]
    returns: ParameterSchema

    # # TODO Need to make this validator dynamic, test eventually
    # @model_validator(mode="after")
    # def param_cocrdinate(self) -> "FunctionDifinition":
    #     parameters = sorted(self.parameters.items(), key=lambda x: x[0])
    #     required = sorted(self.required_para.items(), key=lambda x: x[0])
    #     if parameters != required:
    #         raise ValueError("Unmatching parameters")
    #     return self


class ParameterFetch(BaseModel):
    prompt: str = ""
    name: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)