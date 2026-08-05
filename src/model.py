from pydantic import BaseModel, Field, field_validator, RootModel
from typing import Literal, Optional, Dict, Union, TypeAlias


class PromptWrite(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def is_empty(cls, prompt: str):
        if not prompt.strip():
            raise ValueError("Prompt should not be empty")
        return prompt


class ParameterSchema(BaseModel):
    """Using itself BaseModel enable self-referential"""

    type: str
    properties: Optional[Dict[str, "ParameterSchema"]] = None


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


ParameterValue: TypeAlias = RootModel[
    Union[str, float, int, bool, Dict[str, "ParameterValue"]]
]
ParameterValue.model_rebuild()


class ParameterFetch(BaseModel):
    prompt: str = ""
    name: str = ""
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
