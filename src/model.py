from pydantic import BaseModel, Field, field_validator, RootModel
from typing import TypeAlias, Union


class PromptWrite(BaseModel):
    """One entry of ``function_calling_tests.json``: a single user prompt.
    ``function_calling_tests.json`` の1エントリ（ユーザープロンプト1件）を表す。
    Attributes:
        prompt: The natural-language request to translate into a function
            call. 関数呼び出しへ変換すべき自然言語のリクエスト。
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def is_empty(cls, prompt: str) -> str:
        """Reject a prompt that is empty or contains only whitespace.
        空文字列、または空白文字のみのプロンプトを拒否する。
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
    関数の1パラメータ（または戻り値）のスキーマ。
    Self-referential via ``properties`` so that ``type: "object"``
    parameters can nest arbitrarily deep.
    ``properties`` を通じて自己参照するため、``type: "object"``の
    パラメータは任意の深さでネストできる。
    Attributes:
        type: The JSON-ish type name, e.g. "string", "number", "integer",
            "boolean", or "object". JSON的な型名（"string"、"number"、
            "integer"、"boolean"、"object"のいずれか）。
        properties: For ``type == "object"``, the nested parameter schemas
            keyed by name; ``None`` for scalar types.
            ``type == "object"``の場合、名前をキーとするネストした
            パラメータスキーマ。スカラー型の場合は``None``。
    """
    type: str
    properties: dict[str, "ParameterSchema"] | None = None


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``: a callable function.
    ``functions_definition.json`` の1エントリ（呼び出し可能な関数1つ）を表す。
    Attributes:
        name: The function's name, e.g. "fn_add_numbers".
            関数名（例: "fn_add_numbers"）。
        description: A natural-language description of what the function
            does, used to build the function-selection prompt.
            関数の説明文。関数選択用プロンプトの構築に使われる。
        parameters: The function's parameters, keyed by name.
            関数の各パラメータ（名前をキーとする）。
        returns: The schema of the function's return value.
            関数の戻り値のスキーマ。
    """
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


ParameterValue: TypeAlias = RootModel[
    Union[str | float | int | bool | dict[str, "ParameterValue"]]
]
ParameterValue.model_rebuild()


class ParameterFetch(BaseModel):
    """Accumulator for one prompt's generated answer.
    1つのプロンプトに対して生成された回答を蓄積するための入れ物。
    Filled in incrementally while ``engine()`` processes a prompt, then
    dumped as one entry of the output JSON via ``model_dump()``.
    ``engine()``がプロンプトを処理する過程で少しずつ埋められ、最終的に
    ``model_dump()``で出力JSONの1エントリとして書き出される。
    Attributes:
        prompt: The original user prompt, copied verbatim.
            元のユーザープロンプト（そのままコピーされる）。
        name: The name of the function chosen by the model, or "fn_none"
            if no function matched. モデルが選んだ関数名。一致する関数が
            ない場合は"fn_none"。
        parameters: The generated arguments, keyed by parameter name.
            生成された各引数（パラメータ名をキーとする）。
    """
    prompt: str = ""
    name: str = ""
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
