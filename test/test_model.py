"""pydanticモデル(src/model.py)のバリデーションテスト."""
import pytest
from pydantic import ValidationError

from src.model import (
    PromptWrite,
    ParameterSchema,
    FunctionDefinition,
    ParameterValue,
    ParameterFetch,
)


class TestPromptWrite:
    def test_valid_prompt(self) -> None:
        p = PromptWrite(prompt="Hello world")
        assert p.prompt == "Hello world"

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptWrite(prompt="")

    def test_whitespace_only_prompt_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptWrite(prompt="   \n\t")

    def test_missing_prompt_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptWrite()


class TestParameterSchema:
    def test_scalar_schema(self) -> None:
        s = ParameterSchema(type="string")
        assert s.type == "string"
        assert s.properties is None

    def test_nested_object_schema(self) -> None:
        """再帰的な自己参照(objectがobjectを含む)が構築できる."""
        s = ParameterSchema(
            type="object",
            properties={
                "inner": {
                    "type": "object",
                    "properties": {"leaf": {"type": "number"}},
                }
            },
        )
        assert s.properties is not None
        assert s.properties["inner"].properties is not None
        assert s.properties["inner"].properties["leaf"].type == "number"


class TestFunctionDefinition:
    def test_valid_function(self) -> None:
        fn = FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            returns={"type": "number"},
        )
        assert fn.name == "fn_add_numbers"
        assert set(fn.parameters.keys()) == {"a", "b"}

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FunctionDefinition(
                name="fn_missing_returns",
                description="No returns field.",
                parameters={},
            )


class TestParameterValue:
    """RootModel[Union[str, float, int, bool, dict[...]]]の受理テスト."""

    @pytest.mark.parametrize(
        "raw",
        ["hello", 3.14, 42, True],
    )
    def test_accepts_each_scalar_type(self, raw: object) -> None:
        wrapped = ParameterValue(raw)
        assert wrapped.root == raw

    def test_nested_dict_is_recursively_wrapped(self) -> None:
        """dict値は要素ごとに再帰的にParameterValueへ包まれる."""
        wrapped = ParameterValue({"nested": "value"})
        assert isinstance(wrapped.root, dict)
        assert wrapped.root["nested"].root == "value"

    def test_nested_dict_of_values(self) -> None:
        wrapped = ParameterValue(
            {"a": ParameterValue(1), "b": ParameterValue("x")}
        )
        assert isinstance(wrapped.root, dict)


class TestParameterFetch:
    def test_defaults(self) -> None:
        pf = ParameterFetch()
        assert pf.prompt == ""
        assert pf.name == ""
        assert pf.parameters == {}

    def test_independent_default_dicts(self) -> None:
        """default_factoryのおかげでインスタンス間でparametersが共有されない."""
        a = ParameterFetch()
        b = ParameterFetch()
        a.parameters["x"] = ParameterValue(1)
        assert "x" not in b.parameters
