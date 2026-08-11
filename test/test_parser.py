"""Parser(src/parser.py)の入力検証テスト: 欠損・不正JSON・スキーマ不一致."""
import json
from pathlib import Path

import pytest

from src.parser import (
    Parser,
    InputFileError,
    InputFormatError,
    InputSchemaError,
)

VALID_FUNCS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    }
]
VALID_PROMPTS = [{"prompt": "What is the sum of 2 and 3?"}]


def _write(path: Path, data: object) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestValidInput:
    def test_parses_both_files(self, tmp_path: Path) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        parser = Parser(func_path, prompt_path)
        assert len(parser.func_list) == 1
        assert parser.func_list[0].name == "fn_add_numbers"
        assert len(parser.prompt_list) == 1
        assert parser.prompt_list[0].prompt == "What is the sum of 2 and 3?"


class TestMissingFile:
    def test_missing_functions_file(self, tmp_path: Path) -> None:
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        with pytest.raises(InputFileError):
            Parser(str(tmp_path / "does_not_exist.json"), prompt_path)

    def test_missing_prompts_file(self, tmp_path: Path) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        with pytest.raises(InputFileError):
            Parser(func_path, str(tmp_path / "does_not_exist.json"))


class TestInvalidJson:
    def test_malformed_json_syntax(self, tmp_path: Path) -> None:
        func_path = tmp_path / "functions.json"
        func_path.write_text("{not valid json", encoding="utf-8")
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        with pytest.raises(InputFormatError):
            Parser(str(func_path), prompt_path)

    def test_empty_file(self, tmp_path: Path) -> None:
        func_path = tmp_path / "functions.json"
        func_path.write_text("", encoding="utf-8")
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        with pytest.raises(InputFormatError):
            Parser(str(func_path), prompt_path)


class TestSchemaMismatch:
    def test_json_is_not_a_list(self, tmp_path: Path) -> None:
        func_path = _write(tmp_path / "functions.json", {"not": "a list"})
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        with pytest.raises(InputSchemaError):
            Parser(str(func_path), prompt_path)

    def test_function_missing_required_key(self, tmp_path: Path) -> None:
        bad_funcs = [{"name": "fn_incomplete"}]
        func_path = _write(tmp_path / "functions.json", bad_funcs)
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        with pytest.raises(InputSchemaError):
            Parser(func_path, prompt_path)

    def test_prompt_wrong_shape(self, tmp_path: Path) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        bad_prompts = [{"prompt": 12345}]
        prompt_path = _write(tmp_path / "prompts.json", bad_prompts)
        with pytest.raises(InputSchemaError):
            Parser(func_path, prompt_path)

    def test_empty_prompt_string_rejected(self, tmp_path: Path) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        prompt_path = _write(tmp_path / "prompts.json", [{"prompt": "   "}])
        with pytest.raises(InputSchemaError):
            Parser(func_path, prompt_path)
