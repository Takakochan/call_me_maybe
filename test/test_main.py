"""Tests for src/__main__.py's CLI pipeline (Small_LLM_Model, engine stubbed).
src/__main__.pyのCLIパイプラインのテスト(Small_LLM_Modelとengineをスタブ化).

To avoid loading a real model or hitting HF downloads, main_module's
Small_LLM_Model and engine are monkeypatched; only CLI-arg parsing, the
input/output file wiring, and the verbose banner are verified.
実LLMのロードやHFダウンロードを避けるため、main_module.Small_LLM_Modelと
main_module.engineをmonkeypatchし、CLI引数の解釈・入出力ファイルの配線・
verboseバナーだけを検証する。
"""
import json
import sys
from pathlib import Path

import pytest

import src.__main__ as main_module
from src.parser import ParserError


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


def _make_fake_model_class(vocab_path: str, merges_path: str = "") -> type:
    """Build a Small_LLM_Model-compatible stub class that needs no real load.
    Small_LLM_Model互換の、ロード不要なスタブクラスを生成する."""

    class _FakeModel:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name

        def get_path_to_vocab_file(self) -> str:
            return vocab_path

        def get_path_to_merges_file(self) -> str:
            return merges_path

        def __str__(self) -> str:
            return f"FakeModel({self.model_name})"

    return _FakeModel


class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        args = main_module.parse_args()
        assert (
            args.functions_definition
            == "data/input/functions_definition.json"
        )
        assert args.input == "data/input/function_calling_tests.json"
        assert args.output == "data/output/function_calling_results.json"
        assert args.model == "Qwen/Qwen3-0.6B"
        assert args.verbose is False

    def test_verbose_flag_short_and_long(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "-v"])
        assert main_module.parse_args().verbose is True
        monkeypatch.setattr(sys, "argv", ["prog", "--verbose"])
        assert main_module.parse_args().verbose is True

    def test_overrides_paths_and_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            [
                "prog",
                "--functions_definition", "custom_funcs.json",
                "--input", "custom_prompts.json",
                "--output", "custom_out.json",
                "--model", "Qwen/Qwen3-1.7B",
            ],
        )
        args = main_module.parse_args()
        assert args.functions_definition == "custom_funcs.json"
        assert args.input == "custom_prompts.json"
        assert args.output == "custom_out.json"
        assert args.model == "Qwen/Qwen3-1.7B"


class TestMainPipeline:
    def test_writes_engine_output_as_json_and_creates_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        vocab_path = _write(tmp_path / "vocab.json", {"a": 0})
        # Nested, not-yet-existing directory: exercises os.makedirs(out_dir).
        output_path = tmp_path / "nested" / "out" / "results.json"

        monkeypatch.setattr(
            sys, "argv",
            [
                "prog",
                "--functions_definition", func_path,
                "--input", prompt_path,
                "--output", str(output_path),
            ],
        )
        monkeypatch.setattr(
            main_module, "Small_LLM_Model", _make_fake_model_class(vocab_path)
        )
        expected_result = [
            {
                "prompt": "What is the sum of 2 and 3?",
                "name": "fn_add_numbers",
                "parameters": {"a": 2.0, "b": 3.0},
            }
        ]
        monkeypatch.setattr(
            main_module,
            "engine",
            lambda parser, model, vocab, verbose: expected_result,
        )

        main_module.main()

        assert output_path.exists()
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written == expected_result

    def test_verbose_prints_introduction_banner_to_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        func_path = _write(tmp_path / "functions.json", VALID_FUNCS)
        prompt_path = _write(tmp_path / "prompts.json", VALID_PROMPTS)
        vocab_path = _write(tmp_path / "vocab.json", {"a": 0, "b": 1})
        merges_path = tmp_path / "merges.txt"
        merges_path.write_text("#version: dummy\n", encoding="utf-8")
        output_path = tmp_path / "results.json"

        monkeypatch.setattr(
            sys, "argv",
            [
                "prog",
                "--functions_definition", func_path,
                "--input", prompt_path,
                "--output", str(output_path),
                "--verbose",
            ],
        )
        monkeypatch.setattr(
            main_module,
            "Small_LLM_Model",
            _make_fake_model_class(vocab_path, str(merges_path)),
        )
        monkeypatch.setattr(
            main_module, "engine", lambda parser, model, vocab, verbose: []
        )

        main_module.main()

        captured = capsys.readouterr()
        assert "Vocabulary size:" in captured.err
        assert "Vocab file:" in captured.err

    def test_propagates_parser_error_for_missing_input_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            [
                "prog",
                "--functions_definition",
                str(tmp_path / "does_not_exist.json"),
                "--input", str(tmp_path / "also_missing.json"),
                "--output", str(tmp_path / "out.json"),
            ],
        )

        with pytest.raises(ParserError):
            main_module.main()
