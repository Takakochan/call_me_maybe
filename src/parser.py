import json
from pydantic import ValidationError
from .model import PromptWrite, FunctionDefinition


class ParserError(Exception):
    """Base for error while persing"""

    pass


class InputFileError(ParserError):
    """The input file could not be opened at all."""

    def __init__(self, path: str, detail: str) -> None:
        """Build the error with the offending path and a human-readable reason.
        Args:
            path: The file path that could not be opened.
                開けなかったファイルのパス。
            detail: Short human-readable explanation, e.g. "File not found".
                短い人間可読の説明（例: "File not found"）。
        """
        self.path = path
        self.detail = detail
        super().__init__(f"{detail}: {path}")


class InputSchemaError(ParserError):
    """The input JSON parsed fine but does not match the expected schema.
    入力JSON自体は解析できたが、期待するスキーマと一致しない場合のエラー。
    Covers cases such as: the top level is not a JSON array, or an entry
    is missing a required key or has a value of the wrong type.
    トップレベルがJSON配列でない、必須キーが欠けている、値の型が
    間違っている、といったケースを含む。
    """

    def __init__(self, path: str, original: ValidationError | str) -> None:
        """Build the error with the offending path and the validation cause.
        問題のパスと、バリデーション失敗の原因からエラーを構築する。
        Args:
            path: The file path whose contents failed schema validation.
                スキーマ検証に失敗したファイルのパス。
            original: The pydantic ``ValidationError`` that triggered this,
                or a plain string describing the mismatch.
                原因となったpydanticの``ValidationError``、または不一致を
                説明する単純な文字列。
        """
        self.path = path
        self.original = original
        super().__init__(f"Schema mismatch {original}: {path}")


class InputFormatError(ParserError):
    """The input file's contents are not syntactically valid JSON.
    入力ファイルの中身が構文的に正しいJSONではない場合のエラー。
    Wraps the standard library's ``json.JSONDecodeError`` with the file
    path, so the error message can point at exactly which file is broken.
    標準ライブラリの``json.JSONDecodeError``をファイルパスと共にラップし、
    どのファイルが壊れているかをエラーメッセージで明示できるようにする。
    """
    def __init__(self, path: str, original: json.JSONDecodeError) -> None:
        """Build the error with the offending path and the decode error.
        問題のパスと、JSONデコードエラーからエラーを構築する。
        Args:
            path: The file path whose contents are not valid JSON.
                有効なJSONではなかったファイルのパス。
            original: The underlying ``json.JSONDecodeError``, used to
                report the exact line/column of the syntax error.
                元となった``json.JSONDecodeError``。構文エラーの正確な
                行・列を報告するために使う。
        """
        self.path = path
        self.original = original
        super().__init__(
            f"Invalid JSON {path}: {original.msg} "
            f"(line {original.lineno}, "
            f"column {original.colno})"
        )


def _load_json(path: str) -> list:
    """Load a JSON file and check that its top level is a list.
    This is the single place where the three failure stages are ordered:
    file-not-found, then invalid syntax, then wrong shape - so that, e.g.,
    a missing file is never mistakenly reported as a schema error.
    3段階の失敗（ファイル未検出→構文エラー→形式不一致）の順序を一箇所に
    まとめている。これにより、例えばファイル未検出がスキーマエラーとして
    誤って報告されることがない。
    Args:
        path: Path to the JSON file to load.
            読み込むJSONファイルのパス。
    Returns:
        The parsed JSON content, guaranteed to be a ``list``.
        パース済みのJSON内容。必ず``list``であることが保証される。
    Raises:
        InputFileError: If the file does not exist or cannot be read.
            ファイルが存在しない、または読み取れない場合。
        InputFormatError: If the file's contents are not valid JSON.
            ファイルの中身が有効なJSONでない場合。
        InputSchemaError: If the parsed JSON's top level is not a list.
            パース済みJSONのトップレベルがリストでない場合。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, list):
                raise InputSchemaError(
                    path, "Invalid JSON format, Expected a JSON array"
                )
            return raw
    except FileNotFoundError as e:
        raise InputFileError(path, "File not found") from e
    except PermissionError as e:
        raise InputFileError(path, "Permission required") from e
    except json.JSONDecodeError as e:
        raise InputFormatError(path, e) from e


class Parser:
    """Loads and validates the two mandatory input files.
    On success, exposes the validated prompts and function definitions as
    typed pydantic model lists, ready for the generation pipeline.
    Attributes:
        prompt_list: The validated list of prompts from ``prompt_path``.
        func_list: The validated list of function definitions from
    """
    def __init__(self, func_path: str, prompt_path: str) -> None:
        """Parse and validate both input files immediately.
        Args:
            func_path: Path to ``functions_definition.json``.
                ``functions_definition.json``へのパス。
            prompt_path: Path to ``function_calling_tests.json``.
                ``function_calling_tests.json``へのパス。
        Raises:
            ParserError: If either file is missing, malformed, or does
                not match the expected schema (see the subclasses above
                for the specific cause).
        """
        self.prompt_list: list[PromptWrite] = self._parse_prompt(prompt_path)
        self.func_list: list[FunctionDefinition] = self._parse_func(func_path)

    def _parse_prompt(self, path: str) -> list[PromptWrite]:
        """Load ``function_calling_tests.json`` as a list of prompts.
        Args:
            path: Path to the prompts JSON file.
        Returns:
            One ``PromptWrite`` per entry in the file.
        Raises:
            InputSchemaError: If an entry does not match ``PromptWrite``'s
                schema (e.g. missing/wrong-typed "prompt" key).
        """
        raw = _load_json(path)
        try:
            return [PromptWrite(**entry) for entry in raw]
        except (ValidationError, TypeError) as e:
            raise InputSchemaError(
                path,
                f"Unexpected prompt type/form - {e}"
            ) from e

    def _parse_func(self, path: str) -> list[FunctionDefinition]:
        """Load ``functions_definition.json`` as a list of function defs.
        Args:
            path: Path to the function-definitions JSON file.
        Returns:
            One ``FunctionDefinition`` per entry in the file.
        Raises:
            InputSchemaError: If an entry does not match
                ``FunctionDefinition``'s schema (e.g. missing "returns" key).
        """
        raw = _load_json(path)
        try:
            return [FunctionDefinition(**entry) for entry in raw]
        except (ValidationError, TypeError) as e:
            raise InputSchemaError(
                path,
                f"Unexpected function type/form - {e}"
            ) from e
