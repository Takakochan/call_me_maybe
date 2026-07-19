import json
from pydantic import ValidationError
from .model import PromptWrite, FunctionDefinition


class ParserError(Exception):
    """Base for error while persing"""

    pass


class InputFileError(ParserError):
    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{detail}: {path}")


class InputSchemaError(ParserError):
    def __init__(self, path: str, original: ValidationError | str) -> None:
        self.path = path
        self.original = original
        super().__init__(f"Schema mismatch {original}: {path}")


class InputFormatError(ParserError):
    def __init__(self, path: str, original: json.JSONDecodeError) -> None:
        self.path = path
        self.original = original
        super().__init__(
            f"Invalid JSON {path}: {original.msg} "
            f"(line {original.lineno}, "
            f"column {original.colno})"
        )


def _load_json(path: str) -> list:
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
    def __init__(self, func_path: str, prompt_path: str) -> None:
        self.prompt_list: list[PromptWrite] = self._parse_prompt(prompt_path)
        self.func_list: list[FunctionDefinition] = self._parse_func(func_path)

    def _parse_prompt(self, path: str) -> list[PromptWrite]:
        raw = _load_json(path)
        try:
            return [PromptWrite(**entry) for entry in raw]
        except (ValidationError, TypeError) as e:
            raise InputSchemaError(
                path, f"Unexpected prompt type/form - {e}"
            ) from e

    def _parse_func(self, path: str) -> list[FunctionDefinition]:
        raw = _load_json(path)
        try:
            return [FunctionDefinition(**entry) for entry in raw]
        except (ValidationError, TypeError) as e:
            raise InputSchemaError(
                path, f"Unexpected function type/form - {e}"
            ) from e
