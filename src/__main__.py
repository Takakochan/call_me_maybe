import json
import time
import os
import argparse
import sys

from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser, ParserError
from .generate import engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to write the results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """モデルと語彙を準備して get_allowed_ids を試す."""
    start = time.time()
    args = parse_args()
    parser = Parser(args.functions_definition, args.input)
    output_path = args.output
    model = Small_LLM_Model()
    # model = Small_LLM_Model(model_name="Qwen/Qwen3-1.7B")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_put = engine(parser, model)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_put, f, indent=2, ensure_ascii=False)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    try:
        main()
    except ParserError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
