import json
import time
import os
import argparse
import sys
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser, ParserError
from .generate import engine


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ``python -m src`` entry point.。
    Returns:
        The parsed arguments, including ``functions_definition``,
        ``input``, ``output``, ``model``, and ``verbose`` - each with a
        sensible default so the program can be run with no flags at all.
    """
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
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name (default: Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show diagnostic output during generation.",
    )

    return parser.parse_args()


def introduction_model(
    args: argparse.Namespace,
    model: Small_LLM_Model,
    vocab: Any
) -> None:
    """Print a diagnostic banner describing the loaded model and vocab. """
    print("=" * 50, file=sys.stderr)
    print(f"Model:          {args.model}", file=sys.stderr)
    print(f"Vocabulary size:{len(vocab):,}", file=sys.stderr)
    print(f"Vocab file:     {model.get_path_to_vocab_file()}", file=sys.stderr)
    print(
        f"Merges file:      {model.get_path_to_merges_file()}", file=sys.stderr
    )
    print("=" * 50, file=sys.stderr)


def main() -> None:
    """Run the full CLI pipeline: parse inputs, generate, write output.
    """
    start = time.time()
    args = parse_args()
    parser = Parser(args.functions_definition, args.input)
    output_path = args.output
    model = Small_LLM_Model(args.model)
    print(model)
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    introduction_model(args, model, vocab)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_put = engine(parser, model, vocab, verbose=args.verbose)
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
