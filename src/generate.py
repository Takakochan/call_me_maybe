import numpy as np
import re
from typing import Any
import argparse

from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser
from .model import FunctionDefinition, ParameterFetch, ParameterValue
from .bpe_tokenizer import encode


def vprint(to_print: str, verbose: argparse.Namespace) -> None:
    if verbose:
        print(to_print)


def get_allowed_ids(remaining: str, vocab: dict[str, int]) -> list[int]:
    """Return token IDs that can legally start spelling remaining.
    Args:
        renaining: target srt which are not written ex) "fn_greet"
        vocab: token str -> ID dict
    Returns:
        IDs list of all tokens which match the begining of remaining
    """
    allowed = []
    for i in range(1, len(remaining) + 1):
        token = remaining[:i]
        # print(token)
        if token in vocab:
            # print(f"Matched token: {token}")
            allowed.append(vocab[token])
    # print(allowed)
    return allowed


def get_union_allowed_ids(
    candidates: dict[str, str], vocab: dict[str, int]
) -> list[int]:
    unioned_ids = set()
    for i in candidates.values():
        # print(f"i id {i}")
        for id in get_allowed_ids(i, vocab):
            unioned_ids.add(id)
    # print(f"Union Allowed ids: {unioned_ids}")
    return list(unioned_ids)


def update_candidates(
    candidates: dict[str, str],
    chosen_str: str
) -> dict[str, str]:
    """chosen_strで始まる候補だけ残し、そのぶん削った新しい辞書を返す."""
    return {
        name: remaining[len(chosen_str):]
        for name, remaining in candidates.items()
        if remaining.startswith(chosen_str)
    }


def vocab_id_to_token(vocab: dict) -> dict[int, str]:
    return {v: key for key, v in vocab.items()}


def build_dynamic_prompt(
    prompt: str,
    definitions: list[FunctionDefinition]
) -> str:
    lines = [f"- {d.name}: {d.description}" for d in definitions]
    return (
        "You translate user requests into function calls.\n"
        "Available functions:\n" + "\n".join(lines) + "\n"
        f"User request: {prompt}\n"
    )


def get_parameter_type_list(
    chosen_func: str, funcs: list[FunctionDefinition]
) -> list[str]:
    """returns like '"a": ', eventually need to use replace("'", '"')"""
    intro_list = []
    for func in funcs:
        if func.name != chosen_func:
            continue
        for parameter in func.parameters.values():
            info = f"{parameter.type}"
            intro_list.append(info)
    return intro_list


def masked_argmax(
    model: Small_LLM_Model,
    generated: list,
    allowed_ids: list[int],
    discouraged_ids: list[int],
) -> int:
    logits_np = np.array(model.get_logits_from_input_ids(generated))
    mask = np.full_like(logits_np, -np.inf)
    mask[allowed_ids] = 0.0
    mask[discouraged_ids] = -8.0
    return int(np.argmax(logits_np + mask))


def generate_value(
    parameter_title: str,
    param_schema: Any,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> tuple[Any, str]:
    """Generate a value for a parameter. if it's object, recurse.
    if it's scalar original logic
    """
    TERMINATOR_LIST = [vocab[","], vocab["}"]]
    DIGITS_IDS = [
        t_id
        for token, t_id in vocab.items()
        if token.isdigit() or token == "," or token == "."
    ]
    ALL_IDS = [
        t_id
        for token, t_id in vocab.items()
        if token != vocab[","] and token != vocab["}"] and not token.isdigit()
    ]

    if param_schema.type == "object":
        if param_schema.properties is None:
            raise ValueError(
                f'Schema for "{parameter_title}" has type "object" '
                "but no properties defined"
            )
        prompt = prompt + '"' + parameter_title + '": {'
        nested_value: dict[str, Any] = {}
        items = list(param_schema.properties.items())
        for i, (sub_name, sub_schema) in enumerate(items):
            if i > 0:
                prompt += ", "
            sub_value, prompt = generate_value(
                sub_name,
                sub_schema,
                prompt,
                model,
                vocab,
                id_to_token,
                verbose,
                merge_ranks,
            )
            nested_value[sub_name] = sub_value
        prompt = prompt + "}"
        return nested_value, prompt

    elif param_schema.type == "number" or param_schema.type == "integer":
        prompt = prompt + '"' + parameter_title + '": '
        value = ""
        if verbose:
            print("\nParam_type number")
            print(f"Param_name {parameter_title}")
            print(f"Prompt: {prompt}")
        generated = encode(prompt, merge_ranks, vocab)
        while True:
            allowed_ids = DIGITS_IDS if not value \
                else DIGITS_IDS + TERMINATOR_LIST
            discouraged_ids: list[int] = []
            chosen = masked_argmax(
                model, generated,
                allowed_ids, discouraged_ids
            )
            chosen_tok = id_to_token[chosen]
            if chosen in TERMINATOR_LIST:
                break
            value += chosen_tok
            generated.append(chosen)
            if len(value) > 15:
                raise RuntimeError(f"Runaway number generation: {value!r}")
        if param_schema.type == "number":
            return float(value), f"{prompt}{value}"
        else:
            return int(value), f"{prompt}{value}"

    elif param_schema.type == "boolean":
        prompt = prompt + '"' + parameter_title + '": '
        if verbose:
            print("\nParam_type boolean")
            print(f"Param_name {parameter_title}")
            print(f"Prompt: {prompt}")
        generated = encode(prompt, merge_ranks, vocab)
        candidates = {"true": "true", "false": "false"}
        chosen_bool: Any = None
        while chosen_bool is None:
            allowed_ids = get_union_allowed_ids(candidates, vocab)
            chosen = masked_argmax(model, generated, allowed_ids, [])
            generated.append(chosen)
            chosen_str = id_to_token[chosen]
            candidates = update_candidates(candidates, chosen_str)
            for name, remaining in candidates.items():
                if remaining == "":
                    chosen_bool = name
        value = chosen_bool == "true"
        return value, f"{prompt}{chosen_bool}"

    elif param_schema.type == "string":
        prompt = prompt + '"' + parameter_title + '": "'
        value = ""
        if verbose:
            print(f"Param_type {param_schema.type}")
            print(f"Param_name {parameter_title}")
            print(f"Prompt: {prompt}")
        generated = encode(prompt, merge_ranks, vocab)
        while True:
            allowed_ids = ALL_IDS if not value else ALL_IDS + TERMINATOR_LIST
            parameter_id = get_allowed_ids(parameter_title, vocab)
            discouraged_ids = DIGITS_IDS + parameter_id
            chosen = masked_argmax(
                model, generated, allowed_ids,
                discouraged_ids
            )
            chosen_tok = model.decode(chosen)
            vprint(f"Chosen token:=={chosen_tok}==", verbose)
            if not value:
                chosen_tok = chosen_tok.lstrip(" ")
                chosen_tok = chosen_tok.rstrip(",")
            if '"' in chosen_tok:
                prefix = chosen_tok.split('"')[0]
                value += prefix
                break
            if "," in chosen_tok:
                break
            value += chosen_tok

            vprint(f"Built Value:=={value}==", verbose)
            generated.append(chosen)
            if len(value) > 60:
                raise RuntimeError(f"Runaway string generation: {value!r}")
        return value, prompt + value + '"'
    else:
        raise ValueError(f"Unsupported parameter type: {param_schema.type}")


def generate_parameter(
    param_fetch_dict: ParameterFetch,
    user_prompt: str,
    chosen_func: str,
    parameters: dict,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> str:
    prompt = (
        '"prompt": '
        + user_prompt
        + "\n"
        + '"function": '
        + chosen_func
        + "\n"
        + '"parameters": {'
    )
    if bool(re.search(r"\b[A-Z]+\b", user_prompt)):
        prompt = (
            "When only a value is capital letter and plural, "
            + "spell it out completely, including the final 's'.\n"
            + 'Example: \'with CATS\' -> replacement: "CATS" (not "CAT")\n'
            + prompt
        )
    items = list(parameters.items())
    for i, (title, schema) in enumerate(items):
        if i > 0:
            prompt += ", "
        value, prompt = generate_value(
            title, schema, prompt, model,
            vocab, id_to_token, verbose, merge_ranks
        )
        param_fetch_dict.parameters[title] = ParameterValue(value)
    prompt = prompt + "}"
    return prompt


def generate_function_call(
    user_prompt: str,
    funcs: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> str:
    """1プロンプト分の生成パイプライン。選ばれた関数名を返す."""
    full_prompt = build_dynamic_prompt(user_prompt, funcs)
    vprint("let AI model chose a FUNCTION by feeding dynamic prompt", verbose)
    vprint(f"Dynamic prompt: {full_prompt}", verbose)
    generated = encode(full_prompt, merge_ranks, vocab)
#    prefix = '{"name": "'
    prefix_ids = encode(full_prompt, merge_ranks, vocab)
    generated.extend(prefix_ids)
    candidates = {f.name: f.name for f in funcs}
    candidates["fn_none"] = "fn_none"
    chosen_function = None
    while chosen_function is None:
        if not candidates:
            raise RuntimeError(
                "All candidates eliminated - " + "logic bug or invalid input"
            )
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        discouraged_ids: list[int] = []
        chosen = masked_argmax(model, generated, allowed_ids, discouraged_ids)

        generated.append(chosen)
        vprint(f"\n{generated}", verbose)
        chosen_str = id_to_token[chosen]
        vprint(f"Model picked Logit ID: {chosen} \n"
               f"Logit Token: {chosen_str}", verbose)
        candidates = update_candidates(candidates, chosen_str)
        vprint(
            f"ModelChose: {chosen_str!r}, Remaining func name{candidates}",
            verbose
        )
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_function = name
    vprint(f"\nChosen function: {chosen_function}", verbose)
    return chosen_function


def verify_function_choice(
    user_prompt: str,
    chosen_func: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    merge_ranks: dict[tuple[str, str], int],
    verbose: argparse.Namespace,
) -> bool:
    """選ばれた関数がプロンプトに本当に合っているか、モデルに判定させる."""
    verify_prompt = (
        f'User request: "{user_prompt}"\n'
        f"Selected function: {chosen_func}\n"
        f"Is this selected function appropriate for the User request? "
    )
    generated = encode(verify_prompt, merge_ranks, vocab)
    candidates = {"yes": "yes", "no": "no"}
    chosen_answer = None
    while chosen_answer is None:
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        chosen = masked_argmax(model, generated, allowed_ids, [])
        generated.append(chosen)
        chosen_str = id_to_token[chosen]
        candidates = update_candidates(candidates, chosen_str)
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_answer = name
    vprint(f"Verification: {chosen_func} -> {chosen_answer}", verbose)
    return chosen_answer == "yes"


def engine(
    parser: Parser, model: Small_LLM_Model,
    vocab: Any, verbose: argparse.Namespace
) -> list[dict]:
    prompts = parser.prompt_list
    funcs = parser.func_list
    id_to_token = vocab_id_to_token(vocab)
    answer_list: list[dict] = []

    from .bpe_tokenizer import load_merges

    merges_path = model.get_path_to_merges_file()
    merge_ranks = load_merges(merges_path)

    for user_prompt in prompts:
        vprint("\n====================New Request===================", verbose)
        vprint(f"User prompt: {user_prompt}", verbose)
        param_fetch_dict = ParameterFetch()
        param_fetch_dict.prompt = user_prompt.prompt
        vprint("\n========Select Function=======", verbose)
        chosen_func = generate_function_call(
            user_prompt.prompt,
            funcs,
            model,
            vocab,
            id_to_token,
            verbose,
            merge_ranks,
        )

        if not verify_function_choice(
            user_prompt.prompt,
            chosen_func,
            model,
            vocab,
            id_to_token,
            merge_ranks,
            verbose,
        ):
            chosen_func = "fn_none"

        param_fetch_dict.name = chosen_func

        if chosen_func == "fn_none":
            vprint("\n=======No matching function=======", verbose)
        else:
            func = next(d for d in funcs if d.name == chosen_func)
            vprint("\n=======Generate Parameter=======", verbose)
            generate_parameter(
                param_fetch_dict,
                user_prompt.prompt,
                chosen_func,
                func.parameters,
                model,
                vocab,
                id_to_token,
                verbose,
                merge_ranks,
            )
        answer_list.append(param_fetch_dict.model_dump())
        vprint("\n****Complete generation for the prompt****", verbose)
        vprint(param_fetch_dict.model_dump_json(indent=2), verbose)

    return answer_list
