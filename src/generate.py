import json
import numpy as np
import re
from typing import Any
import argparse

from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser
from .model import FunctionDefinition, ParameterFetch


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
        candidates: dict[str, str], chosen_str: str
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
        prompt: str, definitions: list[FunctionDefinition]
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
        for title, parameter in func.parameters.items():
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


def generate_parameter(
    param_fetch_dict: ParameterFetch,
    user_prompt: str,
    chosen_func: str,
    param_type_list: list[str],
    param_name: list[str],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
) -> str:
    # print(f"param_list : {param_type_list}")
    # print(f"Parameter_name is : {param_name}")

    TERMINATOR_LIST = [
        vocab[","], vocab["}"]
    ]
    DIGITS_IDS = [
        t_id for token,
        t_id in vocab.items()
        if token.isdigit()
    ]
    ALL_IDS = [
        t_id
        for token, t_id in vocab.items()
        if token != vocab[","] and token != vocab["}"] and not token.isdigit()
    ]
    # generated_parameter = None
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
    words = []
    for i in range(len(param_type_list)):
        param_type = param_type_list[i]
        parameter_title = param_name[i]
        if param_type == "number":
            
            prompt = prompt + '"' + parameter_title + '": '
            value = ""
            if verbose:
                print(f"Param_type {param_type}")
                print(f"Param_name {param_name}")
                print(f"Prompt: {prompt}")
            generated = model.encode(prompt).flatten().tolist()
            while True:
                allowed_ids = (
                    DIGITS_IDS if not value else DIGITS_IDS + TERMINATOR_LIST
                )
                discouraged_ids: list[int] = []
                chosen = masked_argmax(
                    model,
                    generated,
                    allowed_ids,
                    discouraged_ids
                )
                chosen_tok = id_to_token[chosen]
                if chosen in TERMINATOR_LIST:
                    break
                value += chosen_tok  # 桁を積む
                generated.append(chosen)
                # print(f"Value: {value}")
                if len(value) > 15:
                    raise RuntimeError(f"Runaway number generation: {value!r}")
            param_fetch_dict.parameters[parameter_title] = float(value)
            # print(f"WORDs: {words}")
            prompt = f"{prompt}{value}, "

        elif param_type == "string":
            prompt = prompt + '"' + parameter_title + '": "'
            value = ""
            
            vprint(f"Prompt: {prompt}", verbose)
            generated = model.encode(prompt).flatten().tolist()
            while True:
                allowed_ids = (
                    ALL_IDS if not value else ALL_IDS + TERMINATOR_LIST
                )
                parameter_id = get_allowed_ids(parameter_title, vocab)
                discouraged_ids = DIGITS_IDS + parameter_id
                chosen = masked_argmax(
                    model,
                    generated,
                    allowed_ids,
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
                
                vprint(f"Completed Value: {value}", verbose)
                if len(value) > 60:
                    raise RuntimeError(f"Runaway number generation: {value!r}")
            param_fetch_dict.parameters[parameter_title] = value
            words.append(value)
            # print(f"WORDs: {words}")
            prompt = prompt + value + ", "

    prompt = prompt[:-2] + "}"
    # print(prompt)
    return prompt


def generate_function_call(
    user_prompt: str,
    funcs: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace
    
) -> str:
    """1プロンプト分の生成パイプライン。選ばれた関数名を返す."""
    full_prompt = build_dynamic_prompt(user_prompt, funcs)
    vprint('', verbose)
    vprint(f"Dynamic prompt: {full_prompt}", verbose)
    generated = model.encode(full_prompt).flatten().tolist()
    prefix = '{"name": "'
    prefix_ids = model.encode(prefix).flatten().tolist()
    generated.extend(prefix_ids)
    candidates = {f.name: f.name for f in funcs}
    chosen_function = None
    while chosen_function is None:
        if not candidates:
            raise RuntimeError(
                "All candidates eliminated - " +
                "logic bug or invalid input")
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        discouraged_ids: list[int] = []
        chosen = masked_argmax(model, generated, allowed_ids, discouraged_ids)

        generated.append(chosen)
        vprint(generated, verbose)
        chosen_str = id_to_token[chosen]
        vprint(f"ModelPicked Logit ID: {chosen} \n Logit Token: {chosen_str}", verbose)
        candidates = update_candidates(candidates, chosen_str)
        vprint(f"ModelChose: {chosen_str!r}, Remaining func name{candidates}", verbose)
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_function = name
    vprint(f"Chosen function: {chosen_function}", verbose)
    return chosen_function


def engine(
    parser: Parser,
    model: Small_LLM_Model,
    vocab: Any,
    verbose: argparse.Namespace
) -> list[dict]:
    prompts = parser.prompt_list
    funcs = parser.func_list
    id_to_token = vocab_id_to_token(vocab)
    answer_list: list[dict] = []
    for user_prompt in prompts:
        if verbose:
            print(f"User prompt: {user_prompt}")
        param_fetch_dict = ParameterFetch()
        param_fetch_dict.prompt = user_prompt.prompt
        chosen_func = generate_function_call(
            user_prompt.prompt,
            funcs,
            model,
            vocab,
            id_to_token,
            verbose
        )
        param_fetch_dict.name = chosen_func
        parameter_type_list = get_parameter_type_list(chosen_func, funcs)
        func = next(d for d in funcs if d.name == chosen_func)
        param_name = list(func.parameters)
        generate_parameter(
            param_fetch_dict,
            user_prompt.prompt,
            chosen_func,
            parameter_type_list,
            param_name,
            model,
            vocab,
            id_to_token,
            verbose
        )
        answer_list.append(param_fetch_dict.model_dump())
        # print(param_fetch_dict.model_dump_json(indent=2))

    return answer_list
