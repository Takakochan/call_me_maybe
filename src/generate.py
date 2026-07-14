import json
import numpy as np
import time

from llm_sdk.llm_sdk import Small_LLM_Model
from parser import Parser

from model import FunctionDefinition, PromptWrite


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


def update_candidates(candidates: dict[str, str], chosen_str: str) -> dict[str, str]:
    """chosen_strで始まる候補だけ残し、そのぶん削った新しい辞書を返す."""
    return {
        name: remaining[len(chosen_str):]
        for name, remaining in candidates.items()
        if remaining.startswith(chosen_str)
    }


def vocab_id_to_token(vocab: dict) -> dict[int, str]:
    return {v: key for key, v in vocab.items()}


def build_dynamic_prompt(prompt: str, definitions: list[FunctionDefinition]) -> str:
    lines = [f"- {d.name}: {d.description}" for d in definitions]
    return (
        "You translate user requests into function calls.\n"
        "Available functions:\n" + "\n".join(lines) + "\n"
        f"User request: {prompt}\n"
    )


# def generate_parameter(
#         user_prompt: str,
#         chosen_func: str,
#         funcs: list[FunctionDefinition],
#         model: Small_LLM_Model,
#         vocab: dict[str, int],
#         id_to_token: dict[int, str]
# ) -> list[str]:

#     param_dic = [element.parameters for element in funcs if element.name == chosen_func]
#     prompt_parameter = 'function:' + chosen_func + str(len(param_dic[0])) + '"parameters": {"'
    
#     generated = model.encode(prompt_parameter).flatten().tolist()
#     li_user_prompt = user_prompt.split()
#     candidate = {word.replace('?', ''): word.replace('?', '') for word in li_user_prompt if word}
#     print()
#     print(f"Cndidates: {candidate}")
#     chosen_param: list = []
#     while len(chosen_param) < len(param_dic[0]):
#         para_prefix = 'parameters": {"'
#         para_after = '": '
#         mult_para = ', "'
#         end_para = '}'
#         print(candidate)
#         if not candidate:
#             raise RuntimeError("At function Generate_parameter All candidate eliminated - logic bug or invalid input")
#         allowed_ids = get_union_allowed_ids(candidate, vocab)
#         logits_np = np.array(model.get_logits_from_input_ids(generated))
#         mask = np.full_like(logits_np, -np.inf)
#         mask[allowed_ids] = 0.0
#         chosen = int(np.argmax(logits_np + mask))
#         generated.append(chosen)
#         chosen_str = id_to_token[chosen]
#         # print(f"Chosen_str: {chosen_str}")
#         chosen_param.append(chosen_str)

#         # candidate = update_candidates(candidate, chosen_str)
#         # for name, remaining in candidate.items():
#         #     if remaining == "":
#         #         chosen_param.append(name)
#     print(f"Chosen param: {chosen_param}")
#     return chosen_param

#test1
def generate_parameter(
        user_prompt: str,
        chosen_func: str,
        funcs: list[FunctionDefinition],
        model: Small_LLM_Model,
        vocab: dict[str, int],
        id_to_token: dict[int, str]
) -> list[str]:
    para_prefix = 'parameters": {"'
    para_after = '": '
    mult_para = ', "'
    end_para = '}'

    param_dic = [element.parameters for element in funcs if element.name == chosen_func]
    prompt_parameter = 'function:' + chosen_func + str(len(param_dic[0])) + para_prefix
    
    generated = model.encode(prompt_parameter).flatten().tolist()
    li_user_prompt = user_prompt.split()
    candidate = {word.replace('?', ''): word.replace('?', '') for word in li_user_prompt if word}
    copy_candidate = candidate
    print()
    print(f"Cndidates: {candidate}")
    chosen_param: list = []
    while len(chosen_param) < len(param_dic[0]):
        # print(candidate)
        if not candidate:
            raise RuntimeError("At function Generate_parameter All candidate eliminated - logic bug or invalid input")
        allowed_ids = get_union_allowed_ids(candidate, vocab)
        # print(generated)
        logits_np = np.array(model.get_logits_from_input_ids(generated))
        mask = np.full_like(logits_np, -np.inf)
        mask[allowed_ids] = 0.0
        chosen = int(np.argmax(logits_np + mask))
        generated.append(chosen)
        # print(f"Generated: {generated}")
        chosen_str = id_to_token[chosen]
        print(f"Chosen_str: {chosen_str}")
        candidate = update_candidates(candidate, chosen_str)
        for name, remaining in candidate.items():
            if remaining == "":
                chosen_param.append(name)
                candidate = copy_candidate
                del candidate[name]
                ids = model.encode("and ").flatten().tolist()
                generated.extend(ids)


    print(f"Chosen param: {chosen_param}")
    return chosen_param

def generate_function_call(
    user_prompt: str,
    funcs: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str]
) -> str:
    """1プロンプト分の生成パイプライン。選ばれた関数名を返す."""
    full_prompt = build_dynamic_prompt(user_prompt, funcs)
    # print(full_prompt)
    generated = model.encode(full_prompt).flatten().tolist()
    prefix = '{"name": "'
    prefix_ids = model.encode(prefix).flatten().tolist()
    generated.extend(prefix_ids)
    candidates = {f.name: f.name for f in funcs}
    chosen_function = None
    while chosen_function is None:
        if not candidates:
            raise RuntimeError("All candidates eliminated - logic bug or invalid input")
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        logits_np = np.array(model.get_logits_from_input_ids(generated))
        mask = np.full_like(logits_np, -np.inf)
        mask[allowed_ids] = 0.0
        chosen = int(np.argmax(logits_np + mask))

        generated.append(chosen)
        # print(generated)
        chosen_str = id_to_token[chosen]
        # print(f"Model picked \n Logit ID: {chosen} \n Logit Token: {chosen_str}")
        candidates = update_candidates(candidates, chosen_str)
        # print(f"Model Chose: {chosen_str!r}, Remaining func names{candidates}")
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_function = name
    # print(f"Chosen function: {chosen_function}")
    generate_parameter(
        user_prompt,
        chosen_function,
        funcs,
        model,
        vocab,
        id_to_token
    )
    return chosen_function



# def main() -> None:
#     """モデルと語彙を準備して get_allowed_ids を試す."""
#     start = time.time()
#     model = Small_LLM_Model()
#     #parser = Parser(sys.argv[1], sys.argv[2])
#     parser = Parser("/home/tkunugi/sgoinfre/CallMeMaybe/data/input/function_calling_tests.json", "/home/tkunugi/sgoinfre/CallMeMaybe/data/input/functions_definition.json")
#     prompts = parser.prompt_list
#     funcs = parser.func_list
#     vocab_path = model.get_path_to_vocab_file()
#     with open(vocab_path, "r", encoding="utf-8") as f:
#         vocab = json.load(f)
#     id_to_token = vocab_id_to_token(vocab)
#     for user_prompt in prompts:
#         print(user_prompt)
#         string = generate_function_call(
#             user_prompt.prompt,
#             funcs,
#             model,
#             vocab,
#             id_to_token
#         )
#         print()
#     end = time.time()
#     print(end - start)


    # for prompt in prompt_list:
    #     generated = model.encode(prompt).flatten().tolist()
    #     prefix = '{"name": "'
    #     prefix_ids = model.encode(prefix).flatten().tolist()
    #     print(build_dynamic_prompt(prompt, funcs))

    # function_names = [func.name for func in funcs]
    # # print(function_names)
    # candidates = {name: name for name in function_names}
    # updated_candidates = update_candidates(candidates, "fn_")


    # vocab_path = model.get_path_to_vocab_file()
    # with open(vocab_path, "r", encoding="utf-8") as f:
    #     vocab = json.load(f)
    # # ket = next(iter(vocab))
    # # print(ket)
    # # print(vocab[ket])

    # allowed_ids = get_union_allowed_ids(updated_candidates, vocab)
    # print(allowed_ids)

    # # prompt = "what is the sum of 4 and 38?"
    # # generated = model.encode(prompt).flatten().tolist()
    # # print(generated)

    # prefix = '{"name": "'
    # prefix_ids = model.encode(prefix).flatten().tolist()

    # generated.extend(prefix_ids)
    # print(generated)

    # # chosen_func = None
    # # while chosen_func is None:
    # #     allowed_ids = get_union_allowed_ids(updated_candidates, vocab)
    # #     logits = np.array(model.get_logits_from_input_ids(allowed_ids))
    # #     mask = np.full_like(logits, -np.inf)
    # #     mask[allowed_ids] = 0.0
    # #     remain = int(np.argmax(mask + logits))
    # #     generated.append(remain)

