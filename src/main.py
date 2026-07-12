import json
import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model
from parser import Parser


def get_allowed_ids(remaining: str, vocab: dict[str, int]) -> list[int]:
    """Return token IDs that can legally start spelling remaining.
    Args:
        renaining: target srt which are not written ex) "fn_greet"
        vocab: token str -> ID dict
    Returns:
        IDs list of all tokens which match the begining of remaining
    """
    allowed = []
    for i in range(1, len(remaining)):
        token = remaining[:i]
        print(token)
        if token in vocab:
            print(f"Matched token: {token}")
            allowed.append(vocab[token])
    print(allowed)
    return allowed


def get_union_allowed_ids(
    candidates: dict[str, str], vocab: dict[str, int]
) -> list[int]:
    unioned_ids = set()
    for i in candidates.values():
        print(f"i id {i}")
        for id in get_allowed_ids(i, vocab):
            unioned_ids.add(id)
    print(f"Union Allowed ids: {unioned_ids}")
    return list(unioned_ids)


def update_candidates(candidates: dict[str, str], chosen_str: str) -> dict[str, str]:
    for original, candidate in candidates.items():
        if chosen_str in candidate:
            candidates[original] = candidate[len(chosen_str) :]
    delete_items = []
    for key, value in candidates.items():
        if key == value:
            delete_items.append(key)
    for d in delete_items:
        del candidates[d]
    return candidates


def main() -> None:
    """モデルと語彙を準備して get_allowed_ids を試す."""
    # model = Small_LLM_Model()

    # function_names = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]
    # candidates = {name: name for name in function_names}
    # updated_candidates = update_candidates(candidates, "fn_")

    # vocab_path = model.get_path_to_vocab_file()
    # print(vocab_path)
    # with open(vocab_path, "r", encoding="utf-8") as f:
    #     vocab = json.load(f)

    # ket = next(iter(vocab))
    # print(ket)
    # print(vocab[ket])

    # allowed_ids = get_union_allowed_ids(updated_candidates, vocab)
    # print(allowed_ids)

    # prompt = "what is the sum of 4 and 38?"
    # generated = model.encode(prompt).flatten().tolist()
    # print(generated)

    # prefix = '{"name": "'
    # prefix_ids = model.encode(prefix).flatten().tolist()

    # generated.extend(prefix_ids)
    # print(generated)

    # chosen_func = None
    # while chosen_func is None:
    #     allowed_ids = get_union_allowed_ids(updated_candidates, vocab)
    #     logits = np.array(model.get_logits_from_input_ids(allowed_ids))
    #     mask = np.full_like(logits, -np.inf)
    #     mask[allowed_ids] = 0.0
    #     remain = int(np.argmax(mask + logits))
    #     generated.append(remain)

    # for target_id in prefix_ids:
    #     step_logits = np.array(model.get_logits_from_input_ids(generated))
    #     step = step_logits.flatten().tolist()
    #     vo = step.index(max(step_logits))
    #     # print(f"INDEX of Most high proba: {vo}")
    #     mask = np.full_like(step_logits, -np.inf)
    #     mask[target_id] = 0.0
    #     chosen = int(np.argmax(step_logits + mask))
    #     # print(chosen)
    #     # print(step_logits)
    #     # print(f"MODEL WANTED: {[k for k, v in vocab.items() if v == vo]}")
    #     # print(f"FORCED: {[k for k, v in vocab.items() if v == chosen]}")
    #     generated.append(chosen)

    # for name in function_names:
    #     ids = get_allowed_ids(name, vocab)
    #     print(f"{name}: {len(ids)}個の合法な第一歩")
    #     for tid in ids:
    #         print(f"   {tid:6d} {model.decode([tid])!r}")
    try:
        Parser("/home/tkunugi/sgoinfre/CallMeMaybe/data/input/function_calling_tests.json", "/home/tkunugi/sgoinfre/CallMeMaybe/data/input/functions_definition.json")
    except Exception as e:
        raise ValueError(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
