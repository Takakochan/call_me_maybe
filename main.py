import json
from llm_sdk.llm_sdk import Small_LLM_Model


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
        if token in vocab:
            allowed.append(vocab[token])

    return allowed

def get_union_allowed_ids(candidates: dict[str, str], vocab: dict[str, int]) -> list[int]:
    for i in candidates.values(): 
        print(f"i id {i}")
        get_allowed_ids(i, vocab)


def update_candidates(candidates: dict[str, str], chosen_str: str) -> dict[str, str]:
    for original, candidate in candidates.items():
        if chosen_str in candidate:
            candidates[original] = candidate[len(chosen_str):]
    delete_items = []
    for key, value in candidates.items():
        if key == value:
            delete_items.append(key)
    for d in delete_items:
        del candidates[d]
    return candidates


def main() -> None:
    """モデルと語彙を準備して get_allowed_ids を試す."""
    model = Small_LLM_Model()
    
    function_names = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]
    candidates = {name: name for name in function_names}
    print(update_candidates(candidates, "fn_g"))


    vocab_path = model.get_path_to_vocab_file()
    print(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    for name in function_names:
        ids = get_allowed_ids(name, vocab)
        print(f"{name}: {len(ids)}個の合法な第一歩")
        for tid in ids:
            print(f"   {tid:6d} {model.decode([tid])!r}")


if __name__ == '__main__':
    main()
