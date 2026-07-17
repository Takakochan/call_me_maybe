import json
import time

from llm_sdk.llm_sdk import Small_LLM_Model
from parser import Parser
from generate import engine

def main() -> None:
    """モデルと語彙を準備して get_allowed_ids を試す."""
    start = time.time()
    #parser = Parser(sys.argv[1], sys.argv[2])
    parser = Parser("/home/tkunugi/sgoinfre/CallMeMaybe/data/input/function_calling_tests.json", "/home/tkunugi/sgoinfre/CallMeMaybe/data/input/functions_definition.json")
    model = Small_LLM_Model()
    # model = Small_LLM_Model(model_name="Qwen/Qwen3-1.7B")
    engine(parser, model)
    
    end = time.time()
    print(end - start)
    # prompts = parser.prompt_list
    # funcs = parser.func_list
    # vocab_path = model.get_path_to_vocab_file()
    # with open(vocab_path, "r", encoding="utf-8") as f:
    #     vocab = json.load(f)
    # id_to_token = vocab_id_to_token(vocab)
    # for user_prompt in prompts:
    #     print()
    #     print(user_prompt)
    #     chosen_func = generate_function_call(
    #         user_prompt.prompt,
    #         funcs,
    #         model,
    #         vocab,
    #         id_to_token
    #     )
    # end = time.time()
    # print(end - start)


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
    #     print(chosen)
    #     print(step_logits)
    #     print(f"MODEL WANTED: {[k for k, v in vocab.items() if v == vo]}")
    #     print(f"FORCED: {[k for k, v in vocab.items() if v == chosen]}")
    #     generated.append(chosen)

    # chosen_function = None
    # while chosen_function is None:
    #     if not candidates:
    #         raise RuntimeError("All candidates eliminated - logic bug or invalid input")
    #     allowed_ids = get_union_allowed_ids(candidates, vocab)
    #     logits_np = np.array(model.get_logits_from_input_ids(generated))
    #     mask = np.full_like(logits_np, -np.inf)
    #     mask[allowed_ids] = 0.0
    #     chosen = int(np.argmax(logits_np + mask))
    #     generated.append(chosen)
    #     print(generated)
    
    #     for g in generated:
    #         vocab_id_to_token = voca_id_to_token(g, vocab)
    #         print(vocab_id_to_token)

    #     chosen_str = vocab_id_to_token[chosen]
    #     candidates = update_candidates(candidates, chosen_str)
    #     print(f"Model Chose: {chosen_str!r}, Remaining func names{candidates}")
    #     for name, remaining in candidates.items():
    #         if remaining == "":
    #             chosen_function = name

    # print(f"Chosen function: {chosen_function}")

    # for name in function_names:
    #     ids = get_allowed_ids(name, vocab)
    #     print(f"{name}: {len(ids)}個の合法な第一歩")
    #     for tid in ids:
    #         print(f"   {tid:6d} {model.decode([tid])!r}")
    # try:
    #     Parser("/home/tkunugi/sgoinfre/CallMeMaybe/data/input/function_calling_tests.json", "/home/tkunugi/sgoinfre/CallMeMaybe/data/input/functions_definition.json")
    # except Exception as e:
    #     raise ValueError(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
