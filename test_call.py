from llm_sdk.llm_sdk import Small_LLM_Model
import torch
import json


model = Small_LLM_Model()

vocab_path = model.get_path_to_vocab_file()
with open(vocab_path, "r", encoding="utf-8")as f:
    vocab = json.load(f)

# print(type(vocab))
# print(len(vocab))
# print(vocab.get("aaple",  "Not found"))

if isinstance(vocab, dict):
    sample_items = list(vocab.items())[15000:15050]
else:
    sample_items = vocab[15000:15050]
# print(sample_items)

# Encoding text
text = "what is the sum of 4 and 38? Total is:"
token_ids = model.encode(text)
# print(type(token_ids), token_ids)
# print("Token IDs:", token_ids)
# print("Number of tokens:", len(token_ids))

# ID to logits
# input_ids = torch.tensor([token_ids])

# raw = model.encode(text)

if isinstance(token_ids, torch.Tensor):
    input_ids = token_ids.long()
    token_ids = token_ids.flatten().tolist()
else:
    token_ids = token_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long)

# print(f"Number of tokens: {len(token_ids)}")
# print(f"Raw ids: {input_ids}")
# print(f"Shape of input_ids: {tuple(input_ids.shape)}")

logits = model.get_logits_from_input_ids(token_ids)
# print(f"Logits length:, {len(logits)}")


logits_t = torch.tensor(logits)
top5 = torch.topk(logits_t, 10)
for tid, score in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"{score:8.3f} {tid:6d} {model.decode([tid])!r}")


# Prefix experiment

prefix = '{"name": "'
print("===============")
print(model.encode(prefix))
print(type(model.encode(prefix).flatten()))
print(model.encode(prefix).flatten().tolist())
prefix_ids = model.encode(prefix).flatten().tolist()

generated = token_ids.copy()
for target_id in prefix_ids:
    step_logits = torch.tensor(model.get_logits_from_input_ids(generated))
    print(f"Step Logits : {step_logits}")  # kinda list

    mask = torch.full_like(step_logits, float("-inf"))
    print(mask)  #tensor list
    mask[target_id] = 0.0
    constrained = step_logits + mask
    # print(constrained)

    chosen = int(torch.argmax(constrained))
    generated.append(chosen)
    print(f"forced: {model.decode([chosen])!r} "
          f"Model's fav: {model.decode([int(torch.argmax(step_logits))])!r})")
    
print(model.decode(generated[len(token_ids):]))




#test1
# def get_parameter(
#         user_prompt: str,
#         chosen_func: str,
#         funcs: list[FunctionDefinition],
#         model: Small_LLM_Model,
#         vocab: dict[str, int],
#         id_to_token: dict[int, str]
# ) -> list[str]:
#     para = 'parameters: { '
#     para_after = '": '
#     mult_para = ', "'
#     end_para = '}'

#     param_dic = [element.parameters for element in funcs if element.name == chosen_func]
#     for param in param_dic:
#         param_prefix_dic = {k: v.type for k, v in param.items()}
#         print(f"Dck: {param_prefix_dic}")

#     param_prefix = list(param_prefix_dic.values())
#     # print(param_prefix)
#     prompt_parameter = 'function: ' + chosen_func + '\n' + str(len(param_dic)) + '\n' + para + ' ' + list(param_prefix_dic.keys())[0] + ' write in ' + list(param_prefix_dic.values())[0]
#     # print(prompt_parameter)
#     generated = model.encode(prompt_parameter).flatten().tolist()
#     li_user_prompt = user_prompt.split()
#     candidate = {word.replace('?', ''): word.replace('?', '') for word in li_user_prompt if word}
#     copy_candidate = candidate
#     print()
#     print(f"Cndidates: {candidate}")
#     chosen_param: list = []
#     while len(chosen_param) < len(param_dic[0]):
#         # print(candidate)
#         if not candidate:
#             raise RuntimeError("At function Generate_parameter All candidate eliminated - logic bug or invalid input")
#         allowed_ids = get_union_allowed_ids(candidate, vocab)
#         # print(generated)
#         logits_np = np.array(model.get_logits_from_input_ids(generated))
#         mask = np.full_like(logits_np, -np.inf)
#         mask[allowed_ids] = 0.0
#         chosen = int(np.argmax(logits_np + mask))
#         generated.append(chosen)
#         # print(f"Generated: {generated}")
#         chosen_str = id_to_token[chosen]
#         # print(f"Chosen_str: {chosen_str}")
#         candidate = update_candidates(candidate, chosen_str)
#         for name, remaining in candidate.items():
#             if remaining == "":
#                 candidate = copy_candidate
#                 del candidate[name]
#                 another_prefix = ' ' + ' pic another parameter ' + param_prefix[len(chosen_param)] 
#                 ids = model.encode(another_prefix).flatten().tolist()
#                 generated.extend(ids)
#                 chosen_param.append(name)
#     print(f"Chosen param: {chosen_param}")
#     return chosen_param
