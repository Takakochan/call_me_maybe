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
