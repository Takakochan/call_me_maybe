import unicodedata
from functools import lru_cache
from llm_sdk.llm_sdk import Small_LLM_Model


def is_letter(ch: str) -> bool:
    """Evaluation for \\p{L}(Category of "Letter" in Unicode)"""
    return unicodedata.category(ch).startswith("L")


def is_number(ch: str) -> bool:
    """Evaluation for \\p{L}(Category of "Number" in Unicode)"""
    return unicodedata.category(ch).startswith("N")


CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]


def match_contraction(text: str, i: int) -> str | None:
    for suffix in CONTRACTIONS:
        end = i + len(suffix)
        if text[i:end].lower() == suffix:
            return text[i:end]
    return None


def match_word(text: str, i: int) -> str | None:
    j = i
    if (
        j < len(text)
        and text[j] not in ("\r", "\n")
        and not is_letter(text[j])
        and not is_number(text[j])
    ):
        j += 1
    start = j
    while j < len(text) and is_letter(text[j]):
        j += 1
    return text[i:j] if j > start else None


def match_number(text: str, i: int) -> str | None:
    return text[i:i + 1] if i < len(text) and is_number(text[i]) else None


def match_puncts(text: str, i: int) -> str | None:
    j = i
    if j < len(text) and text[j] == " ":
        j += 1
    start = j
    while (
        j < len(text)
        and not text[j].isspace()
        and not is_letter(text[j])
        and not is_number(text[j])
    ):
        j += 1
    if j == start:
        return None
    while j < len(text) and text[j] in ("\r", "\n"):
        j += 1
    return text[i:j]


def match_newline(text: str, i: int) -> str | None:
    j = i
    while j < len(text) and text[j].isspace() and text[j] not in ("\r", "\n"):
        j += 1
    nl_start = j
    while j < len(text) and text[j] in ("\r", "\n"):
        j += 1
    return text[i:j] if j > nl_start else None


def match_space(text: str, i: int) -> str | None:
    j = i
    while j < len(text) and text[j].isspace() and text[j] not in ("\r", "\n"):
        j += 1
    if j == i:
        return None
    length = j - i
    if j == len(text):
        return text[i:j]
    if length >= 2:
        return text[i:j - 1]
    return text[i:j]


PRETOKEN_RULES = [
    match_contraction,
    match_word,
    match_number,
    match_puncts,
    match_newline,
    match_space,
]


def pre_tokenize(text: str) -> list[str]:
    """テキストを、正規表現の優先順位通りに「塊」へ分割する."""
    pieces: list[str] = []
    i = 0
    while i < len(text):
        for rule in PRETOKEN_RULES:
            matched = rule(text, i)
            if matched is not None:
                pieces.append(matched)
                i += len(matched)
                break
        else:
            raise ValueError(
                f"No pre-tokenizer rule matched at position {i}: {text[i]!r}"
            )
    return pieces


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """256種類のバイト値を、印刷可能なUnicode文字1つずつに対応させる.
    https://www.mrinitialman.com/HTMLTutorial/Chapters/
    Appendices/Appendices-Characters.html"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs_chars = [chr(c) for c in cs]
    return dict(zip(bs, cs_chars))


def piece_to_byte_symbols(
    piece: str,
    byte_encoder: dict[int, str]
) -> list[str]:
    """1つのpre-token piece(元の文字列)を、byte-symbolのリストに変換する."""
    raw_bytes = piece.encode("utf-8")
    return [byte_encoder[b] for b in raw_bytes]


def load_merges(path: str) -> dict[tuple[str, str], int]:
    """merges.txtを読み込み、(左, 右) -> 優先順位(小さいほど高優先)の辞書を返す."""
    merge_ranks: dict[tuple[str, str], int] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rank = 0
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("#") or not line:
            continue
        left, right = line.split(" ")
        merge_ranks[(left, right)] = rank
        rank += 1
    return merge_ranks


def apply_bpe(
    symbols: list[str],
    merge_ranks: dict[tuple[str, str], int]
) -> list[str]:
    """優先順位の高いペアから繰り返しBPEマージを適用する."""
    symbols = list(symbols)
    while len(symbols) > 1:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        candidates = [
            (merge_ranks[p], i) for i, p in enumerate(pairs)
            if p in merge_ranks
        ]
        if not candidates:
            break
        _, best_i = min(candidates)
        symbols = (
            symbols[:best_i]
            + [symbols[best_i] + symbols[best_i + 1]]
            + symbols[best_i + 2:]
        )
    return symbols


def bpe_encode_piece(
    piece: str,
    byte_encoder: dict[int, str],
    merge_ranks: dict[tuple[str, str], int],
) -> list[str]:
    """1つのpre-token pieceを、最終的なトークン文字列のリストに変換する."""
    symbols = piece_to_byte_symbols(piece, byte_encoder)
    return apply_bpe(symbols, merge_ranks)


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """トークン文字列のリストを、モデルが理解するIDのリストに変換する."""
    return [vocab[t] for t in tokens]


def build_byte_decoder(byte_encoder: dict[int, str]) -> dict[str, int]:
    """byte_encoderの逆引き: 印刷可能文字 → 元のバイト値."""
    return {char: byte_val for byte_val, char in byte_encoder.items()}


def decode_tokens(
    token_ids: list[int],
    id_to_token: dict[int, str],
    byte_decoder: dict[str, int],
) -> str:
    all_symbols = "".join(id_to_token[i] for i in token_ids)
    raw_bytes = bytes(byte_decoder[ch] for ch in all_symbols)
    return raw_bytes.decode("utf-8")


def encode(
    text: str, merge_ranks: dict[tuple[str, str], int], vocab: dict[str, int]
) -> list[int]:
    by_uni = bytes_to_unicode()
    pieces = pre_tokenize(text)
    own_ids = []
    for piece in pieces:
        tokens = bpe_encode_piece(piece, by_uni, merge_ranks)
        ids = tokens_to_ids(tokens, vocab)
        own_ids.extend(ids)

    return own_ids


if __name__ == "__main__":
    import json

    be = bytes_to_unicode()
    model = Small_LLM_Model()
    merges_path = model.get_path_to_merges_file()
    merge_ranks = load_merges(merges_path)
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    text = "What is the sum of 265 and 345?"
    pieces = pre_tokenize(text)
    my_ids = []
    for piece in pieces:
        tokens = bpe_encode_piece(piece, be, merge_ranks)
        ids = tokens_to_ids(tokens, vocab)
        my_ids.extend(ids)

    sdk_ids = model.encode(text).flatten().tolist()

    print("自作:", my_ids)
    print("SDK :", sdk_ids)
    print("一致:", my_ids == sdk_ids)

# if __name__ == "__main__":
#     import json
#     be = bytes_to_unicode()
#     bd = build_byte_decoder(be)
#     model = Small_LLM_Model()
#     merges_path = model.get_path_to_merges_file()
#     merge_ranks = load_merges(merges_path)
#     vocab_path = model.get_path_to_vocab_file()
#     with open(vocab_path, "r", encoding="utf-8") as f:
#         vocab = json.load(f)
#     id_to_token = {v: k for k, v in vocab.items()}

#     test_strings = [
#         "What is the sum of 265 and 345?",
#         "Greet shrek",
#         "don't stop",
#     ]
#     for text in test_strings:
#         pieces = pre_tokenize(text)
#         my_ids = []
#         for piece in pieces:
#             tokens = bpe_encode_piece(piece, be, merge_ranks)
#             ids = tokens_to_ids(tokens, vocab)
#             my_ids.extend(ids)
#         decoded = decode_tokens(my_ids, id_to_token, bd)
#         status = "OK" if decoded == text else "MISMATCH"
#         print(f"[{status}] {text!r} -> {len(my_ids)} tokens -> {decoded!r}")
