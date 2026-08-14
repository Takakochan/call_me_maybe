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
    """Match an English contraction suffix starting at position ``i``.
    Args:
        text: The full text being pre-tokenized.
        i: The index in ``text`` to try matching from.
    Returns:
        The matched contraction substring (e.g. ``"'re"``), or ``None`` if
        no contraction starts at ``i``.
    """
    for suffix in CONTRACTIONS:
        end = i + len(suffix)
        if text[i:end].lower() == suffix:
            return text[i:end]
    return None


def match_word(text: str, i: int) -> str | None:
    """Match a "word" piece starting at position ``i``.
    A word optionally starts with one non-letter/non-number character
    (mirroring the GPT-2 pre-tokenizer's optional leading space handling),
    followed by one or more letters.
    Args:
        text: The full text being pre-tokenized.
        i: The index in ``text`` to try matching from.
    Returns:
        The matched word substring, or ``None`` if no word starts at
    """
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
    """Match a single digit at position ``i``."""
    return text[i:i + 1] if i < len(text) and is_number(text[i]) else None


def match_puncts(text: str, i: int) -> str | None:
    """Match a run of punctuation/symbol characters starting at ``i``.
    Optionally consumes one leading space, then greedily consumes
    characters that are neither whitespace, letters, nor numbers.
    Args:
        text: The full text being pre-tokenized.
        i: The index in ``text`` to try matching from.
    Returns:
        The matched punctuation run, or ``None`` if none starts at ``i``.
    """
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
    """Match trailing whitespace followed by one or more newlines.
    Args:
        text: The full text being pre-tokenized.
        i: The index in ``text`` to try matching from.
    Returns:
        The matched whitespace-plus-newline(s) substring, or ``None`` if
        no newline follows starting at ``i``.
    """
    j = i
    while j < len(text) and text[j].isspace() and text[j] not in ("\r", "\n"):
        j += 1
    nl_start = j
    while j < len(text) and text[j] in ("\r", "\n"):
        j += 1
    return text[i:j] if j > nl_start else None


def match_space(text: str, i: int) -> str | None:
    """Match a run of non-newline whitespace starting at ``i``.
    Mirrors GPT-2's pre-tokenizer: a run of 2 or more spaces keeps its
    last character for the following piece (so that piece can still carry
    a single leading space), while a run at the very end of the text is
    consumed in full.
    Args:
        text: The full text being pre-tokenized.
        i: The index in ``text`` to try matching from.
    Returns:
        The matched whitespace substring, or ``None`` if no non-newline
    """
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
    """Split text into pre-token pieces, trying each rule in priority order.
    At each position, the rules in ``PRETOKEN_RULES`` are tried in order;
    the first one that matches consumes that many characters. This mirrors
    the alternation order of a GPT-2-style pre-tokenizer regex.
    Args:
        text: The raw text to pre-tokenize.
    Returns:
        The list of pre-token piece strings, in order. Concatenating them
        reproduces ``text`` exactly.
    Raises:
        ValueError: If no rule matches at some position (should not
            happen for well-formed text, since ``match_space`` and
            ``match_puncts`` are catch-alls).
    """
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


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """Maps each of the 256 possible byte values to one printable Unicode
    character, so that arbitrary byte sequences can be represented as
    ordinary-looking strings for BPE merging. See:
    https://www.mrinitialman.com/HTMLTutorial/Chapters/
    Appendices/Appendices-Characters.html
    Returns:
        A dict mapping each byte value (0-255) to its corresponding
        printable Unicode character."""
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
    """Convert one pre-token piece into its list of byte-symbol characters.
    The piece is first encoded to raw UTF-8 bytes, then each byte is
    mapped through ``byte_encoder`` to its printable-character stand-in.
    Args:
        piece: The pre-token piece to convert (a plain text substring).
        byte_encoder: The byte-to-symbol mapping from ``bytes_to_unicode``.
    Returns:
        One byte-symbol character per UTF-8 byte of ``piece``.
    """
    raw_bytes = piece.encode("utf-8")
    return [byte_encoder[b] for b in raw_bytes]


def load_merges(path: str) -> dict[tuple[str, str], int]:
    """Load a GPT-2-style ``merges.txt`` file into a rank lookup table.
    Args:
        path: Path to the ``merges.txt`` file (as returned by
            ``Small_LLM_Model.get_path_to_merges_file()``).
    Returns:
        A dict mapping each ``(left, right)`` symbol pair to its merge
        rank, where a lower rank means the merge should be applied
        earlier (i.e. it has higher priority).
    """
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
    """Repeatedly merge the highest-priority adjacent symbol pair.
    At each step, every adjacent pair present in ``merge_ranks`` is a
    candidate; the one with the lowest rank (highest priority) is merged
    into a single symbol, and the process repeats until no known pair
    remains or only one symbol is left.
    Args:
        symbols: The initial list of byte-symbol characters to merge.
        merge_ranks: The merge-priority table from ``load_merges``.
    Returns:
        The final list of merged symbols (tokens), in order.
    """
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
    """Fully BPE-encode one pre-token piece into final token strings.
    Combines ``piece_to_byte_symbols`` (piece -> byte symbols) and
    ``apply_bpe`` (byte symbols -> merged tokens) into a single step.
    Args:
        piece: The pre-token piece to encode.
        byte_encoder: The byte-to-symbol mapping from ``bytes_to_unicode``.
        merge_ranks: The merge-priority table from ``load_merges``
    Returns:
        The list of final token strings for this piece.
    """
    symbols = piece_to_byte_symbols(piece, byte_encoder)
    return apply_bpe(symbols, merge_ranks)


def tokens_to_ids(
    tokens: list[str], vocab: dict[str, int]
) -> list[int]:
    """Map token strings to the model's numeric vocabulary IDs.
    Args:
        tokens: The token strings to look up.
        vocab: The token-string-to-ID mapping (loaded from the model's
    Returns:
        The corresponding list of vocabulary IDs, in the same order.
    """
    return [vocab[t] for t in tokens]


def build_byte_decoder(byte_encoder: dict[int, str]) -> dict[str, int]:
    """Invert ``bytes_to_unicode``'s mapping back to raw byte values.
    Args:
        byte_encoder: The byte-to-symbol mapping from ``bytes_to_unicode``.
    Returns:
        A dict mapping each printable-symbol character back to its
        original byte value.
    """
    return {char: byte_val for byte_val, char in byte_encoder.items()}


def encode(
    text: str, merge_ranks: dict[tuple[str, str], int],
    vocab: dict[str, int]
) -> list[int]:
    """Encode raw text into model token IDs, entirely from scratch.
    Runs the full from-scratch pipeline: pre-tokenize -> byte-level BPE
    per piece -> vocabulary lookup. This is the function used everywhere
    in the main generation path, so the SDK's own ``encode``/``decode``
    are never needed there (see the bonus "recoding the tokenizer").
    Args:
        text: The raw text to encode.
        merge_ranks: The merge-priority table from ``load_merges``.
        vocab: The token-string-to-ID mapping (loaded from the model's
            vocabulary file).
    Returns:
        The list of token IDs representing ``text``.
    """
    by_uni = bytes_to_unicode()
    pieces = pre_tokenize(text)
    own_ids = []
    for piece in pieces:
        tokens = bpe_encode_piece(piece, by_uni, merge_ranks)
        ids = tokens_to_ids(tokens, vocab)
        own_ids.extend(ids)
    return own_ids


def decode_tokens(
    token_ids: list[int],
    id_to_token: dict[int, str],
    byte_decoder: dict[str, int],
) -> str:
    """Decode a full list of token IDs back into text in a single pass.
    All tokens are converted to byte-symbol characters and concatenated
    *before* the final UTF-8 decode step, rather than decoding token by
    token. This matters for multi-byte characters whose UTF-8 bytes are
    split across separate tokens: decoding one token at a time would try
    to interpret an incomplete byte sequence and corrupt the character.
    Args:
        token_ids: The sequence of token IDs to decode.
        id_to_token: The ID-to-token-string mapping (inverse of the
            vocabulary).
        byte_decoder: The symbol-to-byte mapping from ``build_byte_decoder``.
    Returns:
        The fully decoded, UTF-8 text string.
    """
    all_symbols = "".join(id_to_token[i] for i in token_ids)
    raw_bytes = bytes(byte_decoder[ch] for ch in all_symbols)
    return raw_bytes.decode("utf-8")


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
