import pytest

from src.bpe_tokenizer import (
    pre_tokenize,
    bytes_to_unicode,
    build_byte_decoder,
    apply_bpe,
    decode_tokens,
    encode,
)


# ============================================================
# pre_tokenize
# ============================================================

class TestPreTokenize:
    """Unit tests for pre_tokenize.
    pre_tokenizeの単体テスト."""

    def test_basic_sentence(self) -> None:
        result = pre_tokenize("What is the sum of 265 and 345?")
        assert result == [
            "What", " is", " the", " sum", " of", " ",
            "2", "6", "5", " and", " ", "3", "4", "5", "?",
        ]

    def test_digits_isolated(self) -> None:
        """Digits are isolated one at a time (\\p{N} carries no quantifier).
        数字は1文字ずつ分離される(\\p{N}に量指定子が無いため)."""
        result = pre_tokenize("abc123")
        assert "123" not in result
        assert "1" in result and "2" in result and "3" in result

    def test_contractions(self) -> None:
        result = pre_tokenize("don't stop")
        assert result == ["don", "'t", " stop"]

    def test_multiple_contractions(self) -> None:
        result = pre_tokenize("I'll be there, we're ready")
        assert "'ll" in result
        assert "'re" in result

    def test_multiple_spaces(self) -> None:
        """A run of consecutive spaces keeps its last char for the next word.
        連続スペースは最後の1文字を次の単語側に残す."""
        result = pre_tokenize("Hello   world")
        assert result == ["Hello", "  ", " world"]

    def test_trailing_space(self) -> None:
        result = pre_tokenize("trailing space ")
        assert result == ["trailing", " space", " "]

    def test_punctuation(self) -> None:
        result = pre_tokenize("no-space,here;now")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_newlines(self) -> None:
        result = pre_tokenize("line1\nline2")
        assert isinstance(result, list)

    def test_single_char(self) -> None:
        assert pre_tokenize("a") == ["a"]
        assert pre_tokenize("1") == ["1"]
        assert pre_tokenize("!") == ["!"]

    def test_empty_string(self) -> None:
        assert pre_tokenize("") == []

    def test_japanese(self) -> None:
        result = pre_tokenize("こんにちは")
        assert "".join(result) == "こんにちは"

    def test_mixed_language(self) -> None:
        result = pre_tokenize("Hello こんにちは 265")
        joined = "".join(result)
        assert joined == "Hello こんにちは 265"

    def test_emoji(self) -> None:
        result = pre_tokenize("😀😀 test")
        joined = "".join(result)
        assert joined == "😀😀 test"

    def test_accented(self) -> None:
        result = pre_tokenize("café naïve")
        joined = "".join(result)
        assert joined == "café naïve"


# ============================================================
# bytes_to_unicode
# ============================================================

class TestBytesToUnicode:
    """Tests for the bytes_to_unicode conversion table.
    bytes_to_unicode変換表のテスト."""

    def test_bijection(self) -> None:
        """All 256 bytes map to unique characters, with no duplicates.
        256バイト全てが重複なくユニークな文字に対応."""
        be = bytes_to_unicode()
        assert len(be) == 256
        assert len(set(be.values())) == 256

    def test_space_is_g_dot(self) -> None:
        """The ASCII space (0x20) is converted to Ġ (U+0120).
        半角スペース(0x20)がĠ(U+0120)に変換される."""
        be = bytes_to_unicode()
        assert be[0x20] == "Ġ"

    def test_printable_ascii_unchanged(self) -> None:
        """Printable ASCII characters are left unchanged.
        印刷可能なASCII文字はそのまま."""
        be = bytes_to_unicode()
        assert be[ord("A")] == "A"
        assert be[ord("!")] == "!"
        assert be[ord("~")] == "~"


# ============================================================
# apply_bpe
# ============================================================

class TestApplyBpe:
    """Tests for the BPE merge logic.
    BPEマージロジックのテスト."""

    def test_simple_merge(self) -> None:
        symbols = ["a", "b", "c"]
        ranks = {("a", "b"): 0, ("ab", "c"): 1}
        assert apply_bpe(symbols, ranks) == ["abc"]

    def test_no_merge_available(self) -> None:
        symbols = ["x", "y", "z"]
        ranks = {("p", "q"): 0}
        assert apply_bpe(symbols, ranks) == ["x", "y", "z"]

    def test_recursive_merge(self) -> None:
        """The recursive merge 'Ġ Ġ' -> 'ĠĠ' -> 'ĠĠĠĠ' seen in merges.txt.
        merges.txtで見た'Ġ Ġ'→'ĠĠ'→'ĠĠĠĠ'の再帰マージ."""
        G = "Ġ"
        symbols = [G, G, G, G]
        ranks = {(G, G): 0, (G + G, G + G): 1}
        assert apply_bpe(symbols, ranks) == [G * 4]

    def test_single_symbol(self) -> None:
        assert apply_bpe(["a"], {("a", "b"): 0}) == ["a"]

    def test_empty(self) -> None:
        assert apply_bpe([], {}) == []


# ============================================================
# decode (round-trip)
# ============================================================

class TestDecode:
    """Tests for decode_tokens.
    decode_tokensのテスト."""

    def test_ascii_round_trip(self) -> None:
        """An ASCII string survives an encode -> decode round trip.
        ASCII文字列がencode→decodeで元に戻る."""
        be = bytes_to_unicode()
        bd = build_byte_decoder(be)
        mini_vocab = {"What": 0, "Ġis": 1, "Ġfun": 2}
        id_to_token = {v: k for k, v in mini_vocab.items()}
        result = decode_tokens([0, 1, 2], id_to_token, bd)
        assert result == "What is fun"

    def test_japanese_round_trip(self) -> None:
        """Japanese (multi-byte) text decodes correctly.
        日本語(マルチバイト)が正しくdecodeされる."""
        be = bytes_to_unicode()
        bd = build_byte_decoder(be)
        a_bytes = "あ".encode("utf-8")
        a_symbols = "".join(be[b] for b in a_bytes)
        id_to_token = {0: a_symbols}
        result = decode_tokens([0], id_to_token, bd)
        assert result == "あ"

    def test_byte_decoder_is_inverse(self) -> None:
        """byte_decoder is a complete, exact inverse of byte_encoder.
        byte_decoderがbyte_encoderの完全な逆引きになっている."""
        be = bytes_to_unicode()
        bd = build_byte_decoder(be)
        for byte_val, char in be.items():
            assert bd[char] == byte_val


# ============================================================
# encode (top-level pipeline: pre_tokenize -> BPE -> vocab lookup)
# ============================================================

class TestEncode:
    """Tests for encode(): pre_tokenize + bpe_encode_piece + tokens_to_ids.
    encode()のテスト: pre_tokenize+bpe_encode_piece+tokens_to_idsの結合."""

    def test_single_piece_fully_merges_into_one_token(self) -> None:
        """With full merge_ranks, one piece merges into a single token ID.
        merge_ranksが揃っていれば、1ピースが1トークンIDへ完全にマージされる."""
        merge_ranks = {("a", "b"): 0, ("ab", "c"): 1}
        vocab = {"abc": 0}
        assert encode("abc", merge_ranks, vocab) == [0]

    def test_concatenates_ids_across_pretoken_pieces_in_order(self) -> None:
        """IDs stay ordered even when concatenated across pre-token pieces.
        複数のpre-tokenピースに跨っても、IDが順序通りに連結される."""
        be = bytes_to_unicode()
        # "a b" -> pre_tokenize -> ["a", " b"]; no merges, so each byte-symbol
        # ("a", "Ġ", "b") stays its own token.
        vocab = {be[ord("a")]: 0, be[0x20]: 1, be[ord("b")]: 2}
        merge_ranks: dict[tuple[str, str], int] = {}
        assert encode("a b", merge_ranks, vocab) == [0, 1, 2]

    def test_unknown_token_raises_key_error(self) -> None:
        """A token missing from vocab raises KeyError (no unknown tokens).
        vocabに存在しないトークンに出会うとKeyErrorになる(未知語は許容しない)."""
        merge_ranks: dict[tuple[str, str], int] = {}
        vocab: dict[str, int] = {}
        with pytest.raises(KeyError):
            encode("a", merge_ranks, vocab)

    def test_round_trips_through_decode_tokens(self) -> None:
        """IDs from encode() survive a round trip through decode_tokens.
        encodeしたIDをdecode_tokensに通すと、元のテキストに戻る."""
        text = "Hi there, 5!"
        be = bytes_to_unicode()
        bd = build_byte_decoder(be)
        needed_bytes = sorted(set(text.encode("utf-8")))
        vocab = {be[b]: i for i, b in enumerate(needed_bytes)}
        merge_ranks: dict[tuple[str, str], int] = {}

        ids = encode(text, merge_ranks, vocab)
        id_to_token = {v: k for k, v in vocab.items()}
        assert decode_tokens(ids, id_to_token, bd) == text
