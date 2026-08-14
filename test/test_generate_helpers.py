from typing import cast

import numpy as np
import pytest

from llm_sdk.llm_sdk import Small_LLM_Model
from src.generate import (
    get_allowed_ids,
    get_union_allowed_ids,
    update_candidates,
    vocab_id_to_token,
    build_dynamic_prompt,
    get_parameter_type_list,
    masked_argmax,
    _generate_num_value
)

from src.model import FunctionDefinition, ParameterSchema
from src.bpe_tokenizer import bytes_to_unicode


VOCAB = {
    "fn": 0,
    "_add": 1,
    "_numbers": 2,
    "fn_none": 3,
    "a": 4,
    "b": 5,
    "ab": 6,
}


class TestGetAllowedIds:
    def test_prefix_matches_only(self) -> None:
        """Only tokens that prefix-match 'fn_add_numbers' are returned.
        'fn_add_numbers'の前方一致トークンだけが返る."""
        ids = get_allowed_ids("fn_add_numbers", VOCAB)
        assert set(ids) == {VOCAB["fn"]}

    def test_no_match_returns_empty(self) -> None:
        assert get_allowed_ids("zzz", VOCAB) == []

    def test_exact_match(self) -> None:
        ids = get_allowed_ids("ab", VOCAB)
        assert set(ids) == {VOCAB["a"], VOCAB["ab"]}


class TestGetUnionAllowedIds:
    def test_union_across_candidates(self) -> None:
        candidates = {"x": "ab", "y": "fn_none"}
        ids = get_union_allowed_ids(candidates, VOCAB)
        assert VOCAB["a"] in ids
        assert VOCAB["ab"] in ids
        assert VOCAB["fn"] in ids
        assert VOCAB["fn_none"] in ids


class TestUpdateCandidates:
    def test_narrows_to_matching_prefix(self) -> None:
        candidates = {
            "fn_add_numbers": "fn_add_numbers",
            "fn_greet": "fn_greet",
        }
        narrowed = update_candidates(candidates, "fn_a")
        assert set(narrowed.keys()) == {"fn_add_numbers"}
        assert narrowed["fn_add_numbers"] == "dd_numbers"

    def test_keeps_all_candidates_sharing_the_prefix(self) -> None:
        candidates = {"fn_add_numbers": "fn_add_numbers", "fn_none": "fn_none"}
        narrowed = update_candidates(candidates, "fn")
        assert set(narrowed.keys()) == {"fn_add_numbers", "fn_none"}

    def test_eliminates_non_matching(self) -> None:
        candidates = {"true": "true", "false": "false"}
        narrowed = update_candidates(candidates, "t")
        assert set(narrowed.keys()) == {"true"}

    def test_reaches_empty_when_fully_consumed(self) -> None:
        candidates = {"ok": "ok"}
        narrowed = update_candidates(candidates, "ok")
        assert narrowed == {"ok": ""}


class TestVocabIdToToken:
    def test_inverts_mapping(self) -> None:
        inv = vocab_id_to_token(VOCAB)
        for token, idx in VOCAB.items():
            assert inv[idx] == token


class TestBuildDynamicPrompt:
    def test_includes_all_function_names_and_user_request(self) -> None:
        funcs = [
            FunctionDefinition(
                name="fn_add_numbers",
                description="Add two numbers.",
                parameters={"a": {"type": "number"}},
                returns={"type": "number"},
            ),
            FunctionDefinition(
                name="fn_greet",
                description="Greet someone.",
                parameters={"name": {"type": "string"}},
                returns={"type": "string"},
            ),
        ]
        prompt = build_dynamic_prompt("Greet Bob", funcs)
        assert "fn_add_numbers" in prompt
        assert "fn_greet" in prompt
        assert "Greet Bob" in prompt


class TestGetParameterTypeList:
    def test_returns_types_for_chosen_function_only(self) -> None:
        funcs = [
            FunctionDefinition(
                name="fn_add_numbers",
                description="Add two numbers.",
                parameters={"a": {"type": "number"}, "b": {"type": "number"}},
                returns={"type": "number"},
            ),
            FunctionDefinition(
                name="fn_greet",
                description="Greet someone.",
                parameters={"name": {"type": "string"}},
                returns={"type": "string"},
            ),
        ]
        types = get_parameter_type_list("fn_add_numbers", funcs)
        assert types == ["number", "number"]


class _FakeModel:
    """A stub exposing only get_logits_from_input_ids; no real LLM needed.
    get_logits_from_input_idsだけ持つ、実LLM不要のスタブ."""

    def __init__(self, logits: list) -> None:
        self._logits = logits

    def get_logits_from_input_ids(self, input_ids: list) -> list:
        return self._logits


def _fake_model(logits: list) -> Small_LLM_Model:
    """Cast _FakeModel to the Small_LLM_Model type masked_argmax expects.
    masked_argmaxが要求するSmall_LLM_Model型として_FakeModelを渡すためのcast."""
    return cast(Small_LLM_Model, _FakeModel(logits))


class TestMaskedArgmax:
    """The core of constrained decoding: masking actually controls selection.
    制約付きデコーディングの核心: マスキングが実際に選択を制御することを検証."""

    def test_illegal_high_logit_token_is_never_chosen(self) -> None:
        """A token outside allowed_ids is never chosen, however high its logit.
        allowed_idsに無いトークンは、logitがどれだけ高くても選ばれない."""
        logits = [100.0, 1.0, 0.5]  # index 0 has the highest raw score
        model = _fake_model(logits)
        chosen, _ = masked_argmax(
            model, generated=[], allowed_ids=[1, 2], discouraged_ids=[]
        )
        assert chosen != 0
        assert chosen == 1  # highest among the *allowed* ids

    def test_discouraged_is_soft_not_hard(self) -> None:
        """discouraged_ids is -8.0, not -inf; a close margin can still flip.
        discouraged_idsは-8.0の減点であり、僅差なら逆転しうる(-infではない)."""
        logits = [1.0, 0.5]
        model = _fake_model(logits)
        chosen, _ = masked_argmax(
            model, generated=[], allowed_ids=[0, 1], discouraged_ids=[0]
        )
        assert chosen == 1

    def test_discouraged_still_allowed_if_no_alternative(self) -> None:
        """A discouraged id can still win if allowed_ids has no better option.
        discouraged_idsでもallowed_idsに含まれていれば選択され得る(-infではない証拠)."""
        logits = [1.0, -100.0]
        model = _fake_model(logits)
        chosen, _ = masked_argmax(
            model, generated=[], allowed_ids=[0], discouraged_ids=[0]
        )
        assert chosen == 0

    def test_accepts_numpy_array_discouraged_ids(self) -> None:
        """discouraged_ids accepts both a plain list and an np.ndarray.
        discouraged_idsはlistとnp.ndarrayの両方を受け付ける."""
        logits = [1.0, 0.5]
        model = _fake_model(logits)
        chosen, _ = masked_argmax(
            model,
            generated=[],
            allowed_ids=[0, 1],
            discouraged_ids=np.array([0], dtype=np.int64),
        )
        assert chosen == 1

    def test_returns_raw_unmasked_logits_alongside_choice(self) -> None:
        """The 2nd return value is the raw logits array, before masking.
        2つ目の戻り値は、マスク適用前の生logits配列そのもの."""
        logits = [100.0, 1.0, 0.5]
        model = _fake_model(logits)
        _, raw_logits = masked_argmax(
            model, generated=[], allowed_ids=[1, 2], discouraged_ids=[]
        )
        assert list(raw_logits) == logits


def _build_number_vocab(
    prompt_text: str,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Build a vocab that byte-level encodes prompt_text as one token per char.
    promt_textをbyte-levelで1文字=1トークンとしてencode可能なvocabを作る.

    Leaving merge_ranks empty makes apply_bpe perform no merges at all, so
    each byte-level symbol is returned as its own token (a minimal setup
    for tests).
    merge_ranksを空にすることで、apply_bpeが何もマージせずバイト単位の
    シンボルをそのままトークンとして返すようにする(テスト用の最小構成)。
    """
    byte_encoder = bytes_to_unicode()
    symbols = {byte_encoder[b] for b in prompt_text.encode("utf-8")}
    symbols |= {byte_encoder[ord(c)] for c in "0123456789,.}"}
    vocab = {sym: i for i, sym in enumerate(sorted(symbols))}
    return vocab, {}


class _AlwaysDigitModel:
    """A stub maximizing one digit token's logit, for runaway generation.
    常に同じ数字トークンのlogitを最大にする、暴走生成を再現するためのスタブ."""

    def __init__(self, digit_id: int, vocab_size: int) -> None:
        self._digit_id = digit_id
        self._vocab_size = vocab_size

    def get_logits_from_input_ids(self, input_ids: list) -> list:
        logits = [0.0] * self._vocab_size
        logits[self._digit_id] = 100.0
        return logits


class TestGenerateNumberValueRunawayGuard:
    """What backs the README's "very large numbers" claim: the runaway guard.
    READMEが謳う「very large numbers」耐性の実体: 暴走生成防止ガード."""

    def test_raises_on_runaway_digit_generation(self) -> None:
        """Endless same-digit choices raise RuntimeError past 15 characters.
        モデルが際限なく同じ数字を選び続けると、15文字を超えた時点でRuntimeErrorになる."""
        vocab, merge_ranks = _build_number_vocab('"amount": ')
        id_to_token = {v: k for k, v in vocab.items()}
        byte_encoder = bytes_to_unicode()
        digit_id = vocab[byte_encoder[ord("5")]]
        model = cast(
            Small_LLM_Model, _AlwaysDigitModel(digit_id, len(vocab))
        )
        schema = ParameterSchema(type="number")

        with pytest.raises(RuntimeError, match="Runaway number generation"):
            _generate_num_value(
                "amount",
                schema,
                "",
                model,
                vocab,
                id_to_token,
                verbose=False,
                merge_ranks=merge_ranks,
            )
