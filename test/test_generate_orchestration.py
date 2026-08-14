"""Tests for generate.py's orchestration functions (stubs replace the LLM).
generate.pyのオーケストレーション関数のテスト(実LLMは使わずスタブで代替).

generate_function_call/verify_function_choice are driven with token-level
fake models; engine() has generate_function_call/verify_function_choice/
generate_parameter themselves monkeypatched, isolating just its control
flow (continuing after one failure, falling back to fn_none on failed
verification, etc.).
generate_function_call/verify_function_choiceはトークンレベルのフェイクモデルで、
engineはgenerate_function_call/verify_function_choice/generate_parameter自体を
monkeypatchして、制御フロー(1件失敗しても続行、検証失敗でfn_noneへ落ちる等)だけを
切り出して検証する。
"""
from pathlib import Path
from typing import Any, cast

import pytest

import src.generate as generate_module
from llm_sdk.llm_sdk import Small_LLM_Model
from src.generate import (
    build_dynamic_prompt,
    generate_function_call,
    verify_function_choice,
    vocab_id_to_token,
)
from src.bpe_tokenizer import bytes_to_unicode
from src.model import (
    FunctionDefinition,
    ParameterFetch,
    ParameterValue,
    PromptWrite,
)


def _build_char_vocab(*texts: str) -> dict[str, int]:
    """A merge-free vocab containing only the byte-symbols seen in `texts`.
    与えられたテキスト群に登場するバイトシンボルだけを含む、マージ無しvocab.

    Used with empty merge_ranks, this gives a minimal one-char-per-token
    setup that drives encode()/prefix-matching without real BPE data
    (same idea as test_generate_helpers.py's _build_number_vocab).
    merge_ranksを空にして使うことで、1文字=1トークンの最小構成になり、
    encode()やprefix-matchingを本物のBPEデータ無しで駆動できる
    (test_generate_helpers.pyの_build_number_vocabと同じ考え方)。
    """
    seed = '{}[]":,. \n-0123456789truefalsenull'
    be = bytes_to_unicode()
    symbols = {be[b] for b in seed.encode("utf-8")}
    for text in texts:
        symbols |= {be[b] for b in text.encode("utf-8")}
    return {sym: i for i, sym in enumerate(sorted(symbols))}


class _SequentialTargetModel:
    """A stub maximizing only the next char of `target` on each call.
    呼ばれるたびに`target`の次の1文字のlogitだけを最大化するスタブ.

    Each masked_argmax call is one decode step, so this walks the decoder
    down exactly the `target` path - verifying prefix-matching converges
    on the right candidate without a real model.
    masked_argmaxの呼び出し1回=デコードの1ステップに対応するため、これは
    デコーダを`target`という一本道だけに沿って歩かせる - prefix-matchingが
    正しい候補へ収束することを、本物のモデル無しで検証できる。
    """

    def __init__(self, target: str, vocab: dict[str, int]) -> None:
        be = bytes_to_unicode()
        self._target_ids = [vocab[be[b]] for b in target.encode("utf-8")]
        self._vocab_size = len(vocab)
        self._step = 0

    def get_logits_from_input_ids(self, input_ids: list) -> list:
        logits = [0.0] * self._vocab_size
        if self._step < len(self._target_ids):
            logits[self._target_ids[self._step]] = 100.0
        self._step += 1
        return logits


class _FixedLogitsModel:
    """A stub returning fixed logits, for a single yes/no-style decision.
    常に同じlogits配列を返す、単発の決定(yes/no等)を検証するためのスタブ."""

    def __init__(self, logits: list) -> None:
        self._logits = logits

    def get_logits_from_input_ids(self, input_ids: list) -> list:
        return self._logits


class TestGenerateFunctionCall:
    """Function-name selection: prefix matching converges on the right name.
    関数名選択: prefix-matchingで候補が正しく収束するかを検証."""

    def test_walks_to_the_target_function_via_prefix_matching(self) -> None:
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
        user_prompt = "What is the sum of 2 and 3?"
        full_prompt = build_dynamic_prompt(user_prompt, funcs)
        vocab = _build_char_vocab(full_prompt, "fn_none")
        id_to_token = vocab_id_to_token(vocab)
        model = cast(
            Small_LLM_Model, _SequentialTargetModel("fn_add_numbers", vocab)
        )

        chosen = generate_function_call(
            user_prompt, funcs, model, vocab, id_to_token,
            verbose=False, merge_ranks={},
        )

        assert chosen == "fn_add_numbers"

    def test_falls_back_to_fn_none_when_no_function_fits(self) -> None:
        funcs = [
            FunctionDefinition(
                name="fn_add_numbers",
                description="Add two numbers.",
                parameters={"a": {"type": "number"}},
                returns={"type": "number"},
            ),
        ]
        user_prompt = "Completely unrelated request"
        full_prompt = build_dynamic_prompt(user_prompt, funcs)
        vocab = _build_char_vocab(full_prompt, "fn_none")
        id_to_token = vocab_id_to_token(vocab)
        model = cast(Small_LLM_Model, _SequentialTargetModel("fn_none", vocab))

        chosen = generate_function_call(
            user_prompt, funcs, model, vocab, id_to_token,
            verbose=False, merge_ranks={},
        )

        assert chosen == "fn_none"


class TestVerifyFunctionChoice:
    """The second opinion: whether the yes/no logit gap clears the margin.
    セカンドオピニオン: yes/noのlogit差がマージンを超えるかどうかの判定."""

    def _vocab(self, user_prompt: str, chosen_func: str) -> dict[str, int]:
        verify_prompt = (
            f"Is this selected function appropriate for the User request? "
            f'User request: "{user_prompt}"\n'
            f"Selected function: {chosen_func}\n"
            "Answer: "
        )
        vocab = _build_char_vocab(verify_prompt)
        vocab["yes"] = len(vocab)
        vocab["no"] = len(vocab)
        return vocab

    def test_confident_yes_passes(self) -> None:
        user_prompt, chosen_func = "Add 2 and 3", "fn_add_numbers"
        vocab = self._vocab(user_prompt, chosen_func)
        logits = [0.0] * len(vocab)
        logits[vocab["yes"]] = 5.0
        logits[vocab["no"]] = 0.0
        model = cast(Small_LLM_Model, _FixedLogitsModel(logits))

        result = verify_function_choice(
            user_prompt, chosen_func, model, vocab,
            merge_ranks={}, verbose=False,
        )

        assert result is True

    def test_weak_yes_is_rejected_by_margin(self) -> None:
        """A narrow "yes" win stays under the 2.3 threshold, counted as False.
        "yes"が僅差で勝つだけでは閾値(2.3)未満としてFalse扱いになる."""
        user_prompt, chosen_func = "Add 2 and 3", "fn_add_numbers"
        vocab = self._vocab(user_prompt, chosen_func)
        logits = [0.0] * len(vocab)
        logits[vocab["yes"]] = 1.0
        logits[vocab["no"]] = 0.0
        model = cast(Small_LLM_Model, _FixedLogitsModel(logits))

        result = verify_function_choice(
            user_prompt, chosen_func, model, vocab,
            merge_ranks={}, verbose=False,
        )

        assert result is False

    def test_clear_no_is_rejected(self) -> None:
        user_prompt, chosen_func = "Add 2 and 3", "fn_reverse_string"
        vocab = self._vocab(user_prompt, chosen_func)
        logits = [0.0] * len(vocab)
        logits[vocab["no"]] = 5.0
        logits[vocab["yes"]] = 0.0
        model = cast(Small_LLM_Model, _FixedLogitsModel(logits))

        result = verify_function_choice(
            user_prompt, chosen_func, model, vocab,
            merge_ranks={}, verbose=False,
        )

        assert result is False


class _MergesOnlyModel:
    """A stub with only the get_path_to_merges_file() engine() calls directly.
    engine()が直接呼ぶget_path_to_merges_file()だけを持つスタブ.

    Every other model call inside engine() is replaced by monkeypatching
    generate_function_call and friends, so nothing else is needed here.
    engine()内の他のモデル呼び出しは、すべてgenerate_function_call等の
    monkeypatchで置き換えられるため、ここでは不要。
    """

    def __init__(self, merges_path: str) -> None:
        self._merges_path = merges_path

    def get_path_to_merges_file(self) -> str:
        return self._merges_path


def _make_parser(prompts: list[str], funcs: list[FunctionDefinition]) -> Any:
    class _Parser:
        prompt_list = [PromptWrite(prompt=p) for p in prompts]
        func_list = funcs

    return _Parser()


class TestEngine:
    """engine()'s control flow: fn_none on bad verify, batch survives failures.
    engine()の制御フロー: 検証失敗時のfn_none化、1件失敗時のバッチ継続."""

    def _model(self, tmp_path: Path) -> Small_LLM_Model:
        merges_path = tmp_path / "merges.txt"
        merges_path.write_text("#version: dummy\n", encoding="utf-8")
        return cast(Small_LLM_Model, _MergesOnlyModel(str(merges_path)))

    def test_happy_path_records_name_and_parameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        func = FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
            returns={"type": "number"},
        )
        parser = _make_parser(["Add 2 and 3"], [func])

        monkeypatch.setattr(
            generate_module, "generate_function_call",
            lambda *a, **k: "fn_add_numbers",
        )
        monkeypatch.setattr(
            generate_module, "verify_function_choice",
            lambda *a, **k: True,
        )

        def fake_generate_parameter(
            param_fetch_dict: ParameterFetch, *a: Any, **k: Any
        ) -> str:
            param_fetch_dict.parameters["a"] = ParameterValue(2.0)
            param_fetch_dict.parameters["b"] = ParameterValue(3.0)
            return ""

        monkeypatch.setattr(
            generate_module, "generate_parameter", fake_generate_parameter
        )

        result = generate_module.engine(
            parser, self._model(tmp_path), vocab={}, verbose=False
        )

        assert result == [
            {
                "prompt": "Add 2 and 3",
                "name": "fn_add_numbers",
                "parameters": {"a": 2.0, "b": 3.0},
            }
        ]

    def test_failed_verification_falls_back_to_fn_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        func = FunctionDefinition(
            name="fn_add_numbers",
            description="Add.",
            parameters={"a": {"type": "number"}},
            returns={"type": "number"},
        )
        parser = _make_parser(["irrelevant prompt"], [func])

        monkeypatch.setattr(
            generate_module, "generate_function_call",
            lambda *a, **k: "fn_add_numbers",
        )
        monkeypatch.setattr(
            generate_module, "verify_function_choice", lambda *a, **k: False
        )
        called = {"generate_parameter": False}

        def spy_generate_parameter(*a: Any, **k: Any) -> str:
            called["generate_parameter"] = True
            return ""

        monkeypatch.setattr(
            generate_module, "generate_parameter", spy_generate_parameter
        )

        result = generate_module.engine(
            parser, self._model(tmp_path), vocab={}, verbose=False
        )

        assert result == [
            {
                "prompt": "irrelevant prompt",
                "name": "fn_none",
                "parameters": {},
            }
        ]
        assert called["generate_parameter"] is False

    def test_one_prompt_failure_does_not_abort_the_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One runaway generation (RuntimeError) doesn't stop other prompts.
        ある1件の暴走生成(RuntimeError)が、他のプロンプトの処理を止めない."""
        func = FunctionDefinition(
            name="fn_add_numbers",
            description="Add.",
            parameters={"a": {"type": "number"}},
            returns={"type": "number"},
        )
        parser = _make_parser(["prompt one", "prompt two"], [func])

        def flaky_generate_function_call(
            user_prompt: str, *a: Any, **k: Any
        ) -> str:
            if user_prompt == "prompt one":
                raise RuntimeError("simulated runaway generation")
            return "fn_add_numbers"

        monkeypatch.setattr(
            generate_module, "generate_function_call",
            flaky_generate_function_call,
        )
        monkeypatch.setattr(
            generate_module, "verify_function_choice", lambda *a, **k: True
        )
        monkeypatch.setattr(
            generate_module, "generate_parameter", lambda *a, **k: ""
        )

        result = generate_module.engine(
            parser, self._model(tmp_path), vocab={}, verbose=False
        )

        assert result[0] == {
            "prompt": "prompt one", "name": "fn_none", "parameters": {}
        }
        assert result[1]["name"] == "fn_add_numbers"
        assert result[1]["prompt"] == "prompt two"
