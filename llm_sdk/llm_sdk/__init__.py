# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import time
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    logging,
)
from huggingface_hub import hf_hub_download
import os


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps`` when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16`` to keep memory
        usage reasonable; on CPU we keep ``float32`` for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

        # KV cache state: the past_key_values from the most recent forward
        # pass, plus the exact input_ids that produced it. Used to detect,
        # on the next call, whether the new input_ids are just the previous
        # ones plus a few appended tokens - in which case only those new
        # tokens need to be run through the model, reusing the cached
        # attention Key/Value tensors for everything before them.
        self.kv_cache = None
        self._cached_ids: list[int] = []

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    # def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
    #     """
    #     Given a list of input token ids, return the raw logits (no softmax) for the next token.
    #     """
    #     input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
    #     with torch.no_grad():
    #         out = self._model(input_ids=input_tensor)
    #     # Get logits for the last token in the sequence for the batch (batch size 1)
    #     logits = out.logits[0, -1].tolist()
    #     return [float(x) for x in logits]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Given a list of input token ids, return the raw logits (no softmax) for the next token.

        Transparently reuses the KV cache from the previous call when
        ``input_ids`` is simply the previous call's ``input_ids`` with a
        few new tokens appended (the common case when generating token by
        token in a loop): only the new tokens are actually run through the
        model, and the cached Key/Value tensors for everything before them
        are reused instead of being recomputed. If ``input_ids`` does not
        extend the previous call (e.g. a new prompt, or backtracking), the
        cache is discarded and a full forward pass is run instead. Either
        way, the returned logits are identical to always doing a full
        forward pass - this only changes how much work it takes to get
        them.
        """
        reuse_cache = (
            self.kv_cache is not None
            and len(input_ids) >= len(self._cached_ids)
            and input_ids[: len(self._cached_ids)] == self._cached_ids
        )
        if reuse_cache:
            new_ids = input_ids[len(self._cached_ids):]
            if not new_ids:
                new_ids = input_ids[-1:]
            past_key_values = self.kv_cache
        else:
            new_ids = input_ids
            past_key_values = None

        input_tensor = torch.tensor(
            [new_ids], device=self._device, dtype=torch.long
        )
        with torch.no_grad():
            out = self._model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

        self.kv_cache = out.past_key_values
        self._cached_ids = list(input_ids)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> str:
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file", "vocab.json"
        )
        vocab_path = hf_hub_download(repo_id=self._model_name, filename=vocab_file_name)
        return vocab_path

    def get_path_to_merges_file(self) -> str:
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file", "merges.txt"
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name, filename=merges_file_name
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> str:
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file", "tokenizer.json"
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name, filename=tokenizer_file_name
        )
        return tokenizer_path
