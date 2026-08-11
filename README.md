*This project has been created as part of the 42 curriculum by tkunugi.*

# call me maybe — Constrained Decoding for LLM Function Calling

## Description

This project turns a natural-language request into a **structured function call**, using a
small open-weights language model (`Qwen/Qwen3-0.6B`).

Given the prompt `"What is the sum of 40 and 2?"`, the program does **not** answer `42`.
It answers *which tool should be called, and with which arguments*:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The core of the project is **constrained decoding**: instead of asking the model politely
to produce JSON and hoping for the best, the JSON skeleton is written by the program, and
the model is only consulted at the points where a real decision has to be made. At every
such decision point the set of allowed next tokens is computed from the schema, and every
other token in the vocabulary is masked out of the logits before selection.

The result is that **invalid JSON is not merely unlikely — it is unreachable**. The output
is 100% parseable by construction, and the field names, function names and argument types
are guaranteed to match `functions_definition.json`.

The whole pipeline is implemented from scratch with **only `numpy`, `json` and `pydantic`**.
No `transformers`, no `outlines`, no `dspy`, no PyTorch in the submitted code.

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) package manager
- The `llm_sdk/` package, placed next to `src/`

### Installation

```bash
uv sync
```

The model weights are downloaded on first run and cached under `~/.cache/huggingface/hub/`.

### Running

```bash
uv run python -m src
```

With explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input  data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target        | Effect                                                    |
| ------------- | --------------------------------------------------------- |
| `install`     | Install dependencies via `uv sync`                        |
| `run`         | Run the program with default paths                        |
| `debug`       | Run under `pdb`                                           |
| `clean`       | Remove `__pycache__`, `.mypy_cache`, generated output      |
| `lint`        | `flake8 .` and `mypy .` with the mandated flags            |
| `lint-strict` | `flake8 .` and `mypy . --strict`                           |

---

## Example usage

```bash
$ uv run python -m src
$ cat data/output/function_calling_results.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```


Verbose mode prints the generation process step by step to `stderr`, so that the
generated JSON on `stdout` stays clean and pipeable:

```bash
$ uv run python -m src --verbose 2> trace.log
```

---

## Algorithm explanation

### 1. Parse and validate the inputs (fail fast)

Validation happens in three ordered stages, each with its own exception type:

| Stage | Question                                    | Exception           |
| ----- | ------------------------------------------- | ------------------- |
| 1     | Does the file exist and can it be read?     | `InputFileError`    |
| 2     | Is it syntactically valid JSON?             | `InputFormatError`  |
| 3     | Does it match the expected schema?          | `InputSchemaError`  |

The order matters: a missing file must not be reported as a schema error. Stage 3 is
handled by **pydantic** models (`PromptEntry`, `FunctionDefinition`, `ParameterSchema`),
so schema conformance is checked declaratively rather than with hand-written `if` chains.

Crucially, the model is **not** instantiated until validation passes. Loading a 0.6B model
takes seconds; there is no reason to pay that cost before knowing the inputs are usable.

### 2. Write the skeleton, ask only at the decision points

The output object has a fixed shape. Only three kinds of content are unknown:

1. the **function name**,
2. the **argument names** (determined by the chosen function),
3. the **argument values**.

Everything else — `{`, `"name"`, `:`, `,`, `"parameters"`, `}` — is punctuation the schema
already determines. Generating it token by token would mean running a forward pass to
"discover" a character we already know.

So the generator writes the fixed segments directly into the token buffer and only calls
the model at the real decision points. This technique is known as **coalescence**. It cuts
the number of forward passes by a large factor with no loss of correctness, because the
skipped tokens had exactly one legal continuation anyway.

The same reasoning applies to the `"prompt"` field: its value is copied from the input,
never generated.

### 3. Constrained token selection

At a decision point, the algorithm is:

1. Run a forward pass to obtain the logits — one score per vocabulary entry.
2. Compute the set of **candidate strings** still legal at this position. For the function
   name, this is the list of names in `functions_definition.json` that start with what has
   already been emitted.
3. Map candidate strings to **candidate token IDs** by prefix matching against the
   vocabulary, rather than by scanning the full vocabulary for exact matches.
4. Build a mask: `mask = np.full(vocab_size, -np.inf)` , then `mask[candidate_ids] = 0.0`.
   Add it to the logits.
5. Take the `argmax`. Append the chosen token. Repeat until the candidate set collapses to
   a single completed string.

Because the mask is `-inf` everywhere except on legal tokens, an illegal token cannot be
selected even if the model assigns it overwhelming probability. **The schema wins,
always.**

A softer variant of the same mechanism is used for argument values that must be
*synthesised* rather than copied: instead of `-inf`, a finite penalty (`-10.0`) is applied
to discouraged tokens. This biases the model without forbidding, which is the right tool
when the correct answer is not enumerable in advance.

### 4. Why prefix matching, not exact matching

A given string has more than one valid tokenization. `fn_add_numbers` might be one token,
or `fn` + `_add` + `_numbers`, or many other splits. Enumerating every tokenization of
every candidate is combinatorial. Instead, the algorithm asks a local question at each
step: *"which single tokens are a valid prefix of at least one remaining candidate?"* That
question has a cheap answer, and iterating it walks the model down a legal path
regardless of which split it prefers.

### 5. Extraction vs. synthesis

Not all arguments are the same kind of problem:

- **Extraction** — the value appears verbatim in the prompt (`"Reverse the string 'hello'"`
  → `"hello"`). Here the model only has to locate and copy.
- **Synthesis** — the value must be produced from knowledge (`"match any email address"` →
  a regex). Here the model has to actually know something.

These two pull in opposite directions. Prompt instructions that push the model to copy
verbatim improve extraction accuracy and *degrade* synthesis accuracy — a see-saw effect
that showed up clearly in testing. The final prompt template and the logit-bias strategy
are both a deliberate compromise between the two.

---

## Design decisions

**Single source of truth.** `functions_definition.json` drives everything: the candidate
list for constrained decoding, the argument names, the type constraints, *and* the
few-shot prompt template, which is built dynamically from the definitions. Nothing about
the example functions is hardcoded. Swapping in a completely different function set
requires no code change — which is exactly what the peer review will do.

**Pydantic at the boundaries, plain data inside.** Every value entering the program from a
file passes through a pydantic model. Once inside, the data is trusted. This keeps
validation in one obvious place instead of scattering defensive checks through the
generation loop.

**Restricted dependency set as a feature, not a constraint.** The prohibition on
`transformers` / `outlines` forces the constrained-decoding logic to be written explicitly
rather than delegated to a library. The practical consequence is a system with a very
small, fully auditable dependency surface that runs entirely on local weights, with no
external API call and no data leaving the machine. That is precisely the deployment
profile required in regulated or IP-sensitive industrial environments.

**Verbose output on `stderr`.** Diagnostics must never contaminate the data stream. The
`--verbose` / `-v` flag sets a module-level switch in `src/utils.py`; `vprint()` writes to
`stderr` only when the switch is on.

**Tagging before branching.** Every stable state is git-tagged before a new feature branch
is opened, and merged back only after the full test set is re-verified. "Freeze" here
means *tag then attack on a branch*, not *never touch again*.

---

## Features

### Mandatory

- Constrained decoding with logit masking over a schema-derived candidate set
- Dynamic prompt construction from `functions_definition.json`
- Three-stage fail-fast input validation with pydantic
- Full CLI with `--functions_definition`, `--input`, `--output`
- `flake8` and `mypy` clean

### Bonus

- **Multiple model support** — `--model` accepts any compatible checkpoint. Verified with
  `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`.
- **Nested function arguments** — recursive value generation backed by a recursive
  pydantic type (`RootModel[Union[str, float, bool, Dict[str, "ParameterValue"]]]` with
  `model_rebuild()`), so objects can nest to arbitrary depth.
- **Generation visualisation** — `--verbose` traces each decision point: candidate set,
  masked vocabulary size, chosen token.
- **Tokenizer reimplementation** <!-- TODO: keep only if shipped --> — byte-level BPE
  implemented from the vocabulary and merge files exposed by the SDK, removing the direct
  dependency on `encode` in the main code path.

---

## Performance analysis

### Accuracy

On the reference test set of 11 prompts:

| Metric                    | Result       |
| ------------------------- | ------------ |
| Correct function selected | 11/11 (100%) |
| Correct arguments         | 11/11 (100%) |
| Valid, parseable JSON     | 11/11 (100%) |
| Schema-conformant output  | 11/11 (100%) |

The last two rows are guaranteed by the algorithm rather than measured — with `-inf`
masking there is no path through the generator that emits malformed JSON. The first two
are genuinely measured and genuinely fallible: constrained decoding guarantees *shape*,
not *semantics*. Nothing prevents the model from confidently selecting the wrong function;
it only prevents it from selecting a non-existent one.
<!--TODO: function Noneのケース-->
### Speed

Well under the 5-minute budget on the reference hardware
(<!-- TODO: fill in wall-clock time --> on an Intel i5-4570, 4 cores, CPU only, no
acceleration). Coalescence is the dominant factor: skipping the structural tokens removes
the large majority of forward passes.

Scaling to `Qwen3-1.7B` increases runtime roughly in proportion to parameter count.
Notably, accuracy is **not** monotonic in model size under greedy decoding — the larger
model is not uniformly better on this task. Constrained decoding narrows the search space
enough that the smaller model is already sufficient, which is the practically interesting
result: the cheap model plus good structure beats the expensive model plus hope.

### Reliability

Malformed input, missing files, and schema violations are each caught at their own stage
and reported with a message naming the file and the problem. The program exits non-zero
and never raises an unhandled exception.

---

## Challenges faced

### Composite tokens breaking the termination condition

Termination detection originally looked for a closing quote character to know that a
string value was finished. But the tokenizer's vocabulary contains *composite* tokens — a
single token can be `)"`, carrying both a payload character and the terminator. Breaking
on sight of the quote discarded the `)`, silently truncating regex arguments.

**Fix:** before breaking, split the decoded token on `"` and append the pre-quote prefix.
The character is recovered, then generation stops. Reproduced with a minimal case, fixed
on a branch, verified against the full test set before merging.

### Digits arriving one at a time

Numbers were being built digit by digit, which seemed like a bug in the generation loop.
Reading the tokenizer configuration showed it was not: the pre-tokenizer's regex contains
`\p{N}` **with no quantifier**, so digits are always isolated into single-character
pre-tokens before BPE ever runs. Number generation is inherently one digit per step. The
loop was correct; the expectation was wrong.

### Greedy tokenization lock-in

Once a token path is committed to, alternative tokenizations of the same target string
become structurally unreachable. Greedy argmax can therefore paint itself into a corner.
Prefix-based candidate matching mitigates this; beam search is the textbook remedy and is
the natural next step.

### Recursive pydantic models

A self-referential type for nested arguments raises `RecursionError` if declared as a bare
`TypeAlias`. The working form is `RootModel` plus an explicit `model_rebuild()` call
immediately after the class definition. A second, subtler issue: mutating a dict in place
bypasses validation entirely — values must be wrapped in the model constructor for
validation to actually run.

### Stale variable reuse

The same bug class appeared three separate times: a logits array, a loop counter, and a
candidate dictionary each carried a value from the previous iteration into the next. Once
identified as a pattern rather than three unrelated bugs, it became something to check for
deliberately at every loop boundary.

### Multi-byte Character Decode Pitfall

Byte-level BPE tokenizers (used by GPT-2, Qwen, and most modern LLMs) operate on raw bytes, not characters. This means a single multi-byte character like a Japanese kanji can be split across multiple tokens when it is rare enough that no BPE merge rule exists for its byte sequence.

When decoding token-by-token with model.decode(token_id), each fragment is an incomplete UTF-8 sequence. Rather than raising an error, the tokenizer silently replaces it with the Unicode replacement character (U+FFFD, displayed as �). This is by design — both Python's errors="replace" and Rust's from_utf8_lossy implement this behavior.

I verified this experimentally using my own name:

Input	Tokens	Per-token decode	Batch decode
太郎	2	太郎 ✅	太郎 ✅
㓛刀	4	���刀 ❌	㓛刀 ✅

The character 㓛 (U+34DB, CJK Extension A) is not a 常用漢字 (jōyō kanji) and rarely appears in training data. Its 3 UTF-8 bytes received zero BPE merges, resulting in 3 separate tokens — each producing � when decoded individually.

Fix: Instead of decoding each token as it is generated, I accumulate token IDs during the generation loop and decode all of them in a single call at the end. This ensures all bytes are present before UTF-8 decoding occurs.

This issue is documented in the paper "Byte-level Tokenizers Unavoidably Enable LLMs to Generate Ill-formed UTF-8" and is not specific to any one model — it is a structural property of the byte-level BPE architecture itself.

The existing term "subword" assumes that each token fragment is at least a readable character — which holds for English (1 byte = 1 character) but not for multi-byte languages like Japanese. A single byte from 㓛 is not a "sub-word"; it is not even a character. As no established term exists for these incomplete fragments, I coin the term byte-fragment: a token produced by byte-level BPE that represents an incomplete portion of a multi-byte character — not a subword, not a character, but a raw byte-level residue that cannot be decoded to valid text on its own.

---

## Testing strategy

- **Reference set** — the 11 provided prompts, run after every change, compared against
  expected function name and arguments.
- **Regression gate** — no branch is merged until the full reference set passes again.
- **Adversarial prompts** — empty strings, very large numbers, special characters and
  quotes inside string arguments, prompts matching no function well, and functions with
  multiple parameters.
- **Alternate function sets** — a second `functions_definition.json` with entirely
  different function names and parameter names, to confirm nothing is hardcoded.
- **Malformed input files** — missing file, empty file, invalid JSON syntax, valid JSON of
  the wrong shape; each must produce its own specific error message.
- **Static analysis** — `flake8` and `mypy` in the mandated configuration, run as a gate
  rather than an afterthought.

---

## Resources

### Constrained decoding and structured generation

- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023) — the
  paper behind `outlines`; the source of the coalescence idea.
- `outlines` documentation — read for design ideas, not used in the code.

### Tokenization

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*
  (2016) — the original BPE paper.
- OpenAI GPT-2 `encoder.py` — the reference implementation of byte-level BPE and the
  `bytes_to_unicode` mapping.
- HuggingFace `tokenizers` documentation — pre-tokenizer and normalizer pipeline.

### Tooling

- pydantic v2 documentation, in particular recursive models and `model_rebuild()`
- `uv` documentation
- Qwen3 model card

### Use of AI

An AI assistant (Claude) was used as a study and review partner, not as a code generator.
Specifically:

- **Explaining concepts** — byte-level BPE, the role of the pre-tokenizer regex, logit
  masking, recursive pydantic models. Every explanation was verified against primary
  sources (the papers and reference implementations listed above) before being relied on.
- **Confirming diagnoses** — bugs in this project were typically located first and then
  discussed for confirmation and for the general principle behind them. The composite-token
  bug and the stale-variable pattern are examples: root cause identified independently,
  then discussed to generalise.
- **Reviewing structure** — discussing the ordering of the validation stages and the
  organisation of this README.
