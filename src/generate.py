import numpy as np
import re
from typing import Any
import argparse

from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser
from .model import FunctionDefinition, ParameterFetch, ParameterValue
from .bpe_tokenizer import encode, decode_tokens
from .bpe_tokenizer import bytes_to_unicode, build_byte_decoder


def vprint(to_print: str, verbose: argparse.Namespace) -> None:
    """Print a diagnostic line only when verbose mode is on.
    Used throughout the generation pipeline so that ``--verbose`` traces
    never leak into the pipeline's actual JSON output.
    Args:
        to_print: The message to print.
        verbose: Truthy to enable printing; falsy to suppress it.
    """
    if verbose:
        print(to_print)


def get_allowed_ids(remaining: str, vocab: dict[str, int]) -> list[int]:
    """Return token IDs that can legally start spelling remaining.
    Args:
        renaining: target srt which are not written ex) "fn_greet"
        vocab: token str -> ID dict
    Returns:
        IDs list of all tokens which match the begining of remaining
    """
    allowed = []
    for i in range(1, len(remaining) + 1):
        token = remaining[:i]
        # print(token)
        if token in vocab:
            # print(f"Matched token: {token}")
            allowed.append(vocab[token])
    # print(allowed)
    return allowed


def get_union_allowed_ids(
    candidates: dict[str, str], vocab: dict[str, int]
) -> list[int]:
    """Return the union of allowed next-token IDs across all candidates.
    Args:
        candidates: A mapping from candidate name to its remaining
            (not-yet-matched) string, e.g. ``{"fn_greet": "greet"}``.
        vocab: The token-string-to-ID vocabulary mapping.
    Returns:
        The deduplicated list of token IDs that legally continue at
        least one candidate.
    """
    unioned_ids = set()
    for i in candidates.values():
        # print(f"i id {i}")
        for id in get_allowed_ids(i, vocab):
            unioned_ids.add(id)
    # print(f"Union Allowed ids: {unioned_ids}")
    return list(unioned_ids)


def update_candidates(
    candidates: dict[str, str],
    chosen_str: str
) -> dict[str, str]:
    """Narrow candidates to those matching the just-chosen token string.
    Candidates whose remaining string does not start with ``chosen_str``
    are dropped; the rest have ``chosen_str`` stripped from the front of
    their remaining string, ready for the next decoding step.
    Args:
        candidates: A mapping from candidate name to its remaining
            string, as produced by a previous call or the initial seed.
        chosen_str: The token string the model just selected.
    Returns:
        A new mapping containing only the still-matching candidates,
        with ``chosen_str`` consumed from their remaining string.
    """
    return {
        name: remaining[len(chosen_str):]
        for name, remaining in candidates.items()
        if remaining.startswith(chosen_str)
    }


def vocab_id_to_token(vocab: dict) -> dict[int, str]:
    """Invert a token-to-ID vocabulary into an ID-to-token mapping.
    Args:
        vocab: The token-string-to-ID vocabulary mapping.
    Returns:
        The inverse mapping, from ID to token string.
    """
    return {v: key for key, v in vocab.items()}


def build_dynamic_prompt(
    prompt: str,
    definitions: list[FunctionDefinition]
) -> str:
    """Build the few-shot prompt used to select which function to call.
    The available function list is generated dynamically from
    ``definitions``, so nothing about a specific function set is
    hardcoded: swapping in an entirely different set of functions
    requires no code change.
    Args:
        prompt: The user's natural-language request.
        definitions: The available function definitions to advertise.
    Returns:
        The full prompt text, listing every function name and
        description followed by the user's request.
    """
    lines = [f"- {d.name}: {d.description}" for d in definitions]
    return (
        "You translate user requests into function calls.\n"
        "Available functions:\n" + "\n".join(lines) + "\n"
        f"User request: {prompt}\n"
    )


def get_parameter_type_list(
    chosen_func: str, funcs: list[FunctionDefinition]
) -> list[str]:
    """List the parameter types (in order) of one chosen function.
    Args:
        chosen_func: The name of the function to look up.
        funcs: The full list of available function definitions.
    Returns:
        The type name of each parameter of ``chosen_func``, in
        declaration order (e.g. ``["number", "string"]``). Empty if no
        function named ``chosen_func`` is found.
    """
    intro_list = []
    for func in funcs:
        if func.name != chosen_func:
            continue
        for parameter in func.parameters.values():
            info = f"{parameter.type}"
            intro_list.append(info)
    return intro_list


def masked_argmax(
    model: Small_LLM_Model,
    generated_ids: list,
    allowed_ids: list[int],
    discouraged_ids: list[int],
) -> tuple[int, Any]:
    """Pick the next token ID under a hard mask plus a soft penalty.
    ハードなマスクとソフトな減点を適用した上で、次のトークンIDを選ぶ。

    This is the single function implementing constrained decoding: every
    token *not* in ``allowed_ids`` gets ``-inf`` added to its logit, so it
    can never be selected no matter how confident the model is about it
    - the schema always wins. Tokens in ``discouraged_ids`` instead get a
    finite ``-8.0`` penalty, biasing the model away from them without
    forbidding them outright (used e.g. to discourage, but not prevent,
    copying the parameter's own name as its value).
    Args:
        model: The LLM wrapper to query for next-token logits.
        generated: The token IDs generated so far (the current context).
        allowed_ids: Token IDs that are structurally legal here; every
            other token is masked to ``-inf``.
        discouraged_ids: Token IDs to softly penalize (``-8.0``) without
            excluding them, as a plain list or a numpy array of IDs.
    Returns:
        The chosen token ID: the argmax of the masked-and-penalized
        logits.。
    """
    logits_np = np.array(model.get_logits_from_input_ids(generated_ids))
    mask = np.full_like(logits_np, -np.inf)
    mask[allowed_ids] = 0.0
    mask[discouraged_ids] = -8.0
    return int(np.argmax(logits_np + mask)), logits_np


def _non_digit_ids(vocab: dict[str, int]) -> list[int]:
    """List vocabulary IDs for every token except digits, ``,`` and ``}``.
    Used as the legal token set while generating a string value: digits
    are excluded (discouraged separately via ``_digit_ids`` instead, so
    genuinely numeric-looking string content is still reachable), and the
    two structural characters are written by the caller directly rather
    than generated.
    Args:
        vocab: The token-string-to-ID vocabulary mapping.
    Returns:
        The IDs of every token that is not a digit, ``","``, or ``"}"``.
    """
    return [
        t_id for token, t_id in vocab.items()
        if token != vocab[","] and token != vocab["}"] and not token.isdigit()
        ]


def _digit_ids(vocab: dict[str, int]) -> list[int]:
    """List vocabulary IDs for digit tokens (plus ``,`` and ``.``).
    The comma and dot are included because a number's value may need a
    decimal point, and a soft-discouraged comma helps steer away from
    accidentally spelling one into a string value (see
    ``_generate_string_value``).
    Args:
        vocab: The token-string-to-ID vocabulary mapping.
    Returns:
        The IDs of every token that is a digit, ``","``, or ``"."``.
    """
    return [
        t_id for token, t_id in vocab.items()
        if token.isdigit() or token == "," or token == "."
        ]


def _generate_object_value(
    parameter_title: str,
    param_schema: Any,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int]
) -> tuple[dict[str, Any], str]:
    """Generate a nested-object value by recursing into each property.
    Args:
        parameter_title: The name of the object parameter being
        param_schema: The ``ParameterSchema`` with ``type == "object"``
            and a non-``None`` ``properties`` mapping.
        prompt: The JSON text assembled so far.
        user_prompt: The original natural-language user request.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        A tuple of the generated nested dict and the updated prompt
    Raises:
        ValueError: If ``param_schema.properties`` is ``None``.
    """
    if param_schema.properties is None:
        raise ValueError(
            f'Schema for "{parameter_title}" has type "object" '
            "but no properties defined"
        )
    prompt = prompt + '"' + parameter_title + '": {'
    nested_value: dict[str, Any] = {}
    items = list(param_schema.properties.items())
    for i, (sub_name, sub_schema) in enumerate(items):
        if i > 0:
            prompt += ", "
        sub_value, prompt = generate_value(
            sub_name,
            sub_schema,
            prompt,
            model,
            vocab,
            id_to_token,
            verbose,
            merge_ranks,
        )
        nested_value[sub_name] = sub_value
    prompt = prompt + "}"
    return nested_value, prompt


def _generate_num_value(
    parameter_title: str,
    param_schema: Any,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int]
) -> tuple[float | int, str]:
    """Generate a ``number`` or ``integer`` value, one digit at a time.
    Only digit tokens (and a decimal point, once at least one digit has
    been written) are legal, until a comma/brace token signals the end.
    Args:
        parameter_title: The name of the numeric parameter being
        param_schema: The ``ParameterSchema`` with ``type`` of
            ``"number"`` or ``"integer"``.
        prompt: The JSON text assembled so far.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        A tuple of the generated ``float`` or ``int`` value and the
        updated prompt text.
    Raises:
        RuntimeError: If generation exceeds a sanity length limit
            without terminating (indicates a runaway loop).
    """
    digit_ids = _digit_ids(vocab)
    TERMINATOR_LIST = [vocab[","], vocab["}"]]
    prompt = prompt + '"' + parameter_title + '": '
    value = ""
    if verbose:
        print("\nParam_type number")
        print(f"Param_name {parameter_title}")
        print(f"Prompt: {prompt}")
    generated_ids = encode(prompt, merge_ranks, vocab)
    while True:
        allowed_ids = digit_ids if not value \
            else digit_ids + TERMINATOR_LIST
        discouraged_ids: list[int] = []
        chosen = masked_argmax(
            model, generated_ids,
            allowed_ids, discouraged_ids
        )
        chosen_tok = id_to_token[chosen]
        if chosen in TERMINATOR_LIST:
            break
        value += chosen_tok
        generated_ids.append(chosen)
        if len(value) > 15:
            raise RuntimeError(f"Runaway number generation: {value!r}")
    if param_schema.type == "number":
        return float(value), f"{prompt}{value}"
    else:
        return int(value), f"{prompt}{value}"


def _generate_boolean_value(
    parameter_title: str,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int]
) -> tuple[bool, str]:
    """Generate a ``boolean`` value via prefix matching on "true"/"false".
    Args:
        parameter_title: The name of the boolean parameter being
        prompt: The JSON text assembled so far.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        A tuple of the generated ``bool`` value and the updated prompt
    """
    prompt = prompt + '"' + parameter_title + '": '
    if verbose:
        print("\nParam_type boolean")
        print(f"Param_name {parameter_title}")
        print(f"Prompt: {prompt}")
    generated_ids = encode(prompt, merge_ranks, vocab)
    candidates = {"true": "true", "false": "false"}
    chosen_bool: Any = None
    while chosen_bool is None:
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        chosen = masked_argmax(model, generated_ids, allowed_ids, [])
        generated_ids.append(chosen)
        chosen_str = id_to_token[chosen]
        candidates = update_candidates(candidates, chosen_str)
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_bool = name
    value = chosen_bool == "true"
    return value, f"{prompt}{chosen_bool}"


def _generate_str_value(
    parameter_title: str,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int]
) -> tuple[str, str]:
    """Generate a ``string`` value, stopping at the closing quote.
    Any non-digit token is legal at each step (digits and the
    parameter's own name are softly discouraged, to bias away from
    copying the schema itself rather than real content) until a token
    containing a literal ``"`` is produced, signalling the end of the
    value.
    Args:
        parameter_title: The name of the string parameter being
            generated (also used to discourage self-referential copies).
        prompt: The JSON text assembled so far.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        A tuple of the generated string value and the updated prompt
        text (with the closing quote appended).
    Raises:
        RuntimeError: If generation exceeds a sanity length limit
            without terminating (indicates a runaway loop).
    """
    byte_encoder = bytes_to_unicode()
    byte_decoder = build_byte_decoder(byte_encoder)
    all_ids = _non_digit_ids(vocab)
    digit_ids = _digit_ids(vocab)
    prompt = prompt + '"' + parameter_title + '": "'
    # value = ""
    value_ids: list[int] = []
    if verbose:
        print("Param_type string")
        print(f"Param_name {parameter_title}")
        print(f"Prompt: {prompt}")
    generated_ids = encode(prompt, merge_ranks, vocab)
    while True:
        allowed_ids = all_ids if not value_ids \
            else all_ids + [vocab[","], vocab["}"]]
        parameter_id = get_allowed_ids(parameter_title, vocab)
        discouraged_ids = digit_ids + parameter_id
        chosen = masked_argmax(
            model, generated_ids, allowed_ids,
            discouraged_ids
        )
        chosen_tok = id_to_token[chosen]
        vprint(f"Chosen token:=={chosen_tok}==", verbose)
        if '"' in chosen_tok:
            prefix_sym = chosen_tok.split('"')[0]
            if prefix_sym and prefix_sym in vocab:
                value_ids.append(vocab[prefix_sym])
            break
        if chosen in [vocab[","], vocab["}"]]:
            break
        value_ids.append(chosen)
        generated_ids.append(chosen)
        if len(value_ids) > 60:
            raise RuntimeError("Runway string generation")
    if value_ids:
        value = decode_tokens(value_ids, id_to_token, byte_decoder)
    else:
        value = ""
    value = value.lstrip(" ").rstrip(",")
    vprint(f"Completed Value:=={value}==", verbose)
    return value, prompt + value + '"'


def generate_value(
    parameter_title: str,
    param_schema: Any,
    prompt: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> tuple[Any, str]:
    """Generate one parameter's value, dispatching on its schema type.

    Writes the fixed JSON punctuation (key name, colon, braces/quotes)
    directly into ``prompt`` without consulting the model - only the
    actual value content is generated token by token, using constrained
    decoding appropriate to the type. See ``_generate_object_value``,
    ``_generate_number_value``, ``_generate_boolean_value``, and
    ``_generate_string_value`` for the type-specific details.
    Args:
        parameter_title: The name of the parameter being generated.
        param_schema: The ``ParameterSchema`` describing this。
        prompt: The JSON text assembled so far; this call appends to it.
        user_prompt: The original natural-language user request, passed
            through to ``_generate_object_value`` for its recursive
            calls.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.

    Returns:
        A tuple of the generated Python value (``dict``, ``float``,
        ``int``, ``bool``, or ``str`` depending on the schema type) and
        the updated ``prompt`` text with this value's JSON appended.
        生成されたPythonの値（スキーマの型に応じて``dict``、``float``、
        ``int``、``bool``、``str``のいずれか）と、この値のJSONが
        追記された``prompt``のタプル。

    Raises:
        ValueError: If an ``object`` schema has no ``properties``, or
            the schema's ``type`` is not one of the supported kinds.
            ``object``スキーマに``properties``がない場合、または
            スキーマの``type``がサポート対象の種類でない場合。
        RuntimeError: If number or string generation runs away past its
            sanity length limit without terminating.
            数値または文字列の生成が、正常に終端しないまま長さの
            上限を超えて暴走した場合。
    """
    if param_schema.type == "object":
        return _generate_object_value(
            parameter_title, param_schema, prompt, model, vocab,
            id_to_token, verbose, merge_ranks
        )

    elif param_schema.type == "number" or param_schema.type == "integer":
        return _generate_num_value(
            parameter_title, param_schema, prompt, model, vocab,
            id_to_token, verbose, merge_ranks
        )

    elif param_schema.type == "boolean":
        return _generate_boolean_value(
            parameter_title, prompt, model, vocab,
            id_to_token, verbose, merge_ranks
        )

    elif param_schema.type == "string":
        return _generate_str_value(
            parameter_title, prompt, model, vocab,
            id_to_token, verbose, merge_ranks
        )
    else:
        raise ValueError(f"Unsupported parameter type: {param_schema.type}")


def generate_parameter(
    param_fetch_dict: ParameterFetch,
    user_prompt: str,
    chosen_func: str,
    parameters: dict,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> str:
    """Generate every argument of the chosen function, in order.
    Writes each generated value into ``param_fetch_dict.parameters`` as
    a side effect, and also returns the assembled JSON text (mainly
    useful for tracing/debugging).
    Args:
        param_fetch_dict: The accumulator to fill with generated
        user_prompt: The original natural-language user request.
        chosen_func: The name of the function whose parameters are
        parameters: The chosen function's parameter schemas, keyed by
            name (from ``FunctionDefinition.parameters``).
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step generation traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        The full ``"parameters": {...}`` JSON text assembled while
        generating each argument.
    """
    prompt = (
        '"prompt": '
        + user_prompt
        + "\n"
        + '"function": '
        + chosen_func
        + "\n"
        + '"parameters": {'
    )
    if bool(re.search(r"\b[A-Z]+\b", user_prompt)):
        prompt = (
            "When only a value is capital letter and plural, "
            + "spell it out completely, including the final 's'.\n"
            + 'Example: \'with CATS\' -> replacement: "CATS" (not "CAT")\n'
            + prompt
        )
    items = list(parameters.items())
    for i, (title, schema) in enumerate(items):
        if i > 0:
            prompt += ", "
        value, prompt = generate_value(
            title, schema, prompt, model,
            vocab, id_to_token, verbose, merge_ranks
        )
        param_fetch_dict.parameters[title] = ParameterValue(value)
    prompt = prompt + "}"
    return prompt


def generate_function_call(
    user_prompt: str,
    funcs: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    verbose: argparse.Namespace,
    merge_ranks: dict[tuple[str, str], int],
) -> str:
    """Select which function best matches the user's request.
    Builds the dynamic function-list prompt, then walks the model down a
    prefix-matched choice between every known function name (plus
    "fn_none" for "no function fits") using the same constrained-decoding
    machinery as everywhere else - the model can never land on a name
    that is not in ``funcs``.
    Args:
        user_prompt: The user's natural-language request.
        funcs: The available function definitions to choose among.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        verbose: Truthy to print step-by-step selection traces.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
    Returns:
        The name of the chosen function, or ``"fn_none"`` if the model
        determines no function fits.
    Raises:
        RuntimeError: If every candidate gets eliminated before one is
            fully spelled out (indicates a logic bug or corrupt input).
    """
    full_prompt = build_dynamic_prompt(user_prompt, funcs)
    vprint("let AI model chose a FUNCTION by feeding dynamic prompt", verbose)
    vprint(f"Dynamic prompt: {full_prompt}", verbose)
    generated_ids = encode(full_prompt, merge_ranks, vocab)
    # prefix = '{"name": "'
    prefix_ids = encode(full_prompt, merge_ranks, vocab)
    generated_ids.extend(prefix_ids)
    candidates = {f.name: f.name for f in funcs}
    candidates["fn_none"] = "fn_none"
    chosen_function = None
    while chosen_function is None:
        if not candidates:
            raise RuntimeError(
                "All candidates eliminated - " + "logic bug or invalid input"
            )
        allowed_ids = get_union_allowed_ids(candidates, vocab)
        discouraged_ids: list[int] = []
        chosen = masked_argmax(
            model, generated_ids,
            allowed_ids, discouraged_ids
        )

        generated_ids.append(chosen)
        vprint(f"\n{generated_ids}", verbose)
        chosen_str = id_to_token[chosen]
        vprint(f"Model picked Logit ID: {chosen} \n"
               f"Logit Token: {chosen_str}", verbose)
        candidates = update_candidates(candidates, chosen_str)
        vprint(
            f"ModelChose: {chosen_str!r}, Remaining func name{candidates}",
            verbose
        )
        for name, remaining in candidates.items():
            if remaining == "":
                chosen_function = name
    vprint(f"\nChosen function: {chosen_function}", verbose)
    return chosen_function


def verify_function_choice(
    user_prompt: str,
    chosen_func: str,
    model: Small_LLM_Model,
    vocab: dict[str, int],
    id_to_token: dict[int, str],
    merge_ranks: dict[tuple[str, str], int],
    verbose: argparse.Namespace,
) -> bool:
    """Ask the model to double-check that the chosen function actually fits.
    Acts as a second opinion after ``generate_function_call``: the model
    is asked a plain yes/no question, decoded via the same "yes"/"no"
    prefix-matching used elsewhere, so it can only ever answer one of
    those two words.
    Args:
        user_prompt: The user's natural-language request.
        chosen_func: The function name selected by
            ``generate_function_call``, to be verified.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        id_to_token: The inverse of ``vocab``, from ID to token string.
        merge_ranks: The BPE merge-priority table from ``load_merges``.
        verbose: Truthy to print the verification result.
    Returns:
        ``True`` if the model answers "yes" (the choice is appropriate),
        ``False`` if it answers "no".
    """
    verify_prompt = (
        f'User request: "{user_prompt}"\n'
        f"Selected function: {chosen_func}\n"
        f"Is this selected function appropriate for the User request?"
    )
    generated_ids = encode(verify_prompt, merge_ranks, vocab)
    candidates = [vocab["yes"], vocab["no"]]
    _, np_logits = masked_argmax(model, generated_ids, candidates, [])
    yes_logit = np_logits[candidates[0]]
    no_logit = np_logits[candidates[1]]

    return not no_logit - yes_logit >= 2.3


def engine(
    parser: Parser, model: Small_LLM_Model,
    vocab: Any, verbose: argparse.Namespace
) -> list[dict]:
    """Run the full function-calling pipeline over every parsed prompt.
    For each prompt: select a function, verify the selection, generate
    its arguments (skipped for "fn_none"), and record the result. Each
    prompt is processed inside its own ``try``/``except`` so that one
    prompt's failure (e.g. a runaway-generation guard tripping) does not
    abort the rest of the batch; on failure, a schema-compliant
    ``fn_none`` fallback entry is recorded instead.
    Args:
        parser: The validated ``Parser`` holding the prompts and
            function definitions to process.
        model: The LLM wrapper used for constrained decoding.
        vocab: The token-string-to-ID vocabulary mapping.
        verbose: Truthy to print step-by-step traces for every prompt.
    Returns:
        One result dict per prompt, each with exactly the keys
        ``"prompt"``, ``"name"``, and ``"parameters"`` - ready to be
        JSON-serialized as the final output file.
    """
    prompts = parser.prompt_list
    funcs = parser.func_list
    id_to_token = vocab_id_to_token(vocab)
    answer_list: list[dict] = []

    from .bpe_tokenizer import load_merges

    merges_path = model.get_path_to_merges_file()
    merge_ranks = load_merges(merges_path)

    for user_prompt in prompts:
        try:  # for Bonus "Advanced error recovery try/except"
            vprint("\n===============New Request================", verbose)
            vprint(f"User prompt: {user_prompt}", verbose)
            param_fetch_dict = ParameterFetch()
            param_fetch_dict.prompt = user_prompt.prompt
            vprint("\n========Select Function=======", verbose)
            chosen_func = generate_function_call(
                user_prompt.prompt,
                funcs,
                model,
                vocab,
                id_to_token,
                verbose,
                merge_ranks,
            )

            if not verify_function_choice(
                user_prompt.prompt,
                chosen_func,
                model,
                vocab,
                id_to_token,
                merge_ranks,
                verbose,
            ):
                chosen_func = "fn_none"

            param_fetch_dict.name = chosen_func

            if chosen_func == "fn_none":
                vprint("\n=======No matching function=======", verbose)
            else:
                func = next(d for d in funcs if d.name == chosen_func)
                vprint("\n=======Generate Parameter=======", verbose)
                generate_parameter(
                    param_fetch_dict,
                    user_prompt.prompt,
                    chosen_func,
                    func.parameters,
                    model,
                    vocab,
                    id_to_token,
                    verbose,
                    merge_ranks,
                )
            answer_list.append(param_fetch_dict.model_dump())
            vprint("\n****Complete generation for the prompt****", verbose)
            vprint(param_fetch_dict.model_dump_json(indent=2), verbose)
        except Exception as e:
            vprint(f"Error for a prompt {user_prompt.prompt} - {e}", verbose)
            answer_list.append({"prompt": user_prompt.prompt,
                                "name":  "fn_none",
                                "parameters": {},
                                "error_message": str(e)})
    return answer_list
