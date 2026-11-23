from __future__ import annotations

import re
from dataclasses import dataclass

from agent_preflight.models import Finding, Severity
from agent_preflight.rules import (
    CONTEXT_CLIFF_TOKEN_THRESHOLD,
    GPT62_UNSUPPORTED_PARAMS,
    RETIRED_API_ENDPOINTS,
    is_gpt62_model,
    is_gpt6_family,
    is_no_minimal_reasoning_model,
    is_no_none_reasoning_model,
    is_responses_only_tool_model,
    is_rule_table_identifier,
    make_finding,
    match_retired_model,
    parse_suppression_directive,
)

_MODEL_KEYS = frozenset({"model", "modelId", "model_id", "modelName", "model_name", "defaultModel"})
_HISTORY_NAME_RE = re.compile(
    r"(?:^|_)(?:messages|history|conversation|chatHistory|chat_history|inputItems|input_items|turns|threadItems)(?:_|$)",
    re.IGNORECASE,
)
_SYSTEM_VAR_RE = re.compile(
    r"(?:^|_)(?:systemPrompt|system_prompt|developerPrompt|developer_prompt|systemInstructions|basePrompt|sysPrompt)(?:_|$)",
    re.IGNORECASE,
)
_OUTER_TIMEOUT_RE = re.compile(
    r"(?:outer|caller|request|parent|overall|workflow|agent|global|total)Timeout(?:Ms|Sec|Seconds)?|"
    r"(?:outer|caller|request|parent|overall|workflow|agent|global|total)_timeout(?:_ms|_s)?|"
    r"^(?:deadlineMs|deadlineSeconds|timeoutBudget|timeout_budget)$",
    re.IGNORECASE,
)
_INNER_TIMEOUT_RE = re.compile(
    r"(?:inner|child|tool|subprocess|downstream|step|http|client|call)Timeout(?:Ms|Sec|Seconds)?|"
    r"(?:inner|child|tool|subprocess|downstream|step|http|client|call)_timeout(?:_ms|_s)?",
    re.IGNORECASE,
)
_CONTEXT_KEYS = frozenset({
    "max_input_tokens", "maxInputTokens", "max_context_tokens", "maxContextTokens",
    "context_window", "contextWindow", "context_limit", "contextLimit", "token_limit",
    "tokenLimit", "token_budget", "tokenBudget", "max_prompt_tokens", "maxPromptTokens",
    "max_tokens", "maxTokens",
})
_CONTEXT_VAR_RE = re.compile(
    r"(?:context|token|prompt|window|chunk).*(?:limit|budget|max|size|cap|window|tokens|length)|"
    r"(?:max|limit|budget).*(?:input_?tokens|context_?tokens|prompt_?tokens|tokens)",
    re.IGNORECASE,
)
_NON_CLIFF_VAR_RE = re.compile(r"(?:cliff|threshold|timeout|byte|bytes|char|chars|port|year|ms|delay)", re.IGNORECASE)
_DYNAMIC_JS_RE = re.compile(r"\b(?:Date\.now|new\s+Date|crypto\.randomUUID|Math\.random|performance\.now|uuidv4|uuid)\b")
_NOT_IMPLEMENTED_RE = re.compile(r"\b(?:not\s+implemented|todo\s*:?\s*implement)\b", re.IGNORECASE)
_REGEX_PREV_IDENTS = frozenset({"return", "typeof", "throw", "await", "yield", "case", "in", "of", "else"})
_REGEX_PREV_PUNCTS = frozenset({"=", "(", "[", "{", ",", ";", ":", "!", "&&", "||", "??", "?", "+", "-", "*", "/", "=>", "==", "===", "!=", "!=="})


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: str
    value: str
    line: int
    col: int


def _tokenize_js(source: str) -> tuple[list[_Tok], dict[int, set[str] | None], set[int]]:
    tokens: list[_Tok] = []
    suppressions: dict[int, set[str] | None] = {}
    comment_lines: set[int] = set()
    code_lines: set[int] = set()
    n = len(source)
    i = 0
    line = 1
    col = 1

    def record_supp(ln: int, text: str) -> None:
        comment_lines.add(ln)
        has_dir, rule_ids = parse_suppression_directive(text)
        if has_dir:
            if ln in suppressions:
                existing = suppressions[ln]
                if existing is None or rule_ids is None:
                    suppressions[ln] = None
                else:
                    existing.update(rule_ids)
            else:
                suppressions[ln] = rule_ids

    while i < n:
        ch = source[i]
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch in (" ", "\t", "\r"):
            i += 1
            col += 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            start_ln = line
            j = i + 2
            while j < n and source[j] != "\n":
                j += 1
            record_supp(start_ln, source[i + 2 : j])
            col += j - i
            i = j
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            j = i + 2
            cur_ln = line
            buf: list[str] = []
            while j < n:
                if source[j] == "*" and j + 1 < n and source[j + 1] == "/":
                    record_supp(cur_ln, "".join(buf))
                    j += 2
                    col += 2
                    break
                if source[j] == "\n":
                    record_supp(cur_ln, "".join(buf))
                    buf.clear()
                    cur_ln += 1
                    line += 1
                    col = 1
                else:
                    buf.append(source[j])
                    col += 1
                j += 1
            else:
                record_supp(cur_ln, "".join(buf))
            i = j
            continue
        if ch in ('"', "'"):
            quote, start_ln, start_col = ch, line, col
            j = i + 1
            col += 1
            chars: list[str] = []
            while j < n:
                c = source[j]
                if c == "\\":
                    if j + 1 < n:
                        nxt = source[j + 1]
                        if nxt == "\n":
                            line += 1
                            col = 1
                        else:
                            chars.append(nxt)
                            col += 2
                        j += 2
                        continue
                if c == quote:
                    j += 1
                    col += 1
                    break
                if c == "\n":
                    line += 1
                    col = 1
                    j += 1
                    break
                chars.append(c)
                j += 1
                col += 1
            tokens.append(_Tok("STRING", "".join(chars), start_ln, start_col))
            code_lines.add(start_ln)
            i = j
            continue
        if ch == "`":
            start_ln, start_col = line, col
            j = i + 1
            col += 1
            chars = []
            expr_depth = 0
            while j < n:
                c = source[j]
                if c == "\\" and j + 1 < n:
                    chars.extend((c, source[j + 1]))
                    if source[j + 1] == "\n":
                        line += 1
                        col = 1
                    else:
                        col += 2
                    j += 2
                    continue
                if c == "$" and j + 1 < n and source[j + 1] == "{":
                    expr_depth += 1
                    chars.append("${")
                    j += 2
                    col += 2
                    continue
                if expr_depth > 0:
                    if c == "{":
                        expr_depth += 1
                    elif c == "}":
                        expr_depth -= 1
                    chars.append(c)
                    if c == "\n":
                        line += 1
                        col = 1
                    else:
                        col += 1
                    j += 1
                    continue
                if c == "`":
                    j += 1
                    col += 1
                    break
                if c == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                chars.append(c)
                j += 1
            tokens.append(_Tok("TEMPLATE", "".join(chars), start_ln, start_col))
            code_lines.add(start_ln)
            i = j
            continue
        if ch == "/":
            prev = tokens[-1] if tokens else None
            if prev is None or (prev.kind == "IDENT" and prev.value in _REGEX_PREV_IDENTS) or (prev.kind == "PUNCT" and prev.value in _REGEX_PREV_PUNCTS):
                start_ln, start_col = line, col
                j = i + 1
                col += 1
                in_class = False
                while j < n:
                    c = source[j]
                    if c == "\\":
                        j += 2
                        col += 2
                        continue
                    if c == "[":
                        in_class = True
                    elif c == "]" and in_class:
                        in_class = False
                    elif c == "/" and not in_class:
                        j += 1
                        col += 1
                        while j < n and source[j].isalpha():
                            j += 1
                            col += 1
                        break
                    elif c == "\n":
                        break
                    j += 1
                    col += 1
                tokens.append(_Tok("REGEX", source[i:j], start_ln, start_col))
                code_lines.add(start_ln)
                i = j
                continue
        if ch.isdigit():
            start_ln, start_col = line, col
            j = i
            while j < n and (source[j].isalnum() or source[j] in ("_", ".")):
                if source[j] == "." and j + 1 < n and not source[j + 1].isdigit():
                    break
                j += 1
            tokens.append(_Tok("NUMBER", source[i:j], start_ln, start_col))
            code_lines.add(start_ln)
            col += j - i
            i = j
            continue
        if ch.isalpha() or ch in ("_", "$"):
            start_ln, start_col = line, col
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in ("_", "$")):
                j += 1
            tokens.append(_Tok("IDENT", source[i:j], start_ln, start_col))
            code_lines.add(start_ln)
            col += j - i
            i = j
            continue
        matched_punct = next((op for op in ("===", "!==", "...", "==", "!=", "<=", ">=", "=>", "&&", "||", "??", "?.") if source.startswith(op, i)), None)
        if matched_punct is not None:
            tokens.append(_Tok("PUNCT", matched_punct, line, col))
            code_lines.add(line)
            i += len(matched_punct)
            col += len(matched_punct)
            continue
        tokens.append(_Tok("PUNCT", ch, line, col))
        code_lines.add(line)
        i += 1
        col += 1

    return tokens, suppressions, comment_lines - code_lines


def _parse_number(raw: str) -> float | None:
    cleaned = raw.replace("_", "")
    try:
        return float(int(cleaned, 16)) if cleaned.startswith(("0x", "0X")) else float(cleaned)
    except ValueError:
        return None


def _matching_close(tokens: list[_Tok], open_idx: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for idx in range(open_idx, len(tokens)):
        if tokens[idx].kind == "PUNCT":
            if tokens[idx].value == open_ch:
                depth += 1
            elif tokens[idx].value == close_ch:
                depth -= 1
                if depth == 0:
                    return idx
    return len(tokens) - 1


def _split_top_level(tokens: list[_Tok], sep: str = ",") -> list[list[_Tok]]:
    chunks: list[list[_Tok]] = []
    cur: list[_Tok] = []
    bd = pd = kd = 0
    for tok in tokens:
        if tok.kind == "PUNCT":
            if tok.value == "{":
                bd += 1
            elif tok.value == "}":
                bd = max(0, bd - 1)
            elif tok.value == "(":
                pd += 1
            elif tok.value == ")":
                pd = max(0, pd - 1)
            elif tok.value == "[":
                kd += 1
            elif tok.value == "]":
                kd = max(0, kd - 1)
            elif tok.value == sep and bd == 0 and pd == 0 and kd == 0:
                chunks.append(cur)
                cur = []
                continue
        cur.append(tok)
    if cur:
        chunks.append(cur)
    return chunks


def _extract_object_props(obj_tokens: list[_Tok]) -> dict[str, tuple[_Tok, list[_Tok]]]:
    props: dict[str, tuple[_Tok, list[_Tok]]] = {}
    for prop_toks in _split_top_level(obj_tokens, ","):
        if not prop_toks:
            continue
        colon_idx = None
        bd = pd = kd = 0
        for idx, t in enumerate(prop_toks):
            if t.kind == "PUNCT":
                if t.value == "{":
                    bd += 1
                elif t.value == "}":
                    bd = max(0, bd - 1)
                elif t.value == "(":
                    pd += 1
                elif t.value == ")":
                    pd = max(0, pd - 1)
                elif t.value == "[":
                    kd += 1
                elif t.value == "]":
                    kd = max(0, kd - 1)
                elif t.value == ":" and bd == 0 and pd == 0 and kd == 0:
                    colon_idx = idx
                    break
        if colon_idx is not None and colon_idx >= 1:
            k_tok = prop_toks[colon_idx - 1]
            if k_tok.kind in ("IDENT", "STRING"):
                props[k_tok.value] = (k_tok, prop_toks[colon_idx + 1 :])
        elif len(prop_toks) == 1 and prop_toks[0].kind == "IDENT":
            props[prop_toks[0].value] = (prop_toks[0], [prop_toks[0]])
    return props


class _JsTsAuditor:
    def __init__(
        self,
        source: str,
        file_path: str,
        tokens: list[_Tok],
        suppressions: dict[int, set[str] | None],
        standalone_comment_lines: set[int],
    ) -> None:
        self.lines = source.splitlines()
        self.file_path = file_path
        self.tokens = tokens
        self.suppressions = suppressions
        self.standalone_comment_lines = standalone_comment_lines
        self.findings: list[Finding] = []
        self.suppressed_count: int = 0
        self._seen: set[tuple[str, int, int]] = set()
        self.str_vars: dict[str, str] = {}
        self.num_vars: dict[str, float] = {}
        self.dynamic_vars: set[str] = set()
        self.history_vars: set[str] = set()
        self.outer_timeouts: list[tuple[float, str, int, bool]] = []
        self.responses_call_end_idx: int = -1
        self.chat_call_end_idx: int = -1

    def _snippet(self, line: int) -> str:
        return self.lines[line - 1] if 1 <= line <= len(self.lines) else ""

    def _is_suppressed(self, rule_id: str, lines_to_check: tuple[int, ...]) -> bool:
        candidates: list[int] = []
        for ln in lines_to_check:
            candidates.append(ln)
            if ln > 1 and (ln - 1) in self.standalone_comment_lines:
                candidates.append(ln - 1)
        for ln in candidates:
            if ln in self.suppressions:
                allowed = self.suppressions[ln]
                if allowed is None or rule_id in allowed:
                    return True
        return False

    def _emit(
        self,
        rule_id: str,
        message: str,
        tok: _Tok,
        severity_override: Severity | None = None,
        recommendation_override: str | None = None,
        extra_suppress_lines: tuple[int, ...] = (),
    ) -> None:
        key = (rule_id, tok.line, tok.col)
        if key in self._seen:
            return
        self._seen.add(key)
        if self._is_suppressed(rule_id, (tok.line,) + extra_suppress_lines):
            self.suppressed_count += 1
            return
        self.findings.append(
            make_finding(rule_id, message, self.file_path, tok.line, tok.col, snippet=self._snippet(tok.line), severity_override=severity_override, recommendation_override=recommendation_override)
        )

    def _resolve_str_tokens(self, expr: list[_Tok]) -> str | None:
        if len(expr) == 1:
            t = expr[0]
            if t.kind == "STRING":
                return t.value
            if t.kind == "IDENT" and t.value in self.str_vars:
                return self.str_vars[t.value]
        return None

    def _resolve_num_tokens(self, expr: list[_Tok]) -> float | None:
        if len(expr) == 1:
            t = expr[0]
            if t.kind == "NUMBER":
                return _parse_number(t.value)
            if t.kind == "IDENT" and t.value in self.num_vars:
                return self.num_vars[t.value]
        return None

    def _tokens_have_dynamic(self, expr: list[_Tok]) -> str | None:
        for idx, t in enumerate(expr):
            if t.kind == "TEMPLATE" and "${" in t.value:
                if (m := _DYNAMIC_JS_RE.search(t.value)):
                    return m.group(0)
                for dv in self.dynamic_vars:
                    if re.search(rf"\b{re.escape(dv)}\b", t.value):
                        return dv
            if t.kind == "IDENT" and t.value in self.dynamic_vars:
                return t.value
            if t.kind == "IDENT":
                if t.value == "Date" and idx + 2 < len(expr) and expr[idx + 1].value == "." and expr[idx + 2].value == "now":
                    return "Date.now()"
                if t.value == "new" and idx + 1 < len(expr) and expr[idx + 1].value == "Date":
                    return "new Date()"
                if t.value == "crypto" and idx + 2 < len(expr) and expr[idx + 1].value == "." and expr[idx + 2].value == "randomUUID":
                    return "crypto.randomUUID()"
                if t.value == "Math" and idx + 2 < len(expr) and expr[idx + 1].value == "." and expr[idx + 2].value == "random":
                    return "Math.random()"
                if t.value in ("uuidv4", "uuid"):
                    return t.value + "()"
        return None

    def _is_non_empty_expr(self, expr: list[_Tok]) -> bool:
        if not expr:
            return False
        if len(expr) == 1 and expr[0].kind == "IDENT" and expr[0].value in ("undefined", "null"):
            return False
        return not (len(expr) == 2 and expr[0].value == "[" and expr[1].value == "]")

    def _check_model(self, model_str: str, tok: _Tok, extra_lines: tuple[int, ...] = ()) -> None:
        retirement = match_retired_model(model_str)
        if retirement is not None:
            status_label = "already shut down" if retirement.status == "shutdown" else f"retiring on {retirement.shutdown_date}"
            self._emit("APF001", f"Model {model_str!r} is {status_label}; migrate to {retirement.replacement}.", tok, severity_override=retirement.severity, recommendation_override=f"Replace {model_str!r} with {retirement.replacement}.", extra_suppress_lines=extra_lines)

    def _check_array_messages(self, arr_toks: list[_Tok], call_line: int) -> None:
        if len(arr_toks) < 2 or arr_toks[0].value != "[" or arr_toks[-1].value != "]":
            return
        seen_dynamic = False
        for item in _split_top_level(arr_toks[1:-1], ","):
            if len(item) >= 2 and item[0].value == "{" and item[-1].value == "}":
                props = _extract_object_props(item[1:-1])
                role = self._resolve_str_tokens(props["role"][1]) if "role" in props else None
                dyn = self._tokens_have_dynamic(props["content"][1]) if "content" in props else None
                if role in ("system", "developer"):
                    if dyn:
                        self._emit("APF006", f"Dynamic value {dyn!r} inside {role!r} prompt prefix breaks prompt caching.", props["content"][0], extra_suppress_lines=(call_line,))
                    elif seen_dynamic:
                        self._emit("APF006", f"Static {role!r} prompt placed after dynamic prefix breaks prompt caching.", item[0], extra_suppress_lines=(call_line,))
                elif dyn:
                    seen_dynamic = True
            elif self._tokens_have_dynamic(item):
                seen_dynamic = True

    def _check_object_payload(self, obj_toks: list[_Tok], owner_tok: _Tok, is_responses_api: bool, is_chat_api: bool) -> None:
        props = _extract_object_props(obj_toks)
        if not props:
            return
        call_line = owner_tok.line

        role = self._resolve_str_tokens(props["role"][1]) if "role" in props else None
        if role in ("system", "developer") and "content" in props:
            if (dyn := self._tokens_have_dynamic(props["content"][1])):
                self._emit("APF006", f"Dynamic value {dyn!r} inside {role!r} prompt prefix breaks prompt caching.", props["content"][0], extra_suppress_lines=(call_line,))

        if self.outer_timeouts and ("timeout" in props or "timeoutMs" in props):
            t_key = "timeoutMs" if "timeoutMs" in props else "timeout"
            t_tok, t_expr = props[t_key]
            if (nv := self._resolve_num_tokens(t_expr)) is not None:
                outer = min(self.outer_timeouts, key=lambda x: x[0])
                val_s = nv / 1000.0 if (t_key.endswith("Ms") or (outer[3] and nv >= 100)) else nv
                if val_s > outer[0]:
                    self._emit("APF101", f"Child timeout {t_key}={val_s:g}s exceeds caller timeout {outer[1]}={outer[0]:g}s.", t_tok, extra_suppress_lines=(call_line,))

        outer_pair: tuple[float, str] | None = None
        inner_pairs: list[tuple[float, str, _Tok]] = []
        for k_name, (k_tok, v_toks) in props.items():
            if (nv := self._resolve_num_tokens(v_toks)) is None:
                continue
            val_s = nv / 1000.0 if k_name.lower().endswith("ms") else nv
            if _OUTER_TIMEOUT_RE.match(k_name):
                if outer_pair is None or val_s < outer_pair[0]:
                    outer_pair = (val_s, k_name)
            elif _INNER_TIMEOUT_RE.match(k_name):
                inner_pairs.append((val_s, k_name, k_tok))
        if outer_pair is not None:
            for inner_s, inner_name, k_tok in inner_pairs:
                if inner_s > outer_pair[0]:
                    self._emit("APF101", f"Child timeout {inner_name}={inner_s:g}s exceeds caller timeout {outer_pair[1]}={outer_pair[0]:g}s.", k_tok, extra_suppress_lines=(call_line,))

        for ctx_key in _CONTEXT_KEYS:
            if ctx_key in props and (nv := self._resolve_num_tokens(props[ctx_key][1])) is not None and nv > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                self._emit("APF007", f"{ctx_key}={int(nv):,} exceeds the 272,000-token GPT-6.2 pricing cliff.", props[ctx_key][0], extra_suppress_lines=(call_line,))

        model_str: str | None = None
        model_tok: _Tok | None = None
        for mk in _MODEL_KEYS:
            if mk in props:
                model_tok = props[mk][1][0] if props[mk][1] else props[mk][0]
                model_str = self._resolve_str_tokens(props[mk][1])
                break
        if model_str is not None and model_tok is not None:
            self._check_model(model_str, model_tok, extra_lines=(call_line,))

        if not (model_str or is_responses_api or is_chat_api):
            return

        if is_chat_api and model_str and is_responses_only_tool_model(model_str):
            for tk in ("tools", "functions"):
                if tk in props and self._is_non_empty_expr(props[tk][1]):
                    self._emit("APF002", f"chat.completions cannot pass {tk!r} with Responses-only model {model_str!r}.", props[tk][0], extra_suppress_lines=(call_line,))

        effort_val: str | None = None
        effort_tok: _Tok | None = None
        if "reasoning_effort" in props:
            effort_tok = props["reasoning_effort"][0]
            effort_val = self._resolve_str_tokens(props["reasoning_effort"][1])
        elif "reasoning" in props:
            r_toks = props["reasoning"][1]
            if len(r_toks) >= 2 and r_toks[0].value == "{" and r_toks[-1].value == "}":
                r_props = _extract_object_props(r_toks[1:-1])
                if "effort" in r_props:
                    effort_tok = r_props["effort"][0]
                    effort_val = self._resolve_str_tokens(r_props["effort"][1])
        if effort_val is not None and effort_tok is not None and model_str:
            low_eff = effort_val.lower()
            if low_eff == "none" and (is_no_none_reasoning_model(model_str) or is_gpt6_family(model_str)):
                self._emit("APF003", f"Model {model_str!r} rejects reasoning effort 'none' (HTTP 400).", effort_tok, extra_suppress_lines=(call_line,))
            elif low_eff == "minimal" and is_no_minimal_reasoning_model(model_str):
                self._emit("APF003", f"Model {model_str!r} rejects reasoning effort 'minimal' (HTTP 400); use 'low'.", effort_tok, extra_suppress_lines=(call_line,))

        if model_str and is_gpt62_model(model_str):
            for bad_p in sorted(GPT62_UNSUPPORTED_PARAMS):
                if bad_p in props and self._is_non_empty_expr(props[bad_p][1]):
                    self._emit("APF003", f"Parameter {bad_p!r} is unsupported on GPT-6.2 model {model_str!r}.", props[bad_p][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))
            if "service_tier" in props and self._resolve_str_tokens(props["service_tier"][1]) == "flex":
                self._emit("APF003", f"service_tier='flex' is unsupported on GPT-6.2 model {model_str!r}.", props["service_tier"][0], extra_suppress_lines=(call_line,))
            if "prompt_cache_retention" in props and self._is_non_empty_expr(props["prompt_cache_retention"][1]):
                self._emit("APF006", f"Replace legacy 'prompt_cache_retention' with prompt_cache_options: {{ ttl: '30m' }} on {model_str!r}.", props["prompt_cache_retention"][0], extra_suppress_lines=(call_line,))

        if is_responses_api:
            if "response_format" in props and self._is_non_empty_expr(props["response_format"][1]):
                self._emit("APF004", "Responses API rejects 'response_format'; use text: { format: ... }.", props["response_format"][0], extra_suppress_lines=(call_line,))
            for legacy_tok in ("max_tokens", "max_completion_tokens"):
                if legacy_tok in props and self._is_non_empty_expr(props[legacy_tok][1]):
                    self._emit("APF004", f"Responses API rejects {legacy_tok!r}; use 'max_output_tokens'.", props[legacy_tok][0], extra_suppress_lines=(call_line,))
            if "messages" in props and self._is_non_empty_expr(props["messages"][1]):
                self._emit("APF004", "Responses API uses 'input' instead of 'messages'.", props["messages"][0], extra_suppress_lines=(call_line,))
        elif model_str and is_gpt6_family(model_str):
            if "max_tokens" in props and self._is_non_empty_expr(props["max_tokens"][1]):
                self._emit("APF004", f"Legacy 'max_tokens' on {model_str!r}; prefer 'max_output_tokens' (Responses) or 'max_completion_tokens' (Chat).", props["max_tokens"][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))
            if is_gpt62_model(model_str) and "response_format" in props and self._is_non_empty_expr(props["response_format"][1]):
                self._emit("APF004", f"Legacy 'response_format' with {model_str!r}; migrate to Responses API text.format.", props["response_format"][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))

        if "instructions" in props and self._is_non_empty_expr(props["instructions"][1]):
            if (dyn := self._tokens_have_dynamic(props["instructions"][1])):
                self._emit("APF006", f"Dynamic expression {dyn!r} in 'instructions' breaks prompt caching.", props["instructions"][0], extra_suppress_lines=(call_line,))
            elif is_responses_api and model_str and is_gpt62_model(model_str):
                self._emit("APF006", f"GPT-6.2 ({model_str}) appends top-level 'instructions' after 'input'; place reusable guidance in the first developer item in 'input'.", props["instructions"][0], extra_suppress_lines=(call_line,))

        for lk in ("messages", "input"):
            if lk in props:
                v_toks = props[lk][1]
                if len(v_toks) == 1 and v_toks[0].kind == "IDENT":
                    self.history_vars.add(v_toks[0].value)
                elif len(v_toks) >= 2 and v_toks[0].value == "[" and v_toks[-1].value == "]":
                    self._check_array_messages(v_toks, call_line)

    def scan(self) -> tuple[list[Finding], int]:
        toks = self.tokens
        n = len(toks)
        brace_depth = 0
        abstract_depths: list[int] = []
        rule_table_depths: list[int] = []
        i = 0

        while i < n:
            tok = toks[i]
            if tok.kind == "PUNCT" and tok.value == "{":
                brace_depth += 1
                i += 1
                continue
            if tok.kind == "PUNCT" and tok.value == "}":
                while abstract_depths and abstract_depths[-1] >= brace_depth:
                    abstract_depths.pop()
                while rule_table_depths and rule_table_depths[-1] >= brace_depth:
                    rule_table_depths.pop()
                while self.outer_timeouts and self.outer_timeouts[-1][2] >= brace_depth:
                    self.outer_timeouts.pop()
                brace_depth = max(0, brace_depth - 1)
                i += 1
                continue

            if tok.kind == "IDENT" and tok.value in ("abstract", "interface"):
                if tok.value == "interface" or (i + 1 < n and toks[i + 1].value == "class"):
                    abstract_depths.append(brace_depth + 1)

            if tok.kind == "IDENT" and tok.value in ("const", "let", "var") and i + 2 < n and toks[i + 1].kind == "IDENT":
                var_tok = toks[i + 1]
                var_name = var_tok.value
                if _HISTORY_NAME_RE.search(var_name):
                    self.history_vars.add(var_name)
                eq_idx = None
                j = i + 2
                while j < n and toks[j].value != ";" and j < i + 12:
                    if toks[j].kind == "PUNCT" and toks[j].value == "=":
                        eq_idx = j
                        break
                    if toks[j].kind == "PUNCT" and toks[j].value in ("{", "("):
                        break
                    j += 1
                if eq_idx is not None:
                    rhs_start = eq_idx + 1
                    rhs_end = rhs_start
                    bd = pd = kd = 0
                    while rhs_end < n:
                        rt = toks[rhs_end]
                        if rt.kind == "PUNCT":
                            if rt.value == "{":
                                bd += 1
                            elif rt.value == "}":
                                if bd == 0:
                                    break
                                bd -= 1
                            elif rt.value == "(":
                                pd += 1
                            elif rt.value == ")":
                                if pd == 0:
                                    break
                                pd -= 1
                            elif rt.value == "[":
                                kd += 1
                            elif rt.value == "]":
                                if kd == 0:
                                    break
                                kd -= 1
                            elif rt.value == ";" and bd == 0 and pd == 0 and kd == 0:
                                break
                        elif bd == 0 and pd == 0 and kd == 0 and rhs_end > rhs_start and rt.line > toks[rhs_end - 1].line:
                            prev_t = toks[rhs_end - 1]
                            if not (prev_t.kind == "PUNCT" and prev_t.value in ("=", "+", ",", ".", ":", "&&", "||", "??", "=>")):
                                break
                        rhs_end += 1
                    rhs = toks[rhs_start:rhs_end]
                    is_rule_var = is_rule_table_identifier(var_name)
                    if is_rule_var and any(t.value == "{" for t in rhs):
                        rule_table_depths.append(brace_depth + 1)

                    if not is_rule_var and not rule_table_depths:
                        if (s_val := self._resolve_str_tokens(rhs)) is not None:
                            self.str_vars[var_name] = s_val
                            if "model" in var_name.lower():
                                self._check_model(s_val, rhs[0], extra_lines=(var_tok.line,))
                        if (n_val := self._resolve_num_tokens(rhs)) is not None:
                            self.num_vars[var_name] = n_val
                            is_ms_var = var_name.lower().endswith("ms")
                            val_s = n_val / 1000.0 if is_ms_var else n_val
                            if _OUTER_TIMEOUT_RE.match(var_name) and val_s > 0:
                                self.outer_timeouts.append((val_s, var_name, brace_depth, is_ms_var))
                            elif _INNER_TIMEOUT_RE.match(var_name) and self.outer_timeouts:
                                outer = min(self.outer_timeouts, key=lambda x: x[0])
                                if val_s > outer[0]:
                                    self._emit("APF101", f"Child timeout {var_name}={val_s:g}s exceeds outer timeout {outer[1]}={outer[0]:g}s.", rhs[0], extra_suppress_lines=(var_tok.line,))
                            if _CONTEXT_VAR_RE.search(var_name) and not _NON_CLIFF_VAR_RE.search(var_name) and n_val > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                                self._emit("APF007", f"{var_name}={int(n_val):,} exceeds the 272,000-token GPT-6.2 pricing cliff.", rhs[0], extra_suppress_lines=(var_tok.line,))
                        if (dyn := self._tokens_have_dynamic(rhs)):
                            self.dynamic_vars.add(var_name)
                            if _SYSTEM_VAR_RE.search(var_name):
                                self._emit("APF006", f"Dynamic expression {dyn!r} assigned to system prompt {var_name!r} breaks prompt caching.", rhs[0], extra_suppress_lines=(var_tok.line,))

            if tok.kind == "IDENT" and (_HISTORY_NAME_RE.search(tok.value) or tok.value in self.history_vars):
                if i + 1 < n and toks[i + 1].value == "[":
                    pos = _matching_close(toks, i + 1, "[", "]") + 1
                    while pos < n:
                        if toks[pos].value == "." and pos + 1 < n and toks[pos + 1].kind == "IDENT":
                            pos += 2
                        elif toks[pos].value == "[":
                            pos = _matching_close(toks, pos, "[", "]") + 1
                        else:
                            break
                    if pos < n and toks[pos].value == "=":
                        self._emit("APF005", f"In-place edit on conversation history {tok.value!r} drops GPT-6.2 reasoning items.", tok)
                elif i + 3 < n and toks[i + 1].value == "." and toks[i + 2].kind == "IDENT" and toks[i + 3].value == "(":
                    if toks[i + 2].value in ("splice", "shift", "pop", "unshift", "reverse", "sort"):
                        self._emit("APF005", f"Calling .{toks[i + 2].value}() on conversation history {tok.value!r} edits prior turns and drops GPT-6.2 reasoning items.", tok)
                elif i + 5 < n and toks[i + 1].value == "=" and toks[i + 2].value == tok.value and toks[i + 3].value == "." and toks[i + 4].value == "slice":
                    self._emit("APF005", f"Rewriting conversation history {tok.value!r} via .slice() mutates prior turns and drops GPT-6.2 reasoning items.", tok)

            if tok.kind == "IDENT" and tok.value == "catch":
                if i > 0 and toks[i - 1].value == "." and i + 1 < n and toks[i + 1].value == "(":
                    close_p = _matching_close(toks, i + 1, "(", ")")
                    inner_c = toks[i + 2 : close_p]
                    if len(inner_c) >= 4 and inner_c[-2].value == "{" and inner_c[-1].value == "}":
                        self._emit("APF102", "Swallowed promise rejection in .catch(() => {}); log or rethrow.", tok, extra_suppress_lines=(inner_c[-1].line,))
                else:
                    brace_idx = None
                    if i + 1 < n and toks[i + 1].value == "{":
                        brace_idx = i + 1
                    elif i + 1 < n and toks[i + 1].value == "(":
                        close_p = _matching_close(toks, i + 1, "(", ")")
                        if close_p + 1 < n and toks[close_p + 1].value == "{":
                            brace_idx = close_p + 1
                    if brace_idx is not None:
                        close_b = _matching_close(toks, brace_idx, "{", "}")
                        if close_b == brace_idx + 1:
                            self._emit("APF102", "Swallowed exception in empty catch block; log, handle, or rethrow.", tok, extra_suppress_lines=(toks[close_b].line,))

            if tok.kind == "IDENT" and tok.value == "throw" and not abstract_depths:
                j = i + 1
                while j < n and toks[j].value != ";" and toks[j].line <= tok.line + 1:
                    if toks[j].kind == "STRING" and _NOT_IMPLEMENTED_RE.search(toks[j].value):
                        self._emit("APF103", "Not-implemented stub error in non-abstract production code.", tok)
                        break
                    if toks[j].kind == "IDENT" and toks[j].value == "NotImplementedError":
                        self._emit("APF103", "NotImplementedError stub in non-abstract production code.", tok)
                        break
                    j += 1

            if tok.kind == "IDENT" and i + 1 < n and toks[i + 1].value == "(":
                chain_parts = [tok.value]
                back = i - 1
                while back >= 1 and toks[back].value == "." and toks[back - 1].kind == "IDENT":
                    chain_parts.append(toks[back - 1].value)
                    back -= 2
                chain_parts.reverse()
                dotted = ".".join(chain_parts)
                close_p = _matching_close(toks, i + 1, "(", ")")
                arg_toks = toks[i + 2 : close_p]
                args = _split_top_level(arg_toks, ",")

                if not rule_table_depths:
                    if ".beta.assistants" in dotted or ".beta.threads" in dotted:
                        self._emit("APF001", f"Retired Assistants/Threads API call ({dotted}) shut down on 2026-08-26; migrate to Responses API.", tok)
                    for arg in args:
                        for at in arg:
                            if at.kind == "STRING":
                                for ep, (dt, rep) in RETIRED_API_ENDPOINTS.items():
                                    if ep in at.value:
                                        self._emit("APF001", f"Endpoint {ep!r} retires/shut down on {dt}; migrate to {rep}.", at, extra_suppress_lines=(tok.line,))

                    if dotted == "AbortSignal.timeout" and args and (ms := self._resolve_num_tokens(args[0])) is not None and self.outer_timeouts:
                        sec = ms / 1000.0
                        outer = min(self.outer_timeouts, key=lambda x: x[0])
                        if sec > outer[0]:
                            self._emit("APF101", f"AbortSignal.timeout({int(ms)}ms = {sec:g}s) exceeds caller timeout {outer[1]}={outer[0]:g}s.", tok)

                    direct_responses = ".responses." in dotted or dotted.startswith("responses.") or any(t.kind == "STRING" and "/v1/responses" in t.value for t in arg_toks)
                    direct_chat = ".chat.completions." in dotted or dotted.startswith("chat.completions.") or any(t.kind == "STRING" and "/v1/chat/completions" in t.value for t in arg_toks)
                    if direct_responses and close_p > self.responses_call_end_idx:
                        self.responses_call_end_idx = close_p
                    if direct_chat and close_p > self.chat_call_end_idx:
                        self.chat_call_end_idx = close_p
                    is_responses = direct_responses or (i <= self.responses_call_end_idx)
                    is_chat = direct_chat or (i <= self.chat_call_end_idx)
                    for arg in args:
                        if len(arg) >= 2 and arg[0].value == "{" and arg[-1].value == "}":
                            self._check_object_payload(arg[1:-1], tok, is_responses, is_chat)

            if tok.kind == "PUNCT" and tok.value == "{" and not rule_table_depths:
                prev = toks[i - 1] if i > 0 else None
                if prev is not None and prev.value in ("=", "return", ":"):
                    close_b = _matching_close(toks, i, "{", "}")
                    if close_b > i + 1:
                        self._check_object_payload(toks[i + 1 : close_b], tok, False, False)

            i += 1

        return self.findings, self.suppressed_count


def scan_js_source(source: str, file_path: str) -> tuple[list[Finding], int]:
    """Audit TypeScript/TSX/JavaScript source text and return (findings, suppressed_count)."""
    tokens, suppressions, standalone_comments = _tokenize_js(source)
    return _JsTsAuditor(source, file_path, tokens, suppressions, standalone_comments).scan()
