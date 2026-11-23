from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass, field

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

_MODEL_KEY_NAMES = frozenset({"model", "model_id", "model_name", "default_model", "deployment", "engine"})
_HISTORY_NAME_RE = re.compile(
    r"(?:^|_)(?:messages|history|conversation|chat_history|input_items|turns|thread_items|msg_history)(?:_|$)",
    re.IGNORECASE,
)
_SYSTEM_VAR_RE = re.compile(
    r"(?:^|_)(?:system_prompt|developer_prompt|system_instructions|base_prompt|system_msg|developer_msg)(?:_|$)",
    re.IGNORECASE,
)
_OUTER_TIMEOUT_RE = re.compile(
    r"(?:^|_)(?:outer|caller|request|parent|overall|workflow|agent|global|total)_timeout(?:_ms|_s|_sec|_seconds)?$|"
    r"^deadline(?:_seconds|_ms)?$|^timeout_budget(?:_ms|_s)?$",
    re.IGNORECASE,
)
_INNER_TIMEOUT_RE = re.compile(
    r"(?:^|_)(?:inner|child|tool|subprocess|downstream|step|http|client|call)_timeout(?:_ms|_s|_sec|_seconds)?$",
    re.IGNORECASE,
)
_CONTEXT_KEY_NAMES = frozenset({
    "max_input_tokens", "max_context_tokens", "context_window", "context_limit",
    "token_limit", "token_budget", "max_prompt_tokens", "input_token_limit",
    "prompt_token_limit", "chunk_tokens", "max_context_length", "max_tokens",
})
_CONTEXT_VAR_RE = re.compile(
    r"(?:context|token|prompt|window|chunk).*(?:limit|budget|max|size|cap|window|tokens|length)|"
    r"(?:max|limit|budget).*(?:input_tokens|context_tokens|prompt_tokens|tokens)",
    re.IGNORECASE,
)
_NON_CLIFF_VAR_RE = re.compile(r"(?:cliff|threshold|timeout|byte|bytes|char|chars|port|year|ms|delay)", re.IGNORECASE)
_DYNAMIC_CALL_CHAINS = frozenset({
    "datetime.now", "datetime.utcnow", "datetime.datetime.now", "datetime.datetime.utcnow",
    "date.today", "datetime.date.today", "time.time", "time.time_ns", "time.monotonic",
    "time.perf_counter", "uuid.uuid4", "uuid.uuid1", "random.random", "random.randint",
    "random.randrange", "random.choice", "secrets.token_hex", "secrets.token_urlsafe",
    "secrets.randbelow", "os.urandom",
})
_DYNAMIC_SIMPLE_NAMES = frozenset({"utcnow", "uuid4", "uuid1", "perf_counter"})
_ABSTRACT_DECORATORS = frozenset({"abstractmethod", "abstractproperty", "abstractclassmethod", "abstractstaticmethod"})
_ABSTRACT_BASES = frozenset({"ABC", "ABCMeta", "Protocol"})
_RULE_METADATA_CTORS = frozenset({"RuleMetadata", "ModelRetirement", "RuleDefinition", "Finding"})
_TIMEOUT_FUNC_NAMES = frozenset({"timeout", "Timeout", "wait_for"})


@dataclass
class _ScopeFrame:
    str_vars: dict[str, str] = field(default_factory=dict)
    num_vars: dict[str, float] = field(default_factory=dict)
    dynamic_vars: set[str] = field(default_factory=set)
    history_vars: set[str] = field(default_factory=set)
    outer_timeouts: list[tuple[float, str, int]] = field(default_factory=list)


def _attr_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            cur = None
        else:
            break
    parts.reverse()
    return parts


def _dotted_name(node: ast.AST) -> str:
    return ".".join(_attr_chain(node))


def _str_literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _num_literal(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _num_literal(node.operand)
        return -inner if inner is not None else None
    return None


def _int_literal(node: ast.AST | None) -> int | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) else None


def _is_empty_or_none(node: ast.AST | None) -> bool:
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return True
    if isinstance(node, ast.Dict):
        return len(node.keys) == 0
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) == 0
    return False


def _extract_suppressions(source: str) -> tuple[dict[int, set[str] | None], set[int]]:
    suppressions: dict[int, set[str] | None] = {}
    standalone_comment_lines: set[int] = set()
    code_lines: set[int] = set()
    comment_lines: dict[int, str] = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                line_no = tok.start[0]
                comment_lines[line_no] = tok.string
                has_dir, rule_ids = parse_suppression_directive(tok.string)
                if has_dir:
                    if line_no in suppressions:
                        existing = suppressions[line_no]
                        if existing is None or rule_ids is None:
                            suppressions[line_no] = None
                        else:
                            existing.update(rule_ids)
                    else:
                        suppressions[line_no] = rule_ids
            elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                code_lines.add(tok.start[0])
    except tokenize.TokenError:
        for idx, raw_line in enumerate(source.splitlines(), start=1):
            if "#" in raw_line:
                has_dir, rule_ids = parse_suppression_directive(raw_line.split("#", 1)[1])
                if has_dir:
                    suppressions[idx] = rule_ids
                if not raw_line.split("#", 1)[0].strip():
                    standalone_comment_lines.add(idx)
        return suppressions, standalone_comment_lines
    for line_no in comment_lines:
        if line_no not in code_lines:
            standalone_comment_lines.add(line_no)
    return suppressions, standalone_comment_lines


class _PythonAstAuditor(ast.NodeVisitor):
    def __init__(
        self,
        source: str,
        file_path: str,
        suppressions: dict[int, set[str] | None],
        standalone_comment_lines: set[int],
    ) -> None:
        self.lines: list[str] = source.splitlines()
        self.file_path = file_path
        self.suppressions = suppressions
        self.standalone_comment_lines = standalone_comment_lines
        self.scopes: list[_ScopeFrame] = [_ScopeFrame()]
        self.rule_table_depth: int = 0
        self.abstract_class_depth: int = 0
        self.abstract_func_depth: int = 0
        self.current_func_name: str = ""
        self.loop_multiplier: int = 1
        self.findings: list[Finding] = []
        self.suppressed_count: int = 0
        self._seen: set[tuple[str, int, int]] = set()

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
        node: ast.AST,
        severity_override: Severity | None = None,
        recommendation_override: str | None = None,
        extra_suppress_lines: tuple[int, ...] = (),
    ) -> None:
        raw_line = getattr(node, "lineno", 1)
        line = raw_line if isinstance(raw_line, int) else 1
        raw_col = getattr(node, "col_offset", 0)
        col = (raw_col if isinstance(raw_col, int) else 0) + 1
        raw_end_line = getattr(node, "end_lineno", None)
        end_line = raw_end_line if isinstance(raw_end_line, int) else None
        raw_end_col = getattr(node, "end_col_offset", None)
        end_col = (raw_end_col + 1) if isinstance(raw_end_col, int) else None

        key = (rule_id, line, col)
        if key in self._seen:
            return
        self._seen.add(key)

        if self._is_suppressed(rule_id, (line,) + extra_suppress_lines):
            self.suppressed_count += 1
            return

        self.findings.append(
            make_finding(
                rule_id, message, self.file_path, line, col, end_line, end_col,
                self._snippet(line), severity_override, recommendation_override,
            )
        )

    def _resolve_str(self, node: ast.AST | None) -> str | None:
        lit = _str_literal(node)
        if lit is not None:
            return lit
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope.str_vars:
                    return scope.str_vars[node.id]
        return None

    def _resolve_num(self, node: ast.AST | None) -> float | None:
        lit = _num_literal(node)
        if lit is not None:
            return lit
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope.num_vars:
                    return scope.num_vars[node.id]
        if isinstance(node, ast.BinOp):
            left, right = self._resolve_num(node.left), self._resolve_num(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div) and right != 0:
                    return left / right
        return None

    def _active_outer_timeout(self) -> tuple[float, str, int] | None:
        for scope in reversed(self.scopes):
            if scope.outer_timeouts:
                return min(scope.outer_timeouts, key=lambda item: item[0])
        return None

    def _is_dynamic_expr(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope.dynamic_vars:
                    return node.id
            return None
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted in _DYNAMIC_CALL_CHAINS:
                return dotted + "()"
            tail = dotted.split(".")[-1] if dotted else ""
            if tail in _DYNAMIC_SIMPLE_NAMES:
                return dotted + "()"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                for arg in node.args:
                    if (hit := self._is_dynamic_expr(arg)):
                        return hit
                for kw in node.keywords:
                    if (hit := self._is_dynamic_expr(kw.value)):
                        return hit
            if isinstance(node.func, ast.Name) and node.func.id == "str" and node.args:
                return self._is_dynamic_expr(node.args[0])
            if isinstance(node.func, ast.Attribute):
                return self._is_dynamic_expr(node.func.value)
            return None
        if isinstance(node, ast.Attribute):
            return self._is_dynamic_expr(node.value)
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.FormattedValue) and (hit := self._is_dynamic_expr(part.value)):
                    return hit
            return None
        if isinstance(node, ast.BinOp):
            return self._is_dynamic_expr(node.left) or self._is_dynamic_expr(node.right)
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if (hit := self._is_dynamic_expr(elt)):
                    return hit
        return None

    def _check_model_usage(self, raw_model: str, node: ast.AST, extra_lines: tuple[int, ...] = ()) -> None:
        if self.rule_table_depth > 0:
            return
        retirement = match_retired_model(raw_model)
        if retirement is not None:
            status_label = "already shut down" if retirement.status == "shutdown" else f"retiring on {retirement.shutdown_date}"
            self._emit(
                "APF001", f"Model {raw_model!r} is {status_label}; migrate to {retirement.replacement}.", node,
                severity_override=retirement.severity,
                recommendation_override=f"Replace {raw_model!r} with {retirement.replacement}.",
                extra_suppress_lines=extra_lines,
            )

    def _check_payload_mapping(
        self,
        entries: Mapping[str, tuple[ast.AST, ast.AST]],
        owner_node: ast.AST,
        is_responses_api: bool,
        is_chat_completions_api: bool,
    ) -> None:
        if self.rule_table_depth > 0:
            return
        raw_line = getattr(owner_node, "lineno", 1)
        call_line = raw_line if isinstance(raw_line, int) else 1

        model_val_node: ast.AST | None = next((entries[k][1] for k in _MODEL_KEY_NAMES if k in entries), None)
        model_str = self._resolve_str(model_val_node)
        if model_val_node is not None and model_str is not None:
            self._check_model_usage(model_str, model_val_node, extra_lines=(call_line,))

        if is_chat_completions_api and model_str and is_responses_only_tool_model(model_str):
            for tool_key in ("tools", "functions"):
                if tool_key in entries and not _is_empty_or_none(entries[tool_key][1]):
                    self._emit(
                        "APF002", f"chat.completions cannot pass {tool_key!r} with Responses-only model {model_str!r}.",
                        entries[tool_key][0], extra_suppress_lines=(call_line,),
                    )

        effort_node: ast.AST | None = None
        effort_owner_node: ast.AST | None = None
        if "reasoning_effort" in entries:
            effort_owner_node, effort_node = entries["reasoning_effort"]
        elif "reasoning" in entries:
            r_owner, r_val = entries["reasoning"]
            if isinstance(r_val, ast.Dict):
                for dk, dv in zip(r_val.keys, r_val.values):
                    if _str_literal(dk) == "effort":
                        effort_owner_node = dk if dk is not None else r_owner
                        effort_node = dv
                        break
        effort_val = self._resolve_str(effort_node)
        if effort_val is not None and effort_owner_node is not None and model_str:
            effort_low = effort_val.lower()
            if effort_low == "none" and (is_no_none_reasoning_model(model_str) or is_gpt6_family(model_str)):
                self._emit("APF003", f"Model {model_str!r} rejects reasoning effort 'none' (HTTP 400).", effort_owner_node, extra_suppress_lines=(call_line,))
            elif effort_low == "minimal" and is_no_minimal_reasoning_model(model_str):
                self._emit("APF003", f"Model {model_str!r} rejects reasoning effort 'minimal' (HTTP 400); use 'low'.", effort_owner_node, extra_suppress_lines=(call_line,))

        if model_str and is_gpt62_model(model_str):
            for bad_param in sorted(GPT62_UNSUPPORTED_PARAMS):
                if bad_param in entries and not _is_empty_or_none(entries[bad_param][1]):
                    self._emit("APF003", f"Parameter {bad_param!r} is unsupported on GPT-6.2 model {model_str!r}.", entries[bad_param][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))
            if "service_tier" in entries and self._resolve_str(entries["service_tier"][1]) == "flex":
                self._emit("APF003", f"service_tier='flex' is unsupported on GPT-6.2 model {model_str!r}; use background mode.", entries["service_tier"][0], extra_suppress_lines=(call_line,))
            if "prompt_cache_retention" in entries and not _is_empty_or_none(entries["prompt_cache_retention"][1]):
                self._emit("APF006", f"Replace legacy 'prompt_cache_retention' with prompt_cache_options={{'ttl': '30m'}} on {model_str!r}.", entries["prompt_cache_retention"][0], extra_suppress_lines=(call_line,))

        if is_responses_api:
            if "response_format" in entries and not _is_empty_or_none(entries["response_format"][1]):
                self._emit("APF004", "Responses API rejects 'response_format'; use text={'format': ...}.", entries["response_format"][0], extra_suppress_lines=(call_line,))
            for legacy_tok in ("max_tokens", "max_completion_tokens"):
                if legacy_tok in entries and not _is_empty_or_none(entries[legacy_tok][1]):
                    self._emit("APF004", f"Responses API rejects {legacy_tok!r}; use 'max_output_tokens'.", entries[legacy_tok][0], extra_suppress_lines=(call_line,))
            if "messages" in entries and not _is_empty_or_none(entries["messages"][1]):
                self._emit("APF004", "Responses API uses 'input' instead of 'messages'.", entries["messages"][0], extra_suppress_lines=(call_line,))
        elif model_str and is_gpt6_family(model_str):
            if "max_tokens" in entries and not _is_empty_or_none(entries["max_tokens"][1]):
                self._emit("APF004", f"Legacy 'max_tokens' on {model_str!r}; prefer 'max_output_tokens' (Responses) or 'max_completion_tokens' (Chat).", entries["max_tokens"][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))
            if is_gpt62_model(model_str) and "response_format" in entries and not _is_empty_or_none(entries["response_format"][1]):
                self._emit("APF004", f"Legacy 'response_format' with {model_str!r}; migrate to Responses API text={{'format': ...}}.", entries["response_format"][0], severity_override=Severity.WARNING, extra_suppress_lines=(call_line,))

        if "instructions" in entries and not _is_empty_or_none(entries["instructions"][1]):
            instr_key_node, instr_val_node = entries["instructions"]
            dyn_hit = self._is_dynamic_expr(instr_val_node)
            if dyn_hit:
                self._emit("APF006", f"Dynamic expression {dyn_hit!r} in 'instructions' breaks prompt caching.", instr_val_node, extra_suppress_lines=(call_line,))
            elif is_responses_api and model_str and is_gpt62_model(model_str):
                self._emit("APF006", f"GPT-6.2 ({model_str}) appends top-level 'instructions' after 'input'; place reusable guidance in the first developer item in 'input'.", instr_key_node, extra_suppress_lines=(call_line,))

        for list_key in ("messages", "input"):
            if list_key in entries:
                list_val = entries[list_key][1]
                if isinstance(list_val, ast.List):
                    self._check_message_list_prefix(list_val, extra_line=call_line)

        for ctx_key in _CONTEXT_KEY_NAMES:
            if ctx_key in entries:
                val = self._resolve_num(entries[ctx_key][1])
                if val is not None and val > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                    self._emit("APF007", f"{ctx_key}={int(val):,} exceeds the 272,000-token GPT-6.2 pricing cliff.", entries[ctx_key][1], extra_suppress_lines=(call_line,))

    def _check_message_list_prefix(self, list_node: ast.List, extra_line: int | None = None) -> None:
        extra = (extra_line,) if extra_line is not None else ()
        seen_dynamic = False
        for item in list_node.elts:
            if isinstance(item, ast.Dict):
                role_node: ast.AST | None = None
                content_node: ast.AST | None = None
                for k, v in zip(item.keys, item.values):
                    ks = _str_literal(k)
                    if ks == "role":
                        role_node = v
                    elif ks == "content":
                        content_node = v
                role = self._resolve_str(role_node)
                dyn_hit = self._is_dynamic_expr(content_node)
                if role in ("system", "developer"):
                    if dyn_hit:
                        self._emit("APF006", f"Dynamic value {dyn_hit!r} inside {role!r} prompt prefix breaks prompt caching.", content_node or item, extra_suppress_lines=extra)
                    elif seen_dynamic:
                        self._emit("APF006", f"Static {role!r} prompt placed after a dynamic prefix item breaks prompt caching.", item, extra_suppress_lines=extra)
                elif dyn_hit:
                    seen_dynamic = True
            elif self._is_dynamic_expr(item):
                seen_dynamic = True

    def _check_dict_standalone(self, node: ast.Dict) -> None:
        if self.rule_table_depth > 0:
            return
        entries: dict[str, tuple[ast.AST, ast.AST]] = {}
        for k, v in zip(node.keys, node.values):
            ks = _str_literal(k)
            if ks is not None and k is not None:
                entries[ks] = (k, v)
        if not entries:
            return

        role = self._resolve_str(entries["role"][1]) if "role" in entries else None
        if role in ("system", "developer") and "content" in entries:
            if (dyn_hit := self._is_dynamic_expr(entries["content"][1])):
                self._emit("APF006", f"Dynamic value {dyn_hit!r} inside {role!r} prompt prefix breaks prompt caching.", entries["content"][1])

        has_model = any(k in entries for k in _MODEL_KEY_NAMES)
        has_responses_shape = "text" in entries and "input" in entries
        if has_model or has_responses_shape:
            self._check_payload_mapping(entries, node, is_responses_api=has_responses_shape, is_chat_completions_api=False)

        outer_pair: tuple[float, str] | None = None
        inner_pairs: list[tuple[float, str, ast.AST]] = []
        for key_name, (_k_node, v_node) in entries.items():
            val = self._resolve_num(v_node)
            if val is None:
                continue
            val_s = val / 1000.0 if key_name.lower().endswith("_ms") else val
            if _OUTER_TIMEOUT_RE.match(key_name):
                if outer_pair is None or val_s < outer_pair[0]:
                    outer_pair = (val_s, key_name)
            elif _INNER_TIMEOUT_RE.match(key_name):
                inner_pairs.append((val_s, key_name, v_node))
        if outer_pair is not None:
            for inner_s, inner_name, v_node in inner_pairs:
                if inner_s > outer_pair[0]:
                    self._emit("APF101", f"Child timeout {inner_name}={inner_s:g}s exceeds caller timeout {outer_pair[1]}={outer_pair[0]:g}s.", v_node)

    def _register_history_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name) and _HISTORY_NAME_RE.search(target.id):
            self.scopes[-1].history_vars.add(target.id)

    def _is_history_var(self, name: str) -> bool:
        return bool(_HISTORY_NAME_RE.search(name)) or any(name in s.history_vars for s in reversed(self.scopes))

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decos = {_dotted_name(d).split(".")[-1] for d in node.decorator_list}
        is_abstract = bool(decos & _ABSTRACT_DECORATORS)
        prev_func = self.current_func_name
        self.current_func_name = node.name
        if is_abstract:
            self.abstract_func_depth += 1

        frame = _ScopeFrame()
        all_args = list(node.args.posonlyargs) + list(node.args.args)
        num_no_default = len(all_args) - len(node.args.defaults)
        for idx, arg in enumerate(all_args):
            d_idx = idx - num_no_default
            if d_idx >= 0:
                default_node = node.args.defaults[d_idx]
                num_val = self._resolve_num(default_node)
                if num_val is not None and "timeout" in arg.arg.lower():
                    val_s = num_val / 1000.0 if arg.arg.lower().endswith("_ms") else num_val
                    if val_s > 0:
                        frame.outer_timeouts.append((val_s, arg.arg, node.lineno))
                if (str_val := self._resolve_str(default_node)) is not None:
                    frame.str_vars[arg.arg] = str_val
            if _HISTORY_NAME_RE.search(arg.arg):
                frame.history_vars.add(arg.arg)
        for kw_arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if kw_default is not None:
                num_val = self._resolve_num(kw_default)
                if num_val is not None and "timeout" in kw_arg.arg.lower():
                    val_s = num_val / 1000.0 if kw_arg.arg.lower().endswith("_ms") else num_val
                    if val_s > 0:
                        frame.outer_timeouts.append((val_s, kw_arg.arg, node.lineno))
                if (str_val := self._resolve_str(kw_default)) is not None:
                    frame.str_vars[kw_arg.arg] = str_val
            if _HISTORY_NAME_RE.search(kw_arg.arg):
                frame.history_vars.add(kw_arg.arg)

        self.scopes.append(frame)
        try:
            for d in node.decorator_list:
                self.visit(d)
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.scopes.pop()
            if is_abstract:
                self.abstract_func_depth -= 1
            self.current_func_name = prev_func

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names = {_dotted_name(b).split(".")[-1] for b in node.bases}
        is_abs_cls = bool(base_names & _ABSTRACT_BASES) or any(
            kw.arg == "metaclass" and _dotted_name(kw.value).split(".")[-1] in _ABSTRACT_BASES for kw in node.keywords
        )
        if is_abs_cls:
            self.abstract_class_depth += 1
        try:
            self.generic_visit(node)
        finally:
            if is_abs_cls:
                self.abstract_class_depth -= 1

    def _record_binding(self, name: str, rhs: ast.AST, line: int) -> None:
        s_val, num_val, dyn_val = self._resolve_str(rhs), self._resolve_num(rhs), self._is_dynamic_expr(rhs)
        if s_val is not None:
            self.scopes[-1].str_vars[name] = s_val
        if num_val is not None:
            self.scopes[-1].num_vars[name] = num_val
            val_s = num_val / 1000.0 if name.lower().endswith("_ms") else num_val
            if _OUTER_TIMEOUT_RE.match(name) and val_s > 0:
                self.scopes[-1].outer_timeouts.append((val_s, name, line))
            elif _INNER_TIMEOUT_RE.match(name):
                outer = self._active_outer_timeout()
                if outer is not None and val_s > outer[0]:
                    self._emit("APF101", f"Child timeout {name}={val_s:g}s exceeds outer timeout {outer[1]}={outer[0]:g}s.", rhs, extra_suppress_lines=(line,))
        if dyn_val is not None:
            self.scopes[-1].dynamic_vars.add(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        target_names: list[str] = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                target_names.append(t.id)
                self._register_history_target(t)
            elif isinstance(t, ast.Subscript):
                base: ast.AST = t.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and self._is_history_var(base.id):
                    self._emit("APF005", f"In-place subscript edit on conversation history {base.id!r} drops GPT-6.2 reasoning items.", node)

        is_rule_assign = any(is_rule_table_identifier(n) for n in target_names)
        if is_rule_assign:
            self.rule_table_depth += 1
        try:
            if (sliced_hist := self._expr_slices_history(node.value)) is not None:
                self._emit("APF005", f"Slicing/rewriting conversation history {sliced_hist!r} mutates earlier turns and drops GPT-6.2 reasoning items.", node)
                for n in target_names:
                    self.scopes[-1].history_vars.add(n)

            if self.rule_table_depth == 0:
                for n in target_names:
                    if "model" in n.lower() and (s_val := self._resolve_str(node.value)) is not None:
                        self._check_model_usage(s_val, node.value, extra_lines=(node.lineno,))
                    if _SYSTEM_VAR_RE.search(n) and (dyn := self._is_dynamic_expr(node.value)):
                        self._emit("APF006", f"Dynamic expression {dyn!r} assigned to system/developer prompt variable {n!r} breaks prompt caching.", node.value, extra_suppress_lines=(node.lineno,))
                    if _CONTEXT_VAR_RE.search(n) and not _NON_CLIFF_VAR_RE.search(n):
                        if (iv := _int_literal(node.value)) is not None and iv > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                            self._emit("APF007", f"{n}={iv:,} exceeds the 272,000-token GPT-6.2 pricing cliff.", node.value, extra_suppress_lines=(node.lineno,))

            for n in target_names:
                self._record_binding(n, node.value, node.lineno)

            if isinstance(node.value, ast.List) and any(self._is_history_var(n) for n in target_names):
                self._check_message_list_prefix(node.value, extra_line=node.lineno)

            self.visit(node.value)
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    self.visit(t)
        finally:
            if is_rule_assign:
                self.rule_table_depth -= 1

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        is_rule_assign = isinstance(node.target, ast.Name) and is_rule_table_identifier(node.target.id)
        if is_rule_assign:
            self.rule_table_depth += 1
        try:
            if isinstance(node.target, ast.Name) and node.value is not None:
                n = node.target.id
                self._register_history_target(node.target)
                if (sliced_hist := self._expr_slices_history(node.value)) is not None:
                    self._emit("APF005", f"Slicing/rewriting conversation history {sliced_hist!r} mutates earlier turns and drops GPT-6.2 reasoning items.", node)
                    self.scopes[-1].history_vars.add(n)
                if self.rule_table_depth == 0:
                    if "model" in n.lower() and (s_val := self._resolve_str(node.value)) is not None:
                        self._check_model_usage(s_val, node.value, extra_lines=(node.lineno,))
                    if _SYSTEM_VAR_RE.search(n) and (dyn := self._is_dynamic_expr(node.value)):
                        self._emit("APF006", f"Dynamic expression {dyn!r} assigned to system/developer prompt variable {n!r} breaks prompt caching.", node.value, extra_suppress_lines=(node.lineno,))
                    if _CONTEXT_VAR_RE.search(n) and not _NON_CLIFF_VAR_RE.search(n):
                        if (iv := _int_literal(node.value)) is not None and iv > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                            self._emit("APF007", f"{n}={iv:,} exceeds the 272,000-token GPT-6.2 pricing cliff.", node.value, extra_suppress_lines=(node.lineno,))
                self._record_binding(n, node.value, node.lineno)
            if node.value is not None:
                self.visit(node.value)
        finally:
            if is_rule_assign:
                self.rule_table_depth -= 1

    def _expr_slices_history(self, expr: ast.AST) -> str | None:
        for child in ast.walk(expr):
            if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                if self._is_history_var(child.value.id) and isinstance(child.slice, ast.Slice):
                    sl = child.slice
                    if sl.lower is not None or sl.upper is not None or sl.step is not None:
                        return child.value.id
        return None

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base: ast.AST = target.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and self._is_history_var(base.id):
                    self._emit("APF005", f"Deleting from conversation history {base.id!r} mutates earlier turns and drops GPT-6.2 reasoning items.", node)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        pushed = 0
        for item in node.items:
            self.visit(item.context_expr)
            t_val = self._extract_with_timeout(item.context_expr)
            if t_val is not None and t_val > 0:
                raw_ln = getattr(item.context_expr, "lineno", node.lineno)
                ln = raw_ln if isinstance(raw_ln, int) else node.lineno
                self.scopes[-1].outer_timeouts.append((t_val, "enclosing timeout", ln))
                pushed += 1
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            for _ in range(pushed):
                self.scopes[-1].outer_timeouts.pop()

    def _extract_with_timeout(self, expr: ast.AST) -> float | None:
        if not isinstance(expr, ast.Call):
            return None
        dotted = _dotted_name(expr.func)
        tail = dotted.split(".")[-1] if dotted else ""
        if tail not in _TIMEOUT_FUNC_NAMES and "timeout" not in dotted.lower():
            return None
        for kw in expr.keywords:
            if kw.arg in ("timeout", "delay", "seconds"):
                if (val := self._resolve_num(kw.value)) is not None:
                    return val
        return self._resolve_num(expr.args[0]) if expr.args else None

    def visit_For(self, node: ast.For) -> None:
        mult = 1
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            if len(node.iter.args) == 1 and (n := self._resolve_num(node.iter.args[0])) is not None and n >= 2:
                mult = int(n)
            elif len(node.iter.args) >= 2:
                start, stop = self._resolve_num(node.iter.args[0]), self._resolve_num(node.iter.args[1])
                if start is not None and stop is not None and stop - start >= 2:
                    mult = int(stop - start)
        prev_mult = self.loop_multiplier
        self.loop_multiplier = prev_mult * mult
        try:
            self.generic_visit(node)
        finally:
            self.loop_multiplier = prev_mult

    def visit_Return(self, node: ast.Return) -> None:
        if (
            self.rule_table_depth == 0
            and node.value is not None
            and "model" in self.current_func_name.lower()
            and not any(s in self.current_func_name.lower() for s in ("retire", "deprecat", "rule", "check", "match", "lookup", "normalize", "strip"))
        ):
            if (s_val := _str_literal(node.value)) is not None:
                self._check_model_usage(s_val, node.value, extra_lines=(node.lineno,))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.rule_table_depth == 0:
            operands: list[ast.expr] = [node.left] + list(node.comparators)
            if any(isinstance(op, ast.Name) and any(k in op.id.lower() for k in ("token", "context")) for op in operands):
                for op in operands:
                    if (iv := _int_literal(op)) is not None and iv > CONTEXT_CLIFF_TOKEN_THRESHOLD:
                        self._emit("APF007", f"Token threshold {iv:,} exceeds the 272,000-token GPT-6.2 pricing cliff.", op, extra_suppress_lines=(node.lineno,))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self._check_dict_standalone(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        tail = dotted.split(".")[-1] if dotted else ""
        is_meta_ctor = tail in _RULE_METADATA_CTORS
        if is_meta_ctor:
            self.rule_table_depth += 1
        try:
            if self.rule_table_depth == 0:
                self._analyze_call(node, dotted, tail)
            pushed_wait_for = False
            if dotted.endswith("wait_for") or tail == "wait_for":
                for kw in node.keywords:
                    if kw.arg == "timeout" and (t_val := self._resolve_num(kw.value)) is not None and t_val > 0:
                        self.scopes[-1].outer_timeouts.append((t_val, "wait_for(timeout)", node.lineno))
                        pushed_wait_for = True
                        break
            try:
                self.generic_visit(node)
            finally:
                if pushed_wait_for:
                    self.scopes[-1].outer_timeouts.pop()
        finally:
            if is_meta_ctor:
                self.rule_table_depth -= 1

    def _analyze_call(self, node: ast.Call, dotted: str, tail: str) -> None:
        if ".beta.assistants" in dotted or ".beta.threads" in dotted or dotted.endswith(("beta.assistants", "beta.threads")):
            self._emit("APF001", f"Retired Assistants/Threads API call ({dotted}) shut down on 2026-08-26; migrate to Responses API.", node)

        for arg in node.args:
            if (s := _str_literal(arg)) is not None:
                self._check_endpoint_str(s, arg, node.lineno)
        for kw in node.keywords:
            if kw.arg in ("url", "endpoint", "path") and (s := _str_literal(kw.value)) is not None:
                self._check_endpoint_str(s, kw.value, node.lineno)

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            obj_name = node.func.value.id
            if self._is_history_var(obj_name) and node.func.attr in ("pop", "insert", "remove", "clear", "reverse", "sort"):
                self._emit("APF005", f"Calling .{node.func.attr}() on conversation history {obj_name!r} edits prior turns and drops GPT-6.2 reasoning items.", node)

        kw_entries: dict[str, tuple[ast.AST, ast.AST]] = {kw.arg: (kw, kw.value) for kw in node.keywords if kw.arg is not None}
        for hk in ("input", "messages"):
            if hk in kw_entries:
                hk_val = kw_entries[hk][1]
                if isinstance(hk_val, ast.Name):
                    self.scopes[-1].history_vars.add(hk_val.id)

        is_responses = ".responses." in dotted or dotted.startswith("responses.") or any((s := _str_literal(a)) is not None and "/v1/responses" in s for a in node.args)
        is_chat = ".chat.completions." in dotted or dotted.startswith("chat.completions.") or any((s := _str_literal(a)) is not None and "/v1/chat/completions" in s for a in node.args)

        if kw_entries:
            self._check_payload_mapping(kw_entries, node, is_responses, is_chat)

        for body_kw in ("json", "payload", "body"):
            if body_kw in kw_entries:
                body_val = kw_entries[body_kw][1]
                if isinstance(body_val, ast.Dict):
                    d_entries: dict[str, tuple[ast.AST, ast.AST]] = {
                        ks: (dk, dv)
                        for dk, dv in zip(body_val.keys, body_val.values)
                        if (ks := _str_literal(dk)) is not None and dk is not None
                    }
                    if d_entries:
                        self._check_payload_mapping(d_entries, node, is_responses, is_chat)

        if tail not in _TIMEOUT_FUNC_NAMES and tail != "wait_for":
            outer = self._active_outer_timeout()
            if outer is not None and "timeout" in kw_entries and (t_val := self._resolve_num(kw_entries["timeout"][1])) is not None:
                effective = t_val * self.loop_multiplier
                if t_val > outer[0]:
                    self._emit("APF101", f"Inner timeout={t_val:g}s on {dotted or tail}() exceeds {outer[1]}={outer[0]:g}s.", kw_entries["timeout"][1], extra_suppress_lines=(node.lineno,))
                elif self.loop_multiplier > 1 and effective > outer[0]:
                    self._emit("APF101", f"Loop retry budget ({self.loop_multiplier} x {t_val:g}s = {effective:g}s) exceeds {outer[1]}={outer[0]:g}s.", kw_entries["timeout"][1], extra_suppress_lines=(node.lineno,))

    def _check_endpoint_str(self, raw_url: str, node: ast.AST, call_line: int) -> None:
        for path, (shutdown_date, replacement) in RETIRED_API_ENDPOINTS.items():
            if path in raw_url:
                self._emit("APF001", f"Endpoint {path!r} retires/shut down on {shutdown_date}; migrate to {replacement}.", node, extra_suppress_lines=(call_line,))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self.rule_table_depth == 0:
            is_swallowed = all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in node.body)
            if is_swallowed:
                body_lines = tuple(ln for s in node.body if isinstance((ln := getattr(s, "lineno", None)), int))
                self._emit("APF102", "Swallowed exception in except block (only pass/constant); log, handle, or re-raise.", node, extra_suppress_lines=body_lines)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if self.rule_table_depth == 0 and self.abstract_class_depth == 0 and self.abstract_func_depth == 0 and node.exc is not None:
            exc = node.exc
            exc_name = exc.id if isinstance(exc, ast.Name) else (_dotted_name(exc.func).split(".")[-1] if isinstance(exc, ast.Call) else (exc.attr if isinstance(exc, ast.Attribute) else ""))
            if exc_name == "NotImplementedError":
                self._emit("APF103", "NotImplementedError stub in non-abstract production code.", node)
        self.generic_visit(node)


def scan_python_source(source: str, file_path: str) -> tuple[list[Finding], int]:
    """Audit Python source text and return (findings, suppressed_count)."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return [], 0
    suppressions, standalone_comments = _extract_suppressions(source)
    auditor = _PythonAstAuditor(source, file_path, suppressions, standalone_comments)
    auditor.visit(tree)
    return auditor.findings, auditor.suppressed_count
