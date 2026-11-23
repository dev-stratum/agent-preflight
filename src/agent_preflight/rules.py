from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent_preflight.models import Category, Finding, RuleMetadata, Severity

CONTEXT_CLIFF_TOKEN_THRESHOLD: int = 272_000

_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_SUPPRESSION_RE = re.compile(
    r"\bpreflight\s*:\s*ignore(?:-(?:next-line|line))?"
    r"(?:\s*\[([^\]]+)\]|\s+((?:APF\d{3}(?:\s*,\s*APF\d{3})*)))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ModelRetirement:
    """Lifecycle record for a retired or retiring OpenAI model."""

    model_id: str
    shutdown_date: str
    status: str
    replacement: str
    severity: Severity


MODEL_RETIREMENTS: dict[str, ModelRetirement] = {
    "computer-use-preview": ModelRetirement("computer-use-preview", "2026-07-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-4o-search-preview": ModelRetirement("gpt-4o-search-preview", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-4o-mini-search-preview": ModelRetirement("gpt-4o-mini-search-preview", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-4o-audio-preview": ModelRetirement("gpt-4o-audio-preview", "2026-07-23", "shutdown", "gpt-audio-1.5", Severity.ERROR),
    "gpt-4o-mini-audio-preview": ModelRetirement("gpt-4o-mini-audio-preview", "2026-07-23", "shutdown", "gpt-audio-1.5", Severity.ERROR),
    "gpt-5-chat-latest": ModelRetirement("gpt-5-chat-latest", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5-codex": ModelRetirement("gpt-5-codex", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5.1-chat-latest": ModelRetirement("gpt-5.1-chat-latest", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5.1-codex": ModelRetirement("gpt-5.1-codex", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5.1-codex-max": ModelRetirement("gpt-5.1-codex-max", "2026-07-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-5.1-codex-mini": ModelRetirement("gpt-5.1-codex-mini", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5.2-codex": ModelRetirement("gpt-5.2-codex", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "o3-deep-research": ModelRetirement("o3-deep-research", "2026-07-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "o4-mini-deep-research": ModelRetirement("o4-mini-deep-research", "2026-07-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "babbage-002": ModelRetirement("babbage-002", "2026-09-28", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "davinci-002": ModelRetirement("davinci-002", "2026-09-28", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-3.5-turbo-instruct": ModelRetirement("gpt-3.5-turbo-instruct", "2026-09-28", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-3.5-turbo-1106": ModelRetirement("gpt-3.5-turbo-1106", "2026-09-28", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-5.4-cyber": ModelRetirement("gpt-5.4-cyber", "2026-10-01", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-3.5-turbo-0125": ModelRetirement("gpt-3.5-turbo-0125", "2026-10-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-3.5-turbo": ModelRetirement("gpt-3.5-turbo", "2026-10-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-4-0613": ModelRetirement("gpt-4-0613", "2026-10-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-4-1106-preview": ModelRetirement("gpt-4-1106-preview", "2026-10-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-4-turbo": ModelRetirement("gpt-4-turbo", "2026-10-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-4": ModelRetirement("gpt-4", "2026-10-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "gpt-4.1-nano": ModelRetirement("gpt-4.1-nano", "2026-10-23", "shutdown", "gpt-5.6-luna or gpt-6-luna", Severity.ERROR),
    "o1": ModelRetirement("o1", "2026-10-23", "shutdown", "gpt-6.2-astra", Severity.ERROR),
    "o1-pro": ModelRetirement("o1-pro", "2026-10-23", "shutdown", "gpt-6.2-astra (or gpt-5.6-sol reasoning.mode=pro)", Severity.ERROR),
    "o3-mini": ModelRetirement("o3-mini", "2026-10-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "o4-mini": ModelRetirement("o4-mini", "2026-10-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-4o-2024-05-13": ModelRetirement("gpt-4o-2024-05-13", "2026-10-23", "shutdown", "gpt-6.2-sol", Severity.ERROR),
    "gpt-image-1": ModelRetirement("gpt-image-1", "2026-10-23", "shutdown", "gpt-image-2.6-sunburst", Severity.ERROR),
    "gpt-image-1-mini": ModelRetirement("gpt-image-1-mini", "2026-12-01", "imminent", "gpt-image-2.6-flare", Severity.ERROR),
    "gpt-image-1.5": ModelRetirement("gpt-image-1.5", "2026-12-01", "imminent", "gpt-image-2.6-sunburst", Severity.ERROR),
    "chatgpt-image-latest": ModelRetirement("chatgpt-image-latest", "2026-12-01", "imminent", "gpt-image-2.6-sunburst", Severity.ERROR),
    "o3-pro": ModelRetirement("o3-pro", "2026-12-11", "imminent", "gpt-6.2-astra (or gpt-5.6-sol reasoning.mode=pro)", Severity.ERROR),
    "o3": ModelRetirement("o3", "2026-12-11", "imminent", "gpt-6.2-astra (or gpt-5.6-sol)", Severity.ERROR),
    "gpt-5-pro": ModelRetirement("gpt-5-pro", "2026-12-11", "imminent", "gpt-6.2-astra (or gpt-5.6-sol reasoning.mode=pro)", Severity.ERROR),
    "gpt-5": ModelRetirement("gpt-5", "2026-12-11", "imminent", "gpt-6.2-astra (or gpt-5.6-sol)", Severity.ERROR),
    "gpt-5-mini": ModelRetirement("gpt-5-mini", "2026-12-11", "imminent", "gpt-6.2-sol (or gpt-5.6-terra)", Severity.ERROR),
    "gpt-5-nano": ModelRetirement("gpt-5-nano", "2026-12-11", "imminent", "gpt-6.2-sol (or gpt-5.6-luna)", Severity.ERROR),
    "tts-1": ModelRetirement("tts-1", "2027-01-06", "deprecated", "gpt-audio-1.5", Severity.WARNING),
    "tts-1-hd": ModelRetirement("tts-1-hd", "2027-01-06", "deprecated", "gpt-audio-1.5", Severity.WARNING),
    "gpt-4o-mini-tts": ModelRetirement("gpt-4o-mini-tts", "2027-01-06", "deprecated", "gpt-audio-1.5", Severity.WARNING),
    "gpt-4o": ModelRetirement("gpt-4o", "2027-01-14", "deprecated", "gpt-6.2-sol", Severity.WARNING),
    "gpt-4o-mini": ModelRetirement("gpt-4o-mini", "2027-01-14", "deprecated", "gpt-6-luna or gpt-6.2-sol", Severity.WARNING),
    "gpt-4.1": ModelRetirement("gpt-4.1", "2027-01-14", "deprecated", "gpt-6.2-sol", Severity.WARNING),
    "gpt-4.1-mini": ModelRetirement("gpt-4.1-mini", "2027-01-14", "deprecated", "gpt-6-luna or gpt-6.2-sol", Severity.WARNING),
    "gpt-realtime": ModelRetirement("gpt-realtime", "2027-01-20", "deprecated", "gpt-realtime-2.1", Severity.WARNING),
    "gpt-4o-realtime": ModelRetirement("gpt-4o-realtime", "2027-01-20", "deprecated", "gpt-realtime-2.1", Severity.WARNING),
    "gpt-audio": ModelRetirement("gpt-audio", "2027-01-20", "deprecated", "gpt-audio-1.5", Severity.WARNING),
    "gpt-5.5-cyber-preview": ModelRetirement("gpt-5.5-cyber-preview", "2027-02-01", "deprecated", "gpt-6.2-astra", Severity.WARNING),
    "gpt-image-2": ModelRetirement("gpt-image-2", "2027-02-10", "deprecated", "gpt-image-2.6-sunburst", Severity.WARNING),
    "whisper-1": ModelRetirement("whisper-1", "2027-02-26", "deprecated", "gpt-live-transcribe or gpt-transcribe", Severity.WARNING),
    "gpt-4o-transcribe": ModelRetirement("gpt-4o-transcribe", "2027-02-26", "deprecated", "gpt-transcribe", Severity.WARNING),
    "gpt-4o-mini-transcribe": ModelRetirement("gpt-4o-mini-transcribe", "2027-02-26", "deprecated", "gpt-transcribe", Severity.WARNING),
    "gpt-5.1": ModelRetirement("gpt-5.1", "2027-04-01", "deprecated", "gpt-6-sol or gpt-6.2-sol", Severity.WARNING),
    "gpt-5.3-codex": ModelRetirement("gpt-5.3-codex", "2027-04-01", "deprecated", "gpt-6-sol or gpt-6.2-astra", Severity.WARNING),
    "gpt-5.4-nano": ModelRetirement("gpt-5.4-nano", "2027-04-01", "deprecated", "gpt-6-luna or gpt-6.2-sol", Severity.WARNING),
}

RETIRED_API_ENDPOINTS: dict[str, tuple[str, str]] = {
    "/v1/assistants": ("2026-08-26", "Responses API (/v1/responses)"),
    "/v1/threads": ("2026-08-26", "Responses API (/v1/responses)"),
    "/v1/prompts": ("2026-11-30", "Responses API developer messages in input"),
    "/v1/evals": ("2026-11-30", "Responses API or repository eval harness"),
}

RESPONSES_ONLY_TOOL_MODELS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-astra", "gpt-6.2-sol"})
NO_NONE_REASONING_MODELS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-astra", "gpt-6.2-sol"})
NO_MINIMAL_REASONING_MODELS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-astra", "gpt-6.2-sol"})
GPT62_MODELS: frozenset[str] = frozenset({"gpt-6.2-astra", "gpt-6.2-sol"})
GPT62_UNSUPPORTED_PARAMS: frozenset[str] = frozenset({"temperature", "top_p", "logprobs", "top_logprobs"})

RULE_TABLE_NAME_HINTS: tuple[str, ...] = (
    "RULE_", "RULES", "DEPRECATED_MODEL", "RETIRED_MODEL", "SHUTDOWN_MODEL",
    "OBSOLETE_MODEL", "UNSUPPORTED_PARAM", "BLOCKED_MODEL", "MODEL_RETIREMENT",
    "LIFECYCLE_TABLE", "CATALOG", "REGISTRY", "MIGRATION_MAP", "NO_NONE_REASONING",
    "NO_MINIMAL_REASONING", "RESPONSES_ONLY", "RESPONSES_REQUIRED", "GPT62_MODELS",
    "GPT62_UNSUPPORTED", "PRICING_CLIFF", "CONTEXT_CLIFF_TOKEN_THRESHOLD",
)

RULES: dict[str, RuleMetadata] = {
    "APF001": RuleMetadata(
        "APF001", "ModelDeprecation",
        "Retired or imminently shutting down OpenAI model or endpoint",
        "Flags OpenAI models and endpoints shut down or scheduled for 2026-2027 shutdown (including o3-pro and gpt-5-pro on 2026-12-11, and gpt-5.1 and gpt-5.3-codex on 2027-04-01).",
        Category.API_LIFECYCLE, Severity.ERROR, "8.0", "high", ("security", "openai-api", "deprecation", "reliability"),
        "Migrate to an active GPT-6.2 model (gpt-6.2-astra or gpt-6.2-sol) on the Responses API.",
    ),
    "APF002": RuleMetadata(
        "APF002", "ResponsesApiRequiredForTools",
        "Function tools passed to Chat Completions for Responses-only model",
        "GPT-6 Astra, GPT-6.1 Sol, and GPT-6.2 models reject tool calling over v1/chat/completions; use v1/responses.",
        Category.RESPONSES_MIGRATION, Severity.ERROR, "8.5", "high", ("security", "openai-api", "gpt-6.2", "responses-api"),
        "Switch from client.chat.completions.create(..., tools=...) to client.responses.create(..., tools=...).",
    ),
    "APF003": RuleMetadata(
        "APF003", "UnsupportedReasoningOrSamplingParam",
        "Rejected reasoning effort or unsupported sampling parameter on GPT-6.x",
        "GPT-6 and GPT-6.2 reject reasoning_effort='none' and 'minimal' with HTTP 400; GPT-6.2 also rejects temperature, top_p, logprobs, top_logprobs, and service_tier='flex'.",
        Category.RESPONSES_MIGRATION, Severity.ERROR, "8.0", "high", ("security", "openai-api", "gpt-6.2", "reasoning"),
        "Use reasoning={'effort': 'low'} (or 'medium'/'high') and omit unsupported sampling/flex fields.",
    ),
    "APF004": RuleMetadata(
        "APF004", "LegacyResponsesParameters",
        "Legacy Chat Completions parameters used in Responses or GPT-6.2 calls",
        "Detects legacy response_format (replace with text.format), max_tokens/max_completion_tokens (replace with max_output_tokens), and messages (replace with input).",
        Category.RESPONSES_MIGRATION, Severity.ERROR, "7.5", "high", ("security", "openai-api", "gpt-6.2", "responses-api"),
        "Replace response_format with text.format, max_tokens with max_output_tokens, and messages with input.",
    ),
    "APF005": RuleMetadata(
        "APF005", "HistoryPrefixMutation",
        "Mid-conversation history edit drops GPT-6.2 reasoning items",
        "Mutating, slicing, popping, or rewriting earlier turns invalidates the prefix and silently drops stored GPT-6.2 reasoning items.",
        Category.RESPONSES_MIGRATION, Severity.WARNING, "6.5", "medium", ("security", "openai-api", "gpt-6.2", "agent-state"),
        "Keep prior turns append-only; use append/push or append a configuration_update item.",
    ),
    "APF006": RuleMetadata(
        "APF006", "PromptCachePrefixBreaker",
        "Dynamic prefix, trailing instructions, or legacy cache param breaks prompt caching",
        "Dynamic timestamps, UUIDs, or random values before/inside static system/developer prompts, top-level instructions, or legacy prompt_cache_retention on GPT-6.2 degrade prompt cache hits.",
        Category.CACHE_AND_COST, Severity.WARNING, "5.5", "high", ("security", "performance", "prompt-cache", "gpt-6.2"),
        "Place static reusable guidance in the leading developer message in input, and use prompt_cache_options={'ttl': '30m'} when configuring retention.",
    ),
    "APF007": RuleMetadata(
        "APF007", "ContextPricingCliff",
        "Context token configuration risks >272K GPT-6.2 pricing cliff",
        "Requests exceeding 272,000 input tokens on GPT-6.2 double input/cache rates and increase output pricing 1.5x across the full request.",
        Category.CACHE_AND_COST, Severity.WARNING, "5.0", "medium", ("security", "cost-control", "gpt-6.2"),
        "Cap input budgets at <=272,000 tokens or explicitly isolate long-context workflows.",
    ),
    "APF101": RuleMetadata(
        "APF101", "CallerOutlivingTimeout",
        "Child or downstream timeout exceeds enclosing caller timeout",
        "Inner tool, HTTP, subprocess, or retry timeouts exceeding the enclosing caller deadline create orphaned work and deadline races.",
        Category.RELIABILITY, Severity.ERROR, "7.8", "high", ("security", "reliability", "timeouts", "agents"),
        "Ensure every inner timeout and bounded retry loop fits strictly within the outer caller deadline.",
    ),
    "APF102": RuleMetadata(
        "APF102", "SwallowedException",
        "Exception or promise rejection silently swallowed",
        "Empty except/catch blocks silently hide API failures, tool execution errors, or partial writes in autonomous agent loops.",
        Category.SAFETY, Severity.ERROR, "7.5", "high", ("security", "reliability", "error-handling"),
        "Log, handle explicitly, or re-raise exceptions instead of using an empty except/catch block.",
    ),
    "APF103": RuleMetadata(
        "APF103", "StubPlaceholderInProduction",
        "Unimplemented stub error in non-abstract production code",
        "Raising NotImplementedError (or throwing 'Not implemented') outside abstract base classes or protocols risks runtime crashes.",
        Category.SAFETY, Severity.ERROR, "7.0", "high", ("security", "reliability", "completeness"),
        "Implement the production path or explicitly mark the base contract with @abstractmethod / abstract.",
    ),
}


def normalize_model_id(raw_model: str) -> str:
    """Normalize provider and fine-tune model identifiers."""
    text = raw_model.strip().lower()
    for prefix in ("openai:chat:", "openai:responses:", "openai/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.startswith("ft:"):
        parts = text.split(":")
        if len(parts) >= 2 and parts[1]:
            text = parts[1]
    return text


def strip_date_suffix(model_id: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model_id)


def match_retired_model(raw_model: str) -> ModelRetirement | None:
    """Return retirement metadata if raw_model matches a retired or retiring model."""
    normalized = normalize_model_id(raw_model)
    if normalized in MODEL_RETIREMENTS:
        return MODEL_RETIREMENTS[normalized]
    return MODEL_RETIREMENTS.get(strip_date_suffix(normalized))


def is_responses_only_tool_model(raw_model: str) -> bool:
    return strip_date_suffix(normalize_model_id(raw_model)) in RESPONSES_ONLY_TOOL_MODELS


def is_no_none_reasoning_model(raw_model: str) -> bool:
    return strip_date_suffix(normalize_model_id(raw_model)) in NO_NONE_REASONING_MODELS


def is_no_minimal_reasoning_model(raw_model: str) -> bool:
    return strip_date_suffix(normalize_model_id(raw_model)) in NO_MINIMAL_REASONING_MODELS


def is_gpt62_model(raw_model: str) -> bool:
    return strip_date_suffix(normalize_model_id(raw_model)) in GPT62_MODELS


def is_gpt6_family(raw_model: str) -> bool:
    return strip_date_suffix(normalize_model_id(raw_model)).startswith("gpt-6")


def is_rule_table_identifier(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in RULE_TABLE_NAME_HINTS)


def compute_fingerprint(rule_id: str, file_path: str, source_line: str) -> str:
    """Compute a stable line-content fingerprint compatible with GitHub Code Scanning."""
    normalized_line = " ".join(source_line.strip().split())
    payload = f"{rule_id}|{file_path}|{normalized_line}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def parse_suppression_directive(comment_text: str) -> tuple[bool, set[str] | None]:
    """Parse an inline suppression directive inside a comment string."""
    match = _SUPPRESSION_RE.search(comment_text)
    if not match:
        return False, None
    raw_ids = match.group(1) or match.group(2)
    if not raw_ids:
        return True, None
    rule_ids = {item.strip().upper() for item in raw_ids.split(",") if item.strip()}
    return True, rule_ids or None


def make_finding(
    rule_id: str,
    message: str,
    file_path: str,
    line: int,
    column: int = 1,
    end_line: int | None = None,
    end_column: int | None = None,
    snippet: str = "",
    severity_override: Severity | None = None,
    recommendation_override: str | None = None,
) -> Finding:
    """Construct a Finding populated from rule catalog defaults."""
    meta = RULES[rule_id]
    sev = severity_override if severity_override is not None else meta.default_severity
    rec = recommendation_override if recommendation_override is not None else meta.recommendation
    fp = compute_fingerprint(rule_id, file_path, snippet)
    return Finding(
        rule_id=rule_id, severity=sev, message=message, file_path=file_path,
        line=line, column=column, end_line=end_line, end_column=end_column,
        snippet=snippet.strip(), recommendation=rec, fingerprint=fp,
    )
