"""Zero-dependency OpenAI API migration and agent reliability pre-flight scanner."""

from __future__ import annotations

from agent_preflight.models import Category, Finding, RuleMetadata, ScanReport, Severity
from agent_preflight.rules import CONTEXT_CLIFF_TOKEN_THRESHOLD, RULES
from agent_preflight.sarif import SARIF_SCHEMA_URL, SARIF_VERSION, build_sarif_report, sarif_to_json
from agent_preflight.stream_gate import (
    StreamGateResult,
    StreamToolCallGate,
    StreamTruncationError,
    ToolCallDraft,
    guard_tool_stream,
)

__version__ = "0.1.0"

__all__ = [
    "CONTEXT_CLIFF_TOKEN_THRESHOLD",
    "Category",
    "Finding",
    "RULES",
    "RuleMetadata",
    "SARIF_SCHEMA_URL",
    "SARIF_VERSION",
    "ScanReport",
    "Severity",
    "StreamGateResult",
    "StreamToolCallGate",
    "StreamTruncationError",
    "ToolCallDraft",
    "__version__",
    "build_sarif_report",
    "guard_tool_stream",
    "sarif_to_json",
]
