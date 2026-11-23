from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity levels aligned with SARIF 2.1.0 and CI gating."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"

    @property
    def rank(self) -> int:
        if self is Severity.ERROR:
            return 3
        if self is Severity.WARNING:
            return 2
        return 1


class Category(str, Enum):
    """Logical rule categories."""

    API_LIFECYCLE = "api-lifecycle"
    RESPONSES_MIGRATION = "responses-migration"
    CACHE_AND_COST = "cache-and-cost"
    RELIABILITY = "reliability"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class RuleMetadata:
    """Static definition metadata for a preflight audit rule."""

    rule_id: str
    name: str
    short_description: str
    full_description: str
    category: Category
    default_severity: Severity
    security_severity: str
    precision: str
    tags: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True, slots=True)
class Finding:
    """Single static analysis finding emitted by a scanner."""

    rule_id: str
    severity: Severity
    message: str
    file_path: str
    line: int
    column: int = 1
    end_line: int | None = None
    end_column: int | None = None
    snippet: str = ""
    recommendation: str = ""
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
        }
        if self.end_line is not None:
            payload["end_line"] = self.end_line
        if self.end_column is not None:
            payload["end_column"] = self.end_column
        if self.snippet:
            payload["snippet"] = self.snippet
        if self.recommendation:
            payload["recommendation"] = self.recommendation
        if self.fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Aggregated report across all scanned files."""

    findings: tuple[Finding, ...]
    files_scanned: int
    suppressed_count: int = 0
    tool_version: str = "0.1.0"
    extra_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.findings if item.severity is Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.findings if item.severity is Severity.WARNING)

    @property
    def note_count(self) -> int:
        return sum(1 for item in self.findings if item.severity is Severity.NOTE)

    def should_fail(self, fail_on: str) -> bool:
        normalized = fail_on.strip().lower()
        if normalized == "none":
            return False
        threshold_map = {"error": Severity.ERROR.rank, "warning": Severity.WARNING.rank, "note": Severity.NOTE.rank}
        threshold = threshold_map.get(normalized)
        if threshold is None:
            raise ValueError(f"Unsupported fail_on threshold: {fail_on!r}")
        return any(item.severity.rank >= threshold for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_version": self.tool_version,
            "files_scanned": self.files_scanned,
            "suppressed_count": self.suppressed_count,
            "summary": {
                "total": self.total_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
                "notes": self.note_count,
            },
            "findings": [item.to_dict() for item in self.findings],
        }
