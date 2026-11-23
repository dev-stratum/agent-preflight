from __future__ import annotations

import json
from typing import Any

from agent_preflight.models import Finding, ScanReport, Severity
from agent_preflight.rules import RULES

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA_URL = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_INFORMATION_URI = "https://github.com/dev-stratum/agent-preflight"


def _sarif_level(severity: Severity) -> str:
    if severity is Severity.ERROR:
        return "error"
    if severity is Severity.WARNING:
        return "warning"
    return "note"


def _normalize_sarif_uri(file_path: str) -> str:
    uri = file_path.replace("\\", "/")
    while uri.startswith("./"):
        uri = uri[2:]
    return uri.lstrip("/")


def _rule_entry(rule_id: str) -> dict[str, Any]:
    meta = RULES[rule_id]
    return {
        "id": meta.rule_id,
        "name": meta.name,
        "shortDescription": {"text": meta.short_description},
        "fullDescription": {"text": meta.full_description},
        "helpUri": f"{TOOL_INFORMATION_URI}#rule-{meta.rule_id.lower()}",
        "help": {
            "text": meta.recommendation,
            "markdown": f"**Recommendation:** {meta.recommendation}\n\n{meta.full_description}",
        },
        "defaultConfiguration": {"level": _sarif_level(meta.default_severity)},
        "properties": {
            "tags": list(meta.tags),
            "precision": meta.precision,
            "problem.severity": _sarif_level(meta.default_severity),
            "security-severity": meta.security_severity,
        },
    }


def _result_entry(finding: Finding, rule_index: int) -> dict[str, Any]:
    uri = _normalize_sarif_uri(finding.file_path)
    region: dict[str, Any] = {"startLine": max(1, finding.line), "startColumn": max(1, finding.column)}
    if finding.end_line is not None and finding.end_line >= finding.line:
        region["endLine"] = finding.end_line
    if finding.end_column is not None and finding.end_column >= 1:
        if finding.end_line is None or finding.end_line > finding.line or finding.end_column > finding.column:
            region["endColumn"] = finding.end_column
    if finding.snippet:
        region["snippet"] = {"text": finding.snippet}

    msg_text = f"{finding.message} Recommendation: {finding.recommendation}" if finding.recommendation else finding.message
    entry: dict[str, Any] = {
        "ruleId": finding.rule_id,
        "ruleIndex": rule_index,
        "level": _sarif_level(finding.severity),
        "message": {"text": msg_text},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"},
                    "region": region,
                }
            }
        ],
    }
    if finding.fingerprint:
        entry["partialFingerprints"] = {"primaryLocationLineHash": finding.fingerprint}
    return entry


def build_sarif_report(report: ScanReport) -> dict[str, Any]:
    """Serialize a ScanReport to an OASIS SARIF 2.1.0 dictionary."""
    ordered_rule_ids = sorted(RULES.keys())
    rule_index_map = {rid: idx for idx, rid in enumerate(ordered_rule_ids)}
    return {
        "$schema": SARIF_SCHEMA_URL,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "agent-preflight",
                        "semanticVersion": report.tool_version,
                        "informationUri": TOOL_INFORMATION_URI,
                        "rules": [_rule_entry(rid) for rid in ordered_rule_ids],
                    }
                },
                "automationDetails": {"id": "agent-preflight/"},
                "originalUriBaseIds": {"%SRCROOT%": {"description": {"text": "Repository root"}}},
                "invocations": [{"executionSuccessful": True}],
                "results": [_result_entry(finding, rule_index_map[finding.rule_id]) for finding in report.findings],
            }
        ],
    }


def sarif_to_json(report: ScanReport, indent: int = 2) -> str:
    """Render a ScanReport as deterministic SARIF 2.1.0 JSON."""
    return json.dumps(build_sarif_report(report), indent=indent) + "\n"
