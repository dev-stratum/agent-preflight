from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from agent_preflight.js_scanner import scan_js_source
from agent_preflight.models import Finding, ScanReport, Severity
from agent_preflight.py_scanner import scan_python_source
from agent_preflight.sarif import sarif_to_json

TOOL_VERSION = "0.1.0"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".py", ".ts", ".tsx", ".js"})
DEFAULT_EXCLUDES: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", "dist", "build",
    ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox", ".eggs",
})


def _to_display_path(file_path: Path, anchor_dir: Path, fallback_root: Path | None = None) -> str:
    resolved_file = file_path.resolve()
    for base in (anchor_dir, fallback_root):
        if base is None:
            continue
        try:
            rel = resolved_file.relative_to(base.resolve()).as_posix()
            if rel != ".":
                return rel
        except ValueError:
            continue
    return file_path.name


def _is_excluded(rel_or_name: str, parts: tuple[str, ...], excludes: frozenset[str]) -> bool:
    if any(part in excludes for part in parts):
        return True
    return any(fnmatch.fnmatch(rel_or_name, pat) or any(fnmatch.fnmatch(p, pat) for p in parts) for pat in excludes)


def _discover_files(targets: Sequence[str], excludes: frozenset[str], anchor_dir: Path) -> list[tuple[Path, str]]:
    discovered: list[tuple[Path, str]] = []
    seen_resolved: set[Path] = set()
    for raw_target in targets:
        target = Path(raw_target)
        if not target.exists():
            raise FileNotFoundError(f"Scan target does not exist: {raw_target}")
        if target.is_file():
            resolved = target.resolve()
            if target.suffix.lower() in SUPPORTED_EXTENSIONS and resolved not in seen_resolved:
                seen_resolved.add(resolved)
                discovered.append((resolved, _to_display_path(resolved, anchor_dir)))
            continue
        resolved_target = target.resolve()
        for root_str, dirnames, filenames in os.walk(target):
            root_path = Path(root_str)
            dirnames[:] = sorted(
                d for d in dirnames
                if not _is_excluded(_to_display_path(root_path / d, anchor_dir, resolved_target), (root_path / d).parts, excludes)
            )
            for fname in sorted(filenames):
                fpath = root_path / fname
                if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                display = _to_display_path(fpath, anchor_dir, resolved_target)
                if _is_excluded(display, fpath.parts, excludes):
                    continue
                resolved = fpath.resolve()
                if resolved not in seen_resolved:
                    seen_resolved.add(resolved)
                    discovered.append((resolved, display))
    discovered.sort(key=lambda item: item[1])
    return discovered


def scan_paths(
    targets: Sequence[str],
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
    disabled_rules: Iterable[str] = (),
    anchor_dir: Path | None = None,
) -> ScanReport:
    """Scan Python, TypeScript, and JavaScript files and return a ScanReport."""
    anchor = (anchor_dir or Path.cwd()).resolve()
    exclude_set = frozenset(item.strip() for item in excludes if item.strip())
    disabled_set = frozenset(item.strip().upper() for item in disabled_rules if item.strip())
    files = _discover_files(targets, exclude_set, anchor)

    all_findings: list[Finding] = []
    suppressed_total = 0
    for abs_path, display_path in files:
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if abs_path.suffix.lower() == ".py":
            file_findings, suppressed = scan_python_source(source, display_path)
        else:
            file_findings, suppressed = scan_js_source(source, display_path)
        suppressed_total += suppressed
        for finding in file_findings:
            if finding.rule_id not in disabled_set:
                all_findings.append(finding)

    all_findings.sort(key=lambda f: (f.file_path, f.line, f.column, f.rule_id))
    return ScanReport(tuple(all_findings), len(files), suppressed_total, TOOL_VERSION)


def format_text_report(report: ScanReport) -> str:
    lines: list[str] = []
    for f in report.findings:
        lines.append(f"{f.file_path}:{f.line}:{f.column}: [{f.severity.value.upper()}] {f.rule_id} {f.message}")
        if f.recommendation:
            lines.append(f"  fix: {f.recommendation}")
    lines.append(
        f"agent-preflight: scanned {report.files_scanned} file(s), "
        f"{report.error_count} error(s), {report.warning_count} warning(s), "
        f"{report.note_count} note(s), {report.suppressed_count} suppressed."
    )
    return "\n".join(lines) + "\n"


def _escape_github_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def format_github_annotations(report: ScanReport) -> str:
    lines: list[str] = []
    for f in report.findings:
        cmd = "error" if f.severity is Severity.ERROR else ("warning" if f.severity is Severity.WARNING else "notice")
        msg = f"{f.rule_id}: {f.message} ({f.recommendation})" if f.recommendation else f"{f.rule_id}: {f.message}"
        lines.append(f"::{cmd} file={f.file_path},line={f.line},col={f.column}::{_escape_github_data(msg)}")
    lines.append(
        f"agent-preflight: scanned {report.files_scanned} file(s), "
        f"{report.error_count} error(s), {report.warning_count} warning(s), "
        f"{report.note_count} note(s), {report.suppressed_count} suppressed."
    )
    return "\n".join(lines) + "\n"


def format_report(report: ScanReport, output_format: str) -> str:
    fmt = output_format.strip().lower()
    if fmt == "text":
        return format_text_report(report)
    if fmt == "json":
        return json.dumps(report.to_dict(), indent=2) + "\n"
    if fmt == "sarif":
        return sarif_to_json(report)
    if fmt == "github":
        return format_github_annotations(report)
    raise ValueError(f"Unsupported output format: {output_format!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-preflight", description="Static OpenAI API migration and agent reliability scanner with SARIF 2.1.0 output.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan (default: current directory).")
    parser.add_argument("--format", choices=("text", "json", "sarif", "github"), default="text", help="Primary output format.")
    parser.add_argument("-o", "--output", default="", help="Write primary output to a file path instead of stdout.")
    parser.add_argument("--sarif-out", default="", help="Additionally write a SARIF 2.1.0 JSON report to this path.")
    parser.add_argument("--fail-on", choices=("error", "warning", "note", "none"), default="error", help="Minimum severity to exit non-zero.")
    parser.add_argument("--exclude", default="", help="Comma-separated directory names or glob patterns to exclude.")
    parser.add_argument("--no-default-excludes", action="store_true", help="Disable built-in exclude directories.")
    parser.add_argument("--disable", default="", help="Comma-separated rule IDs to skip.")
    parser.add_argument("--github-output", default="", help="Append summary outputs to a GitHub Actions GITHUB_OUTPUT file.")
    parser.add_argument("-V", "--version", action="version", version=f"agent-preflight {TOOL_VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    excludes: set[str] = set() if args.no_default_excludes else set(DEFAULT_EXCLUDES)
    if (raw_exclude := str(args.exclude)):
        excludes.update(item.strip() for item in raw_exclude.split(",") if item.strip())
    disabled = [item.strip().upper() for item in str(args.disable).split(",") if item.strip()]
    targets = [str(p) for p in args.paths]

    try:
        report = scan_paths(targets=targets, excludes=excludes, disabled_rules=disabled)
    except FileNotFoundError as exc:
        print(f"agent-preflight error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"agent-preflight I/O error: {exc}", file=sys.stderr)
        return 2

    rendered = format_report(report, str(args.format))
    if (out_file := str(args.output)):
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    if (sarif_out := str(args.sarif_out)):
        sarif_path = Path(sarif_out)
        sarif_path.parent.mkdir(parents=True, exist_ok=True)
        sarif_path.write_text(sarif_to_json(report), encoding="utf-8")

    if (gh_output := str(args.github_output)):
        gh_out = Path(gh_output)
        gh_out.parent.mkdir(parents=True, exist_ok=True)
        with gh_out.open("a", encoding="utf-8") as fh:
            fh.write(f"findings-count={report.total_count}\n")
            fh.write(f"errors-count={report.error_count}\n")
            fh.write(f"warnings-count={report.warning_count}\n")
            fh.write(f"files-scanned={report.files_scanned}\n")
            if sarif_out:
                fh.write(f"sarif-file={sarif_out}\n")

    return 1 if report.should_fail(str(args.fail_on)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
