from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from pytest import CaptureFixture

from agent_preflight import RULES
from agent_preflight.cli import main, scan_paths
from agent_preflight.sarif import SARIF_SCHEMA_URL, SARIF_VERSION, build_sarif_report

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_exit_codes_and_output_formats(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    bad_file = tmp_path / "bad_agent.py"
    bad_file.write_text('client.responses.create(model="o3-pro", input="hello")\n', encoding="utf-8")

    clean_file = tmp_path / "clean_agent.py"
    clean_file.write_text(
        'client.responses.create(model="gpt-6.2-astra", input=[{"role": "developer", "content": "Hi"}])\n',
        encoding="utf-8",
    )

    assert main([str(tmp_path / "does_not_exist.py")]) == 2
    assert main([str(clean_file), "--fail-on", "warning"]) == 0
    capsys.readouterr()

    sarif_path = tmp_path / "out.sarif"
    gh_out = tmp_path / "gh_output.txt"
    rc = main([str(bad_file), "--format", "github", "--sarif-out", str(sarif_path), "--github-output", str(gh_out), "--fail-on", "error"])
    assert rc == 1
    stdout = capsys.readouterr().out
    assert "::error " in stdout
    assert "APF001" in stdout
    assert "findings-count=1" in gh_out.read_text(encoding="utf-8")

    assert main([str(bad_file), "--disable", "APF001"]) == 0
    capsys.readouterr()

    assert main([str(bad_file), "--format", "json", "--fail-on", "none"]) == 0
    json_out = json.loads(capsys.readouterr().out)
    assert json_out["summary"]["total"] == 1
    assert json_out["findings"][0]["rule_id"] == "APF001"
    proc = subprocess.run([sys.executable, "-m", "agent_preflight.cli", str(bad_file), "--fail-on", "error"], capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert "APF001" in proc.stdout



def test_sarif_210_schema_and_codex_security_compatibility(tmp_path: Path) -> None:
    sample = tmp_path / "sub" / "service.ts"
    sample.parent.mkdir(parents=True)
    sample.write_text('await client.chat.completions.create({ model: "gpt-6.2-sol", tools: [{ type: "function" }] });\n', encoding="utf-8")

    report = scan_paths([str(sample)], anchor_dir=tmp_path)
    sarif = build_sarif_report(report)
    assert sarif["version"] == SARIF_VERSION == "2.1.0"
    assert sarif["$schema"] == SARIF_SCHEMA_URL
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "agent-preflight"
    assert len(driver["rules"]) == len(RULES)
    for rule_obj in driver["rules"]:
        sec_sev = float(rule_obj["properties"]["security-severity"])
        assert 0.0 < sec_sev <= 10.0
        assert rule_obj["defaultConfiguration"]["level"] in ("error", "warning", "note")

    assert run["originalUriBaseIds"]["%SRCROOT%"]["description"]["text"] == "Repository root"
    assert run["invocations"][0]["executionSuccessful"] is True
    res = run["results"][0]
    assert res["ruleId"] == "APF002"
    loc = res["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "sub/service.ts"
    assert loc["artifactLocation"]["uriBaseId"] == "%SRCROOT%"
    assert "\\" not in loc["artifactLocation"]["uri"]
    assert loc["region"]["startLine"] == 1
    assert len(res["partialFingerprints"]["primaryLocationLineHash"]) == 32


def test_dogfood_src_zero_findings_and_zero_third_party_imports() -> None:
    src_root = REPO_ROOT / "src"
    report = scan_paths([str(src_root)], anchor_dir=REPO_ROOT)
    assert report.files_scanned >= 9
    assert report.total_count == 0
    assert report.suppressed_count == 0
    assert (src_root / "agent_preflight" / "py.typed").read_bytes() == b"\n"

    allowed_roots = set(sys.stdlib_module_names) | {"agent_preflight"}
    for py_file in sorted((src_root / "agent_preflight").glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in allowed_roots, f"{py_file}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] in allowed_roots, f"{py_file}: {node.module}"


def test_repository_zero_pii_and_zero_machine_path_leaks() -> None:
    banned_paths = (
        "C:" + chr(92),
        "D:" + chr(92),
        chr(47) + "home" + chr(47),
        chr(47) + "Users" + chr(47),
        "%" + "USERPROFILE" + "%",
    )
    email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
    allowed_emails = {"335769801+dev-stratum@users.noreply.github.com"}
    allowed_email_domains = ("@example.com", "@example.org", "@example.net")
    real_sk_re = re.compile(r"\bsk-[A-Za-z0-9\-]{10,}\b")

    targets: list[Path] = []
    for folder in ("src", "tests", ".github"):
        base = REPO_ROOT / folder
        if base.exists():
            targets.extend(p for p in base.rglob("*") if p.is_file())
    for top_name in ("action.yml", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", ".gitignore"):
        if (p := REPO_ROOT / top_name).exists():
            targets.append(p)

    for path in targets:
        if path.suffix in (".pyc", ".pyo"):
            continue
        content = path.read_text(encoding="utf-8")
        for pat in banned_paths:
            assert pat not in content, f"Banned machine path {pat!r} leaked in {path}"
        for email in email_re.findall(content):
            low = email.lower()
            if low not in allowed_emails and not low.endswith(allowed_email_domains):
                raise AssertionError(f"Unexpected email address {email!r} in {path}")
        for key in real_sk_re.findall(content):
            assert key == "sk-test-00000000", f"Suspicious API key pattern {key!r} in {path}"
