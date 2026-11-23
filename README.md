# agent-preflight

[![CI](https://github.com/dev-stratum/agent-preflight/actions/workflows/tests.yml/badge.svg)](https://github.com/dev-stratum/agent-preflight/actions/workflows/tests.yml)
![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![mypy strict](https://img.shields.io/badge/typing-mypy%20--strict-blue)
![SARIF 2.1.0](https://img.shields.io/badge/reporting-SARIF%202.1.0-informational)
![MIT License](https://img.shields.io/badge/license-MIT-green)

**Pre-flight CI/CD shield for OpenAI API migrations, GPT-6.2 readiness, and autonomous agent reliability.**

`agent-preflight` scans Python (`.py`) and TypeScript/JavaScript (`.ts`, `.tsx`, `.js`) codebases to catch breaking model retirements, GPT-6.2 Responses API contract violations, prompt-cache killers, and persistent-agent failure modes before they hit production—and provides a lightweight runtime fail-closed stream gate to stop truncated tool calls from executing partial commands.

## Why agent-preflight?

1. **Imminent OpenAI Model & Surface Shutdowns:** Flags already shut-down models (`gpt-4`, `gpt-4-turbo`, `o1`, `o1-pro`, `o3-mini`, `o4-mini`, `gpt-4.1-nano`, `gpt-5-codex`, `gpt-5.1-codex`), imminent shutdowns (`o3`, `o3-pro`, `gpt-5`, `gpt-5-pro` on **Dec 11, 2026**; `gpt-image-1.5` on **Dec 1, 2026**; `/v1/prompts` and `/v1/evals` on **Nov 30, 2026**), Q1 2027 retirements (`gpt-4o`, `gpt-4.1` on **Jan 14, 2027**), and **Apr 1, 2027** shutdowns (`gpt-5.1`, `gpt-5.3-codex`, `gpt-5.4-nano`).
2. **GPT-6.2 & Responses API Migration Traps:** Catches `chat.completions` passing `tools` to Responses-only models (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.2-sol`, `gpt-6.2-astra`), rejected `reasoning_effort="none"` or `"minimal"`, unsupported sampling/`service_tier="flex"` parameters, legacy `response_format` / `max_tokens` / `messages` vs `text.format` / `max_output_tokens` / `input`, history prefix edits that silently drop reasoning items, prompt-cache breakers (dynamic prefixes, top-level `instructions`, legacy `prompt_cache_retention`), and >272K context pricing cliff budgets.
3. **Agent Reliability & Safety:** Catches child timeouts outliving caller deadlines, swallowed `except: pass` / `catch {}` blocks, production `NotImplementedError` stubs, and truncated SSE/JSON-RPC tool streams.

### Supply-Chain Security by Construction

`agent-preflight` ships with **zero third-party runtime dependencies**, built strictly on the Python Standard Library (`ast`, `tokenize`, `json`, `argparse`, `pathlib`, `re`). It makes zero outbound network calls and collects zero telemetry—delivering deterministic, offline-safe verification for strict enterprise CI environments.

## GitHub Action Quickstart

```yaml
name: Agent Pre-Flight

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read
  security-events: write

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run agent-preflight scan
        id: preflight
        uses: dev-stratum/agent-preflight@v0.1.0
        with:
          path: "src"
          format: "github"
          sarif-file: "agent-preflight.sarif"
          fail-on: "error"
      - name: Upload SARIF 2.1.0 report
        if: always() && steps.preflight.outputs.sarif-file != ''
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: ${{ steps.preflight.outputs.sarif-file }}
```

Uploaded SARIF 2.1.0 alerts integrate natively with GitHub Code Scanning PR annotations and can be pulled into `openai/codex-security` workflows via `npx @openai/codex-security import github`.

## Installation & CLI Usage

Install directly from GitHub:

```bash
pip install git+https://github.com/dev-stratum/agent-preflight.git
```

Run from the command line:

```bash
agent-preflight src/
agent-preflight src/ --format github --sarif-out agent-preflight.sarif --fail-on warning
agent-preflight app.py worker.ts --format json --disable APF007
```

Exit codes: `0` (clean / below `--fail-on`), `1` (findings at or above `--fail-on`), `2` (invalid usage / missing path).

## Rule Reference

| Rule | Severity | Scope | Detects |
| --- | --- | --- | --- |
| `APF001` | `error` / `warning` | API Lifecycle | Retired/retiring OpenAI models (`o3-pro`, `gpt-5-pro`, `gpt-5.1`, `gpt-5.3-codex`, etc.) and `/v1/assistants`, `/v1/threads`, `/v1/prompts`, `/v1/evals` |
| `APF002` | `error` | Responses Migration | `chat.completions` passing `tools`/`functions` to `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.2-astra`, or `gpt-6.2-sol` |
| `APF003` | `error` / `warning` | GPT-6.x Reasoning | Rejected `reasoning_effort="none"` / `"minimal"` and unsupported GPT-6.2 sampling/`service_tier="flex"` params |
| `APF004` | `error` / `warning` | Responses Migration | Legacy `response_format` (use `text.format`), `max_tokens`/`max_completion_tokens` (use `max_output_tokens`), and `messages` (use `input`) |
| `APF005` | `warning` | GPT-6.2 Agent State | In-place mutation, deletion, `pop`/`splice`/`shift`, or slice rewrites on prior turn history dropping GPT-6.2 reasoning items |
| `APF006` | `warning` | Prompt Caching | Dynamic timestamps/UUIDs before or inside system/developer prompts, top-level `instructions`, or legacy `prompt_cache_retention` on GPT-6.2 |
| `APF007` | `warning` | Cost Control | Token budgets exceeding `272,000` tokens (GPT-6.2 2x input/cache and 1.5x output pricing cliff) |
| `APF101` | `error` | Reliability | Inner timeouts or retry budgets exceeding an enclosing caller timeout |
| `APF102` | `error` | Safety | Empty `except: pass`, `catch {}`, or `.catch(() => {})` swallowing failures |
| `APF103` | `error` | Safety | `NotImplementedError` / `"Not implemented"` stubs outside abstract base classes, `@abstractmethod`, or interfaces |

Inline suppressions: `# preflight: ignore`, `# preflight: ignore[APF001]`, or `// preflight: ignore`. Rule-definition constants and comparison expressions (`if model in {...}`) are excluded via AST context.

## Runtime Fail-Closed Stream Gate

```python
from agent_preflight import StreamToolCallGate, guard_tool_stream

for call in guard_tool_stream(sse_lines):  # OpenAI Chat, Responses, Anthropic, or MCP
    execute_tool(call.name, call.arguments)

gate = StreamToolCallGate(protocol="auto")
for event in stream_events:
    gate.feed_event(event)
result = gate.evaluate()
if result.safe_to_execute:
    for call in result.tool_calls:
        execute_tool(call.name, call.arguments)
```
