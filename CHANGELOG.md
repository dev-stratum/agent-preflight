# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- Zero-dependency Python Standard Library static scanners for `.py`, `.ts`, `.tsx`, and `.js`.
- OpenAI lifecycle and GPT-6.2 migration rules (`APF001`–`APF007`) covering 2026–2027 shutdowns, Responses-only tool calling, `reasoning.effort` none/minimal rejections, `text.format`/`max_output_tokens`, prefix edits, prompt caching (`instructions`, `prompt_cache_retention`), and the 272K context pricing cliff.
- Agent reliability and safety rules (`APF101`–`APF103`) for caller-outliving timeouts, swallowed exceptions, and production stubs.
- Inline suppression directives (`# preflight: ignore`, `// preflight: ignore`) and context-aware rule-table exclusions.
- Runtime fail-closed `StreamToolCallGate` and `guard_tool_stream` for OpenAI Chat, Responses API, Anthropic, and MCP streams.
- OASIS SARIF 2.1.0 exporter, cross-OS Composite GitHub Action (`action.yml`), and CLI (`agent-preflight`).
