from __future__ import annotations

import textwrap

from agent_preflight.js_scanner import scan_js_source
from agent_preflight.models import Severity
from agent_preflight.py_scanner import scan_python_source


def test_python_retiring_models_and_endpoints_apf001() -> None:
    source = textwrap.dedent(
        """
        client.responses.create(model="o3-pro", input="hi")
        client.responses.create(model="gpt-5-pro-2025-08-07", input="hi")
        client.responses.create(model="gpt-4.1-nano", input="hi")
        client.responses.create(model="gpt-5.1", input="hi")
        client.responses.create(model="gpt-5.3-codex", input="hi")
        client.beta.assistants.create(name="legacy")
        requests.post("https://api.example.com/v1/threads", headers={"Authorization": "Bearer sk-test-00000000"})
        """
    ).lstrip("\n")
    findings, _ = scan_python_source(source, "app.py")
    apf001 = [f for f in findings if f.rule_id == "APF001"]
    assert len(apf001) == 7
    severities = {f.severity for f in apf001}
    assert Severity.ERROR in severities
    assert Severity.WARNING in severities


def test_python_chat_completions_tools_and_reasoning_and_legacy_params() -> None:
    source = textwrap.dedent(
        """
        model_id = "gpt-6.2-sol"
        client.chat.completions.create(
            model=model_id,
            tools=[{"type": "function", "function": {"name": "search"}}],
            messages=[{"role": "user", "content": "hi"}],
        )
        client.responses.create(
            model="gpt-6.2-astra",
            reasoning={"effort": "none"},
            temperature=0.4,
            service_tier="flex",
            response_format={"type": "json_object"},
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
        client.responses.create(
            model="gpt-6.2-sol",
            reasoning={"effort": "minimal"},
            prompt_cache_retention="24h",
            input="hi",
        )
        """
    ).lstrip("\n")
    findings, _ = scan_python_source(source, "migrate.py")
    rule_ids = {f.rule_id for f in findings}
    assert {"APF002", "APF003", "APF004", "APF006"} <= rule_ids
    assert len([f for f in findings if f.rule_id == "APF003"]) >= 4
    apf004_msgs = [f.message for f in findings if f.rule_id == "APF004"]
    assert any("text=" in m for m in apf004_msgs)
    assert any("max_output_tokens" in m for m in apf004_msgs)
    assert any("input" in m for m in apf004_msgs)


def test_python_history_mutation_prompt_cache_and_context_cliff() -> None:
    source = textwrap.dedent(
        """
        import datetime, uuid

        def run(messages):
            messages[0] = {"role": "user", "content": "edit"}
            messages.pop(0)
            trimmed = messages[1:]
            bad_prompt = [
                {"role": "system", "content": f"Time {datetime.datetime.now()} req {uuid.uuid4()}"},
            ]
            system_prompt: str = f"Now {datetime.datetime.now()}"
            client.responses.create(
                model="gpt-6.2-astra",
                instructions="Reusable policy",
                input=[
                    {"role": "user", "content": f"id={uuid.uuid4()}"},
                    {"role": "developer", "content": "Static developer rules"},
                ],
                max_input_tokens=320_000,
            )
            return trimmed, bad_prompt, system_prompt
        """
    ).lstrip("\n")
    findings, _ = scan_python_source(source, "agent.py")
    rule_ids = {f.rule_id for f in findings}
    assert {"APF005", "APF006", "APF007"} <= rule_ids
    assert len([f for f in findings if f.rule_id == "APF005"]) >= 3
    assert len([f for f in findings if f.rule_id == "APF006"]) >= 4


def test_python_reliability_timeouts_swallowed_except_and_stubs() -> None:
    source = textwrap.dedent(
        """
        import asyncio, subprocess

        async def execute_pipeline():
            with asyncio.timeout(5):
                subprocess.run(["echo", "ok"], timeout=30)
                for _ in range(3):
                    subprocess.run(["echo", "retry"], timeout=2)
            try:
                risky()
            except Exception:
                pass
            raise NotImplementedError("pending")
        """
    ).lstrip("\n")
    findings, _ = scan_python_source(source, "worker.py")
    apf101 = [f for f in findings if f.rule_id == "APF101"]
    assert len(apf101) == 2
    assert any(f.rule_id == "APF102" for f in findings)
    assert any(f.rule_id == "APF103" for f in findings)


def test_python_safe_patterns_abstract_classes_and_inline_suppressions() -> None:
    source = textwrap.dedent(
        """
        from abc import ABC, abstractmethod
        from typing import Protocol

        DEPRECATED_MODELS = {"o3-pro", "gpt-5-pro", "gpt-5.1"}

        class AbstractTool(ABC):
            @abstractmethod
            def invoke(self) -> None:
                raise NotImplementedError

        class ToolProtocol(Protocol):
            def invoke(self) -> None:
                raise NotImplementedError

        def safe_call(history, model: str):
            if model in DEPRECATED_MODELS or model == "o3-pro":
                model = "gpt-6.2-astra"
            history.append({"role": "user", "content": "next turn"})
            client.responses.create(
                model="gpt-6.2-astra",
                reasoning={"effort": "low"},
                text={"format": {"type": "json_object"}},
                max_output_tokens=2048,
                prompt_cache_options={"ttl": "30m"},
                input=[{"role": "developer", "content": "Static prompt"}] + history,
                max_input_tokens=272_000,
            )
            # preflight: ignore[APF001]
            legacy_payload = {"model": "gpt-5.1"}
            try:
                optional_step()
            except ValueError:  # preflight: ignore
                pass
            return legacy_payload
        """
    ).lstrip("\n")
    findings, suppressed = scan_python_source(source, "safe.py")
    assert findings == []
    assert suppressed == 2


def test_typescript_and_tsx_all_rules_and_suppressions() -> None:
    ts_bad = textwrap.dedent(
        """
        async function runWorkflow(messages: Array<Record<string, string>>) {
          const callerTimeoutMs = 5000;
          await client.responses.create({ model: "o3-pro", input: "hi" });
          await client.responses.create({ model: "gpt-5.3-codex", input: "hi" });
          await client.chat.completions.create({
            model: "gpt-6.2-sol",
            tools: [{ type: "function", function: { name: "lookup" } }],
            messages,
          });
          await client.responses.create({
            model: "gpt-6.2-astra",
            reasoning: { effort: "none" },
            response_format: { type: "json_object" },
            max_tokens: 800,
            instructions: "Policy",
            prompt_cache_retention: "24h",
            input: messages,
          });
          messages[0] = { role: "user", content: "edited" };
          messages.splice(0, 1);
          const sysPrompt = `Now: ${Date.now()}`;
          const maxInputTokens = 300_000;
          await fetch("https://api.example.com/tool", { signal: AbortSignal.timeout(15000) });
          try {
            await work();
          } catch (err) {}
          throw new Error("Not implemented");
        }
        """
    ).lstrip("\n")
    findings, _ = scan_js_source(ts_bad, "workflow.tsx")
    observed = {f.rule_id for f in findings}
    assert {"APF001", "APF002", "APF003", "APF004", "APF005", "APF006", "APF007", "APF101", "APF102", "APF103"} <= observed

    ts_safe = textwrap.dedent(
        """
        abstract class BaseConnector {
          abstract send(): Promise<void>;
          fallback(): void {
            throw new Error("Not implemented");
          }
        }

        async function runSafe(history: any[]) {
          history.push({ role: "user", content: "next" });
          await client.responses.create({
            model: "gpt-6.2-astra",
            reasoning: { effort: "low" },
            text: { format: { type: "json_object" } },
            max_output_tokens: 1024,
            prompt_cache_options: { ttl: "30m" },
            input: [{ role: "developer", content: "Static prompt" }, ...history],
          });
          // preflight: ignore[APF001]
          const legacyModel = "gpt-5-pro";
          // preflight: ignore[APF002]
          await client.chat.completions.create({
            model: "gpt-6.2-sol",
            tools: [{ type: "function" }],
          });
          try { await work(); } catch (err) {} // preflight: ignore
          return legacyModel;
        }
        """
    ).lstrip("\n")
    safe_findings, suppressed = scan_js_source(ts_safe, "safe.ts")
    assert safe_findings == []
    assert suppressed == 3
