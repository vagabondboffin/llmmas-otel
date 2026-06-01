"""Tests for MUTATE_INPUT consumption in the observe_llm_call decorator."""
import pytest

from llmmas_otel.decorators import observe_llm_call
from llmmas_otel.injection import InjectionDecision
from llmmas_otel.span_factory import default_span_factory


@pytest.fixture()
def mem_exporter(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("llmmas-otel")
    monkeypatch.setattr(default_span_factory, "_tracer", tracer)
    return exporter


def test_mutate_input_llm_mutates_args_kwargs(monkeypatch, mem_exporter):
    def mutator(args, kwargs):
        return ("mutated prompt",), {"temperature": 0.2}

    monkeypatch.setattr(
        default_span_factory,
        "current_llm_call_decision",
        lambda: InjectionDecision.mutate_input(
            fault_id="test-mutate",
            fault_type="llm.prompt_inject",
            mutator=mutator,
        ),
    )

    seen = {}

    @observe_llm_call(provider_name="test", model="fake-model")
    def fake_llm(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return "result"

    result = fake_llm("original prompt", temperature=1.0)

    assert result == "result"
    assert seen["prompt"] == "mutated prompt"
    assert seen["kwargs"]["temperature"] == 0.2


def test_mutate_input_llm_bad_mutator_does_not_crash(monkeypatch, mem_exporter):
    def bad_mutator(args, kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        default_span_factory,
        "current_llm_call_decision",
        lambda: InjectionDecision.mutate_input(
            fault_id="test-bad-mutator",
            fault_type="llm.prompt_inject",
            mutator=bad_mutator,
        ),
    )

    seen = {}

    @observe_llm_call(provider_name="test", model="fake-model")
    def fake_llm(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return "result"

    result = fake_llm("original prompt", temperature=1.0)

    assert result == "result"
    assert seen["prompt"] == "original prompt"
    assert seen["kwargs"]["temperature"] == 1.0

    spans = mem_exporter.get_finished_spans()
    fault_attrs = [
        span.attributes.get("llmmas.fault.error")
        for span in spans
        if span.attributes.get("llmmas.fault.error")
    ]
    assert fault_attrs, "Expected llmmas.fault.error on span"
    assert fault_attrs[0].startswith("mutator_failed: ")
