from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, BatchSpanProcessor

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def _set_provider(*, service_name: str) -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return provider


def init_console_tracing(*, service_name: str = "llmmas-otel-demo") -> None:
    provider = _set_provider(service_name=service_name)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def init_otlp_tracing(
    *,
    service_name: str = "llmmas-otel-demo",
    endpoint: str = "http://localhost:4317",
    insecure: bool = True,
) -> None:

    provider = _set_provider(service_name=service_name)

    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    provider.add_span_processor(BatchSpanProcessor(exporter))
