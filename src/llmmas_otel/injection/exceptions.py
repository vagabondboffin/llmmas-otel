from __future__ import annotations


class LLMFaultError(RuntimeError):
    """Base class for injected LLM-call faults."""


class LLMRateLimitError(LLMFaultError):
    """Injected rate-limit error."""


class LLMNetworkError(LLMFaultError):
    """Injected network error."""


class LLMTimeoutError(LLMFaultError):
    """Injected timeout error."""