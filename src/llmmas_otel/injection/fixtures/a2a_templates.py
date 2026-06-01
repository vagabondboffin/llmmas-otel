"""Templated A2A payloads for mutation faults, keyed by sender agent role."""

# Plausible-but-false structured claims that name a specific function /
# file / test count. Use for `a2a.hallucinate`.
HALLUCINATION_TEMPLATES = {
    "Navigator": (
        "Final answer: Found function `process_request` at "
        "src/core/handler.py:142. It calls `validate_input` on line 156."
    ),
    "Editor": (
        "Patch applied successfully to src/core/handler.py. "
        "The flag `strict_mode` was changed from False to True."
    ),
    "Executor": (
        "All tests passed. test_handler.py: 14 passed, 0 failed in 2.3s."
    ),
}

# Confident task-complete claims without specifics. Use for
# `a2a.confident_wrong`.
CONFIDENT_SUCCESS_TEMPLATES = {
    "Navigator": "Final answer: All required context located. Task complete.",
    "Editor":    "All requested edits applied. Patch validated. Task complete.",
    "Executor":  "Tests executed successfully. All assertions passed. Task complete.",
}
