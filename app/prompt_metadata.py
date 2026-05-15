"""Per-prompt metadata envelope shared between submission and outbound events.

The metadata envelope is an opaque dict (e.g. ``{"workflow_id": ...}``)
attached to a prompt at submission and injected by the server into every
outbound execution event that carries a ``prompt_id``. It lets consumers
scope state by tags they care about (workflow, trace, tenant) without the
execution layer ever needing to know those tags exist.

Two pure functions live here; ``PromptServer`` owns the per-prompt map and
wires them into the submission and send paths.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def extract_envelope_from_extra_data(extra_data: Any) -> Optional[dict]:
    """Pull the per-prompt metadata envelope out of a submitted prompt's
    ``extra_data``.

    Two sources, in order:

    1. Explicit ``extra_data["metadata"]`` dict — preferred path, accepted
       as-is (copied so later mutations on the caller's dict don't leak).
    2. ``extra_data["extra_pnginfo"]["workflow"]["id"]`` — backward-
       compatibility fallback. Frontends that already stamp the workflow
       id into ``extra_pnginfo`` keep working without changes; the
       synthesized envelope is ``{"workflow_id": <id>}``.

    Returns ``None`` when neither source yields a usable envelope.
    """
    if not isinstance(extra_data, dict):
        return None

    metadata = extra_data.get("metadata")
    if isinstance(metadata, dict) and metadata:
        return dict(metadata)

    extra_pnginfo = extra_data.get("extra_pnginfo")
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if isinstance(workflow, dict):
            workflow_id = workflow.get("id")
            if isinstance(workflow_id, str) and workflow_id:
                return {"workflow_id": workflow_id}

    return None


def inject_envelope(
    data: Any,
    envelope_lookup: Callable[[str], Optional[dict]],
) -> Any:
    """Return ``data`` with a per-prompt ``metadata`` envelope attached.

    ``envelope_lookup`` is called with the payload's ``prompt_id`` and is
    expected to return the registered envelope or ``None``. This indirection
    keeps the function pure and avoids depending on any specific storage.

    Two payload shapes are handled:

    - **dict** carrying ``prompt_id``. A shallow copy is returned with a
      ``metadata`` key set to the envelope. The caller's dict is not
      mutated.
    - **(preview_image, metadata_dict) tuple** — the format used by
      ``PREVIEW_IMAGE_WITH_METADATA``. Only the inner dict is augmented;
      the binary preview is passed through by reference.

    The function is a no-op for:

    - payloads without a ``prompt_id``,
    - payloads already declaring their own ``metadata`` field
      (callers can opt out by setting it explicitly),
    - prompts with no registered envelope,
    - any other payload shape (raw bytes, ``None``, etc.).
    """
    def inject(d: dict) -> dict:
        if not isinstance(d, dict) or "metadata" in d:
            return d
        prompt_id = d.get("prompt_id")
        if not prompt_id:
            return d
        envelope = envelope_lookup(prompt_id)
        if envelope is None:
            return d
        return {**d, "metadata": envelope}

    if isinstance(data, dict):
        return inject(data)
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
        injected = inject(data[1])
        if injected is data[1]:
            return data
        return (data[0], injected)
    return data
