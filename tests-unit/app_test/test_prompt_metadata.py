"""Unit tests for the pure metadata-envelope helpers in
``app.prompt_metadata``. These cover the two functions that PromptServer
wires into submission (``extract_envelope_from_extra_data``) and into the
send chokepoint (``inject_envelope``).
"""

from __future__ import annotations

from app.prompt_metadata import (
    extract_envelope_from_extra_data,
    inject_envelope,
)


class TestExtractEnvelopeFromExtraData:
    def test_explicit_metadata_dict_is_used_as_is(self):
        extra_data = {"metadata": {"workflow_id": "wf-1", "trace_id": "t-9"}}
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-1",
            "trace_id": "t-9",
        }

    def test_explicit_metadata_takes_precedence_over_extra_pnginfo(self):
        extra_data = {
            "metadata": {"workflow_id": "explicit"},
            "extra_pnginfo": {"workflow": {"id": "fallback"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "explicit"
        }

    def test_falls_back_to_extra_pnginfo_workflow_id(self):
        extra_data = {"extra_pnginfo": {"workflow": {"id": "wf-legacy"}}}
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }

    def test_returns_none_when_no_metadata_and_no_workflow_id(self):
        assert extract_envelope_from_extra_data({}) is None
        assert (
            extract_envelope_from_extra_data({"extra_pnginfo": {"workflow": {}}})
            is None
        )

    def test_rejects_non_string_or_empty_workflow_id(self):
        for bad in ["", 123, None, [], {}]:
            extra_data = {"extra_pnginfo": {"workflow": {"id": bad}}}
            assert extract_envelope_from_extra_data(extra_data) is None

    def test_rejects_non_dict_inputs_at_each_level(self):
        assert extract_envelope_from_extra_data(None) is None
        assert extract_envelope_from_extra_data("not-a-dict") is None
        assert (
            extract_envelope_from_extra_data({"extra_pnginfo": "not-a-dict"})
            is None
        )
        assert (
            extract_envelope_from_extra_data(
                {"extra_pnginfo": {"workflow": "not-a-dict"}}
            )
            is None
        )

    def test_empty_explicit_metadata_falls_through_to_workflow_id(self):
        extra_data = {
            "metadata": {},
            "extra_pnginfo": {"workflow": {"id": "wf-legacy"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }

    def test_returned_envelope_is_copy_not_reference(self):
        original = {"workflow_id": "wf-1"}
        result = extract_envelope_from_extra_data({"metadata": original})
        result["new_key"] = "x"
        assert "new_key" not in original

    def test_non_dict_explicit_metadata_falls_through_to_workflow_id(self):
        extra_data = {
            "metadata": "not-a-dict",
            "extra_pnginfo": {"workflow": {"id": "wf-legacy"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }


class TestInjectEnvelope:
    @staticmethod
    def _lookup(table):
        """Build an envelope_lookup callable backed by a dict."""
        return table.get

    def test_injects_envelope_on_dict_with_known_prompt_id(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        assert inject_envelope({"node": "5", "prompt_id": "p1"}, lookup) == {
            "node": "5",
            "prompt_id": "p1",
            "metadata": {"workflow_id": "wf-1"},
        }

    def test_passthrough_when_prompt_id_not_registered(self):
        lookup = self._lookup({})
        data = {"node": "5", "prompt_id": "unknown"}
        assert inject_envelope(data, lookup) == data

    def test_passthrough_when_payload_lacks_prompt_id(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        data = {"status": "ok"}
        assert inject_envelope(data, lookup) is data

    def test_passthrough_when_payload_already_has_metadata(self):
        """If a caller has already set a ``metadata`` field (e.g. for
        opt-out or pre-augmented payloads), the function must not
        overwrite it."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-injected"}})
        data = {"prompt_id": "p1", "metadata": {"workflow_id": "wf-caller"}}
        result = inject_envelope(data, lookup)
        assert result is data
        assert result["metadata"] == {"workflow_id": "wf-caller"}

    def test_does_not_mutate_input_dict(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        original = {"node": "5", "prompt_id": "p1"}
        inject_envelope(original, lookup)
        assert "metadata" not in original

    def test_injects_into_inner_dict_of_preview_metadata_tuple(self):
        """``PREVIEW_IMAGE_WITH_METADATA`` payloads arrive as
        ``(preview_image, metadata_dict)``; the inner dict is the only
        place the envelope can attach."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        preview_image = ("PNG", object(), 256)
        inner = {"node_id": "5", "prompt_id": "p1"}
        result = inject_envelope((preview_image, inner), lookup)
        assert isinstance(result, tuple)
        assert result[0] is preview_image
        assert result[1] == {
            "node_id": "5",
            "prompt_id": "p1",
            "metadata": {"workflow_id": "wf-1"},
        }
        assert "metadata" not in inner

    def test_preview_tuple_passthrough_when_no_envelope_registered(self):
        lookup = self._lookup({})
        preview_image = ("PNG", object(), 256)
        inner = {"node_id": "5", "prompt_id": "unknown"}
        result = inject_envelope((preview_image, inner), lookup)
        assert result == (preview_image, inner)

    def test_preview_tuple_passthrough_when_inner_already_has_metadata(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-injected"}})
        preview_image = ("PNG", object(), 256)
        inner = {"node_id": "5", "prompt_id": "p1", "metadata": {"x": 1}}
        result = inject_envelope((preview_image, inner), lookup)
        assert result == (preview_image, inner)

    def test_non_dict_non_tuple_payloads_passthrough(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        assert inject_envelope(b"raw-bytes", lookup) == b"raw-bytes"
        assert inject_envelope(None, lookup) is None
        assert inject_envelope(42, lookup) == 42

    def test_tuple_of_wrong_arity_passthrough(self):
        """Only the 2-tuple ``(preview, metadata_dict)`` shape is special-
        cased. Other tuples must not be touched."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        triple = (1, {"prompt_id": "p1"}, 3)
        assert inject_envelope(triple, lookup) is triple

    def test_envelope_lookup_called_at_send_time(self):
        """The lookup runs each time the function is called, so a producer
        and consumer that share a backing dict observe the current value."""
        store = {"p1": {"workflow_id": "wf-1"}}
        first = inject_envelope({"prompt_id": "p1"}, store.get)
        store["p1"] = {"workflow_id": "wf-2"}
        second = inject_envelope({"prompt_id": "p1"}, store.get)
        assert first["metadata"] == {"workflow_id": "wf-1"}
        assert second["metadata"] == {"workflow_id": "wf-2"}
