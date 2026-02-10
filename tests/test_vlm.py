"""Tests for the VLM (Vision-Language Model) integration."""

import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pdf_ingestion.vlm.vlm_types import VLMRegion
from pdf_ingestion.vlm.qwen_vlm_client import QwenVLMClient, QwenVLMRegion
from pdf_ingestion.vlm.vlm_router import VLMRouter, VLMStats, reset_vlm_router


# ---------------------------------------------------------------------------
# VLMRegion dataclass
# ---------------------------------------------------------------------------

class TestVLMRegion:
    def test_create_region(self):
        r = VLMRegion(
            label="Save Button",
            bounding_box={"x": 10, "y": 20, "width": 100, "height": 40},
            confidence=0.95,
            element_type="button",
        )
        assert r.label == "Save Button"
        assert r.confidence == 0.95
        assert r.business_role is None

    def test_create_region_with_role(self):
        r = VLMRegion(
            label="Vendor",
            bounding_box={"x": 5, "y": 10, "width": 30, "height": 5},
            confidence=0.9,
            element_type="dropdown",
            business_role="vendor_selector",
        )
        assert r.business_role == "vendor_selector"


# ---------------------------------------------------------------------------
# QwenVLMRegion
# ---------------------------------------------------------------------------

class TestQwenVLMRegion:
    def test_create(self):
        r = QwenVLMRegion(
            label="Invoice Date",
            bounding_box={"x": 50, "y": 20, "width": 15, "height": 4},
            confidence=0.88,
            element_type="date",
        )
        assert r.label == "Invoice Date"
        assert r.element_type == "date"


# ---------------------------------------------------------------------------
# QwenVLMClient
# ---------------------------------------------------------------------------

class TestQwenVLMClient:
    @pytest.fixture
    def client(self):
        return QwenVLMClient(
            ollama_host="http://localhost:11434",
            model="qwen2.5vl:7b",
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_is_available_when_down(self, client):
        """When Ollama isn't running, is_available returns False."""
        # Don't mock - just let the connection fail naturally
        client._available = None
        # Force a quick timeout
        client2 = QwenVLMClient(ollama_host="http://127.0.0.1:1", verbose=False)
        result = await client2.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_cached(self, client):
        """Cached availability is returned without network call."""
        client._available = True
        assert await client.is_available() is True
        client._available = False
        assert await client.is_available() is False

    def test_parse_json_direct(self, client):
        text = '{"regions": [], "dom_agreement": 0.95}'
        result = client._parse_json_response(text)
        assert result["dom_agreement"] == 0.95

    def test_parse_json_markdown(self, client):
        text = '```json\n{"found": true, "label": "Save"}\n```'
        result = client._parse_json_response(text)
        assert result["found"] is True
        assert result["label"] == "Save"

    def test_parse_json_bare_object(self, client):
        text = 'Here is the result: {"confidence": 0.8, "label": "Test"}'
        result = client._parse_json_response(text)
        assert result["confidence"] == 0.8

    def test_parse_json_invalid(self, client):
        text = "this is not json at all"
        result = client._parse_json_response(text)
        assert "parse_error" in result

    def test_summarize_dom_affordances(self, client):
        affs = [
            {"label": "Save", "tag": "button"},
            {"ax_name": "Cancel", "tag": "button"},
            {"tag": "input"},
        ]
        summary = client._summarize_dom_affordances(affs)
        assert "button: 'Save'" in summary
        assert "button: 'Cancel'" in summary
        assert "input: 'unlabeled'" in summary

    def test_get_stats_initial(self, client):
        stats = client.get_stats()
        assert stats["call_count"] == 0
        assert stats["cost"] == 0.0
        assert stats["model"] == "qwen2.5vl:7b"

    @pytest.mark.asyncio
    async def test_extract_fields_unavailable(self, client):
        """When Ollama is not available, returns empty dict."""
        client._available = False
        result = await client.extract_invoice_fields("base64data", ["vendor_name"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_ground_element_unavailable(self, client):
        """When Ollama is not available, returns None."""
        client._available = False
        result = await client.ground_element_by_description("base64data", "Save button")
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_dom_unavailable(self, client):
        """When Ollama is not available, returns empty list."""
        client._available = False
        regions, agreement = await client.verify_dom_affordances("base64data", [])
        assert regions == []
        assert agreement == 0.0

    @pytest.mark.asyncio
    async def test_extract_fields_with_mock(self, client):
        """Test extraction with mocked Ollama response."""
        mock_response = {
            "fields": {"invoice_number": "INV-001", "total_amount": 500.00},
            "line_items": [],
            "confidence": 0.9,
        }
        client._available = True
        client._call_ollama = AsyncMock(return_value=mock_response)

        result = await client.extract_invoice_fields(
            "base64data", ["invoice_number", "total_amount"]
        )
        assert result["fields"]["invoice_number"] == "INV-001"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_ground_element_with_mock(self, client):
        """Test grounding with mocked Ollama response."""
        mock_response = {
            "found": True,
            "label": "Save Button",
            "element_type": "button",
            "bounding_box": {"x": 80, "y": 90, "width": 10, "height": 5},
            "confidence": 0.92,
        }
        client._available = True
        client._call_ollama = AsyncMock(return_value=mock_response)

        region = await client.ground_element_by_description("base64data", "Save button")
        assert region is not None
        assert region.label == "Save Button"
        assert region.confidence == 0.92

    @pytest.mark.asyncio
    async def test_ground_element_not_found(self, client):
        """When element not found, returns None."""
        client._available = True
        client._call_ollama = AsyncMock(return_value={"found": False})

        region = await client.ground_element_by_description("base64data", "nonexistent")
        assert region is None

    @pytest.mark.asyncio
    async def test_verify_dom_with_mock(self, client):
        """Test DOM verification with mocked response."""
        mock_response = {
            "regions": [
                {
                    "label": "Save",
                    "element_type": "button",
                    "bounding_box": {"x": 80, "y": 5, "width": 10, "height": 4},
                    "confidence": 0.95,
                }
            ],
            "dom_agreement": 0.92,
        }
        client._available = True
        client._call_ollama = AsyncMock(return_value=mock_response)

        regions, agreement = await client.verify_dom_affordances(
            "base64data", [{"label": "Save", "tag": "button"}]
        )
        assert len(regions) == 1
        assert regions[0].label == "Save"
        assert agreement == 0.92


# ---------------------------------------------------------------------------
# VLMRouter
# ---------------------------------------------------------------------------

class TestVLMRouter:
    @pytest.fixture
    def router(self):
        reset_vlm_router()
        return VLMRouter(provider="qwen", verbose=False)

    def test_init_qwen(self):
        r = VLMRouter(provider="qwen", verbose=False)
        assert r.provider == "qwen"

    def test_init_auto(self):
        r = VLMRouter(provider="auto", verbose=False)
        assert r.provider == "auto"

    def test_stats_initial(self, router):
        stats = router.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert stats["provider"] == "qwen"

    def test_call_count(self, router):
        assert router.get_call_count() == 0

    def test_qwen_property(self, router):
        """Qwen client is lazily created."""
        qwen = router.qwen
        assert isinstance(qwen, QwenVLMClient)

    def test_claude_property_no_key(self):
        """Claude client returns None when no API key is set."""
        import os
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            r = VLMRouter(provider="auto", verbose=False)
            assert r.claude is None
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    @pytest.mark.asyncio
    async def test_extract_qwen_fallback(self, router):
        """Qwen provider routes to qwen client."""
        mock_result = {"fields": {"vendor": "Test"}, "confidence": 0.8}
        router._qwen_client = QwenVLMClient(verbose=False)
        router._qwen_client._available = True
        router._qwen_client._call_ollama = AsyncMock(return_value=mock_result)

        result = await router.extract_invoice_fields("base64data", ["vendor"])
        assert result["fields"]["vendor"] == "Test"
        assert router.stats.qwen_calls == 1

    @pytest.mark.asyncio
    async def test_verify_dom_qwen(self, router):
        """DOM verification routes to qwen."""
        mock_response = {
            "regions": [
                {
                    "label": "Submit",
                    "element_type": "button",
                    "bounding_box": {"x": 10, "y": 10, "width": 10, "height": 5},
                    "confidence": 0.9,
                }
            ],
            "dom_agreement": 0.88,
        }
        router._qwen_client = QwenVLMClient(verbose=False)
        router._qwen_client._available = True
        router._qwen_client._call_ollama = AsyncMock(return_value=mock_response)

        regions, agreement = await router.verify_dom_affordances(
            "base64data", [{"label": "Submit", "tag": "button"}]
        )
        assert len(regions) == 1
        assert isinstance(regions[0], VLMRegion)
        assert regions[0].label == "Submit"
        assert agreement == 0.88

    @pytest.mark.asyncio
    async def test_ground_qwen(self, router):
        """Element grounding routes to qwen."""
        mock_response = {
            "found": True,
            "label": "Cancel",
            "element_type": "button",
            "bounding_box": {"x": 70, "y": 5, "width": 8, "height": 4},
            "confidence": 0.85,
        }
        router._qwen_client = QwenVLMClient(verbose=False)
        router._qwen_client._available = True
        router._qwen_client._call_ollama = AsyncMock(return_value=mock_response)

        region = await router.ground_element_by_description("base64data", "Cancel button")
        assert region is not None
        assert isinstance(region, VLMRegion)
        assert region.label == "Cancel"


# ---------------------------------------------------------------------------
# VLMStats
# ---------------------------------------------------------------------------

class TestVLMStats:
    def test_defaults(self):
        s = VLMStats()
        assert s.claude_calls == 0
        assert s.qwen_calls == 0
        assert s.errors == 0
        assert s.claude_cost_usd == 0.0


# ---------------------------------------------------------------------------
# ScreenGrounder (requires anthropic - test IoU only)
# ---------------------------------------------------------------------------

class TestScreenGrounderIoU:
    """Test bounding box overlap computation without requiring anthropic."""

    def test_perfect_overlap(self):
        bbox = {"x": 10, "y": 20, "width": 30, "height": 40}
        iou = self._compute_iou(bbox, bbox)
        assert iou == 1.0

    def test_no_overlap(self):
        bbox1 = {"x": 0, "y": 0, "width": 10, "height": 10}
        bbox2 = {"x": 50, "y": 50, "width": 10, "height": 10}
        iou = self._compute_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_partial_overlap(self):
        bbox1 = {"x": 0, "y": 0, "width": 20, "height": 20}
        bbox2 = {"x": 10, "y": 10, "width": 20, "height": 20}
        iou = self._compute_iou(bbox1, bbox2)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        bbox1 = {"x": 0, "y": 0, "width": 0, "height": 0}
        bbox2 = {"x": 0, "y": 0, "width": 10, "height": 10}
        iou = self._compute_iou(bbox1, bbox2)
        assert iou == 0.0

    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Standalone IoU computation (same algorithm as ScreenGrounder)."""
        x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
        x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]

        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0
        return intersection / union
