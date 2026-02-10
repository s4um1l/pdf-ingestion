"""
VLM (Vision-Language Model) integration for invoice/document understanding.

Provides a unified interface to route VLM requests between providers:
- Claude (Anthropic API) - best quality, costs money
- Qwen-VL via Ollama - free, runs locally
- Auto mode - tries Ollama first, falls back to Claude

Usage:
    from pdf_ingestion.vlm import VLMRouter, VLMRegion

    router = VLMRouter(provider="auto")
    fields = await router.extract_invoice_fields(screenshot_b64, ["vendor_name", "total"])
"""

from pdf_ingestion.vlm.vlm_types import VLMRegion
from pdf_ingestion.vlm.qwen_vlm_client import QwenVLMClient, QwenVLMRegion
from pdf_ingestion.vlm.vlm_router import VLMRouter, VLMStats, get_vlm_router, reset_vlm_router

# Claude client is optional (requires anthropic package)
try:
    from pdf_ingestion.vlm.claude_vlm_client import ClaudeVLMClient, ScreenGrounder
except ImportError:
    ClaudeVLMClient = None
    ScreenGrounder = None

__all__ = [
    "VLMRegion",
    "VLMRouter",
    "VLMStats",
    "get_vlm_router",
    "reset_vlm_router",
    "QwenVLMClient",
    "QwenVLMRegion",
    "ClaudeVLMClient",
    "ScreenGrounder",
]
