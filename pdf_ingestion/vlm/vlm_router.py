"""
VLM Router: Unified interface for VLM providers.

Routes requests based on provider setting:
- "claude": Always use Claude (costs money, best quality)
- "qwen": Always use Qwen-VL via Ollama (free, local)
- "auto": Try Qwen first, fall back to Claude

Usage:
    from pdf_ingestion.vlm import VLMRouter

    router = VLMRouter(provider="auto")
    fields = await router.extract_invoice_fields(screenshot_b64, ["vendor_name", "total"])
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import base64
import asyncio
import os
import tempfile

from pdf_ingestion.vlm.vlm_types import VLMRegion
from pdf_ingestion.vlm.qwen_vlm_client import QwenVLMClient, QwenVLMRegion


@dataclass
class VLMStats:
    """Track VLM usage across providers."""

    claude_calls: int = 0
    qwen_calls: int = 0
    claude_cost_usd: float = 0.0
    qwen_cost_usd: float = 0.0
    total_time_ms: float = 0
    errors: int = 0


class VLMRouter:
    """
    Routes VLM requests to Claude or Qwen-VL based on configuration.

    Strategy:
    - "claude": Direct to Claude (reliable, costly)
    - "qwen": Direct to Qwen-VL via Ollama (free, local)
    - "auto": Try Qwen -> Claude (best effort)
    """

    CLAUDE_COST_PER_CALL = 0.01

    def __init__(
        self,
        provider: str = "auto",
        claude_model: str = "claude-sonnet-4-20250514",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5vl:7b",
        verbose: bool = True,
    ):
        self.provider = provider.lower()
        self.verbose = verbose
        self.stats = VLMStats()

        self._claude_client = None
        self._qwen_client: Optional[QwenVLMClient] = None

        self._claude_model = claude_model
        self._ollama_host = ollama_host
        self._ollama_model = ollama_model

        if self.verbose:
            provider_info = {
                "claude": "Claude (API, best quality)",
                "qwen": "Qwen-VL (local, free)",
                "auto": "Auto (Qwen -> Claude fallback)",
            }.get(self.provider, self.provider)
            print(f"VLM Router: {provider_info}")

    @property
    def claude(self):
        """Get or create Claude client (returns None if unavailable)."""
        if self._claude_client is None:
            try:
                from pdf_ingestion.vlm.claude_vlm_client import ClaudeVLMClient

                self._claude_client = ClaudeVLMClient(
                    model=self._claude_model, verbose=self.verbose
                )
            except (ImportError, ValueError) as e:
                if self.verbose:
                    print(f"Claude VLM not available: {e}")
                return None
        return self._claude_client

    @property
    def qwen(self) -> QwenVLMClient:
        """Get or create Qwen client."""
        if self._qwen_client is None:
            self._qwen_client = QwenVLMClient(
                ollama_host=self._ollama_host,
                model=self._ollama_model,
                verbose=self.verbose,
            )
        return self._qwen_client

    def _to_b64_and_path(
        self, screenshot_path_or_b64: str
    ) -> Tuple[str, Optional[str]]:
        """Convert input to (base64, optional_path) tuple."""
        if screenshot_path_or_b64.startswith("/") or screenshot_path_or_b64.endswith(
            ".png"
        ):
            with open(screenshot_path_or_b64, "rb") as f:
                screenshot_b64 = base64.b64encode(f.read()).decode("utf-8")
            return screenshot_b64, screenshot_path_or_b64
        return screenshot_path_or_b64, None

    def _ensure_path(self, screenshot_b64: str, screenshot_path: Optional[str]) -> str:
        """Ensure we have a file path (create temp file if needed)."""
        if screenshot_path:
            return screenshot_path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(screenshot_b64))
            return f.name

    async def verify_dom_affordances(
        self,
        screenshot_path_or_b64: str,
        dom_affordances: List[Dict],
        focus_region: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[VLMRegion], float]:
        """Verify DOM affordances - unified interface."""
        screenshot_b64, screenshot_path = self._to_b64_and_path(
            screenshot_path_or_b64
        )

        if self.provider == "claude":
            if self.claude is None:
                return [], 0.0
            path = self._ensure_path(screenshot_b64, screenshot_path)
            return await self._verify_with_claude(path, dom_affordances, focus_region)

        elif self.provider == "qwen":
            return await self._verify_with_qwen(
                screenshot_b64, dom_affordances, focus_region
            )

        else:  # auto
            if await self.qwen.is_available():
                try:
                    result = await self._verify_with_qwen(
                        screenshot_b64, dom_affordances, focus_region
                    )
                    if result[1] > 0:
                        return result
                except Exception as e:
                    if self.verbose:
                        print(f"  Qwen failed: {e}")
                    self.stats.errors += 1

            # Fallback to Claude
            if self.claude is None:
                return [], 0.0
            path = self._ensure_path(screenshot_b64, screenshot_path)
            return await self._verify_with_claude(path, dom_affordances, focus_region)

    async def _verify_with_claude(
        self,
        screenshot_path: str,
        dom_affordances: List[Dict],
        focus_region: Optional[Dict[str, float]],
    ) -> Tuple[List[VLMRegion], float]:
        if self.claude is None:
            return [], 0.0

        self.stats.claude_calls += 1
        self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL

        loop = asyncio.get_event_loop()
        regions, agreement = await loop.run_in_executor(
            None,
            lambda: self.claude.verify_dom_affordances(
                screenshot_path, dom_affordances, focus_region
            ),
        )
        return regions, agreement

    async def _verify_with_qwen(
        self,
        screenshot_b64: str,
        dom_affordances: List[Dict],
        focus_region: Optional[Dict[str, float]],
    ) -> Tuple[List[VLMRegion], float]:
        self.stats.qwen_calls += 1

        qwen_regions, agreement = await self.qwen.verify_dom_affordances(
            screenshot_b64, dom_affordances, focus_region
        )

        vlm_regions = [
            VLMRegion(
                label=r.label,
                bounding_box=r.bounding_box,
                confidence=r.confidence,
                element_type=r.element_type,
                business_role=r.business_role,
            )
            for r in qwen_regions
        ]

        return vlm_regions, agreement

    async def ground_element_by_description(
        self,
        screenshot_path_or_b64: str,
        element_description: str,
    ) -> Optional[VLMRegion]:
        """Find element by natural language description."""
        screenshot_b64, screenshot_path = self._to_b64_and_path(
            screenshot_path_or_b64
        )

        if self.provider == "claude":
            if self.claude is None:
                return None
            path = self._ensure_path(screenshot_b64, screenshot_path)
            self.stats.claude_calls += 1
            self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.claude.ground_element_by_description(
                    path, element_description
                ),
            )

        elif self.provider == "qwen":
            self.stats.qwen_calls += 1
            qwen_region = await self.qwen.ground_element_by_description(
                screenshot_b64, element_description
            )
            if qwen_region:
                return VLMRegion(
                    label=qwen_region.label,
                    bounding_box=qwen_region.bounding_box,
                    confidence=qwen_region.confidence,
                    element_type=qwen_region.element_type,
                    business_role=qwen_region.business_role,
                )
            return None

        else:  # auto
            if await self.qwen.is_available():
                try:
                    self.stats.qwen_calls += 1
                    qwen_region = await self.qwen.ground_element_by_description(
                        screenshot_b64, element_description
                    )
                    if qwen_region:
                        return VLMRegion(
                            label=qwen_region.label,
                            bounding_box=qwen_region.bounding_box,
                            confidence=qwen_region.confidence,
                            element_type=qwen_region.element_type,
                            business_role=qwen_region.business_role,
                        )
                except Exception:
                    self.stats.errors += 1

            # Fallback to Claude
            if self.claude is None:
                return None

            path = self._ensure_path(screenshot_b64, screenshot_path)
            self.stats.claude_calls += 1
            self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.claude.ground_element_by_description(
                    path, element_description
                ),
            )

    async def extract_invoice_fields(
        self,
        screenshot_path_or_b64: str,
        expected_fields: List[str],
    ) -> Dict[str, Any]:
        """
        Extract invoice fields from screenshot.

        Note: For PDF invoices, use DoclingExtractor instead (more accurate).
        This is for extracting from screenshots of filled forms.
        """
        screenshot_b64, screenshot_path = self._to_b64_and_path(
            screenshot_path_or_b64
        )

        if self.provider == "qwen":
            self.stats.qwen_calls += 1
            return await self.qwen.extract_invoice_fields(
                screenshot_b64, expected_fields
            )

        elif self.provider == "claude" or self.provider == "auto":
            claude_client = self.claude
            if claude_client is not None:
                path = self._ensure_path(screenshot_b64, screenshot_path)
                self.stats.claude_calls += 1
                self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: claude_client.extract_invoice_fields(
                        path, expected_fields
                    ),
                )

            # Claude not available, fallback to Qwen
            if self.provider == "auto":
                self.stats.qwen_calls += 1
                return await self.qwen.extract_invoice_fields(
                    screenshot_b64, expected_fields
                )

        return {}

    async def verify_screen(self, screenshot_bytes: bytes) -> Dict:
        """Verify UI elements from screenshot bytes."""
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        if self.provider == "qwen":
            self.stats.qwen_calls += 1
            return await self.qwen.verify_screen(screenshot_bytes)

        elif self.provider == "claude":
            if self.claude is None:
                return {"regions": [], "error": "Claude not available"}
            self.stats.claude_calls += 1
            self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL
            return await self.claude.verify_screen(screenshot_bytes)

        else:  # auto - prefer Qwen (faster, free)
            if await self.qwen.is_available():
                try:
                    self.stats.qwen_calls += 1
                    result = await self.qwen.verify_screen(screenshot_bytes)
                    if result.get("regions"):
                        return result
                except Exception:
                    self.stats.errors += 1

            if self.claude is None:
                return {"regions": [], "error": "No VLM provider available"}
            self.stats.claude_calls += 1
            self.stats.claude_cost_usd += self.CLAUDE_COST_PER_CALL
            return await self.claude.verify_screen(screenshot_bytes)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.provider,
            "claude_calls": self.stats.claude_calls,
            "qwen_calls": self.stats.qwen_calls,
            "total_calls": self.stats.claude_calls + self.stats.qwen_calls,
            "claude_cost_usd": round(self.stats.claude_cost_usd, 4),
            "qwen_cost_usd": 0.0,
            "total_cost_usd": round(self.stats.claude_cost_usd, 4),
            "errors": self.stats.errors,
            "cost_savings_usd": round(
                self.stats.qwen_calls * self.CLAUDE_COST_PER_CALL, 4
            ),
        }

    def get_call_count(self) -> int:
        """Total call count across all providers."""
        return self.stats.claude_calls + self.stats.qwen_calls


# Singleton
_vlm_router: Optional[VLMRouter] = None


def get_vlm_router(
    provider: Optional[str] = None,
    claude_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> VLMRouter:
    """
    Get unified VLM router (singleton).

    Reads from environment:
    - VLM_PROVIDER: "claude" | "qwen" | "auto" (default: "auto")
    - OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    - OLLAMA_MODEL: Qwen model name (default: qwen2.5vl:7b)
    - CLAUDE_VLM_MODEL: Claude model (default: claude-sonnet-4-20250514)
    """
    global _vlm_router

    if _vlm_router is None:
        _vlm_router = VLMRouter(
            provider=provider or os.getenv("VLM_PROVIDER", "auto"),
            claude_model=claude_model
            or os.getenv("CLAUDE_VLM_MODEL", "claude-sonnet-4-20250514"),
            ollama_host=ollama_host
            or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=ollama_model
            or os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
        )

    return _vlm_router


def reset_vlm_router():
    """Reset singleton (for testing)."""
    global _vlm_router
    _vlm_router = None
