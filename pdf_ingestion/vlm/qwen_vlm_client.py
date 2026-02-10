"""
Qwen2.5-VL Client: Local VLM via Ollama for cost-effective visual verification.

Uses Ollama to run Qwen2.5-VL locally - completely free, no API keys needed.

Setup:
    brew install ollama && ollama serve
    ollama pull qwen2.5vl:7b

Models:
    - qwen2.5vl:3b  (3.2GB) - edge/lightweight
    - qwen2.5vl:7b  (6GB)   - recommended
    - qwen2.5vl:32b (21GB)  - high quality
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import base64
import json
import re
import time


@dataclass
class QwenVLMRegion:
    """Detected UI region from Qwen-VL."""

    label: str
    bounding_box: Dict[str, float]  # {x, y, width, height}
    confidence: float
    element_type: str  # "button", "input", "link", etc.
    business_role: Optional[str] = None


class QwenVLMClient:
    """
    Qwen2.5-VL via Ollama for cost-effective visual verification.

    Cost: $0 (local)
    Speed: 3-10s per image (depending on hardware)

    Use for:
    - DOM verification (agreement check)
    - Element grounding by description
    - Invoice field extraction from screenshots
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
        timeout: float = 120.0,
        verbose: bool = True,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        self.call_count = 0
        self.total_time_ms = 0.0
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Ollama + Qwen-VL is running and model is pulled."""
        if self._available is not None:
            return self._available

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code != 200:
                    self._available = False
                    return False

                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                self._available = any(
                    self.model in name or name.startswith("qwen2-vl") or name.startswith("qwen2.5vl")
                    for name in model_names
                )

                if not self._available and self.verbose:
                    print(f"Qwen-VL model '{self.model}' not found in Ollama")
                    print(f"  Available: {model_names}")
                    print(f"  Run: ollama pull {self.model}")

                return self._available

        except Exception as e:
            if self.verbose:
                print(f"Ollama not available: {e}")
            self._available = False
            return False

    async def verify_dom_affordances(
        self,
        screenshot_b64: str,
        dom_affordances: List[Dict],
        focus_region: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[QwenVLMRegion], float]:
        """
        Verify DOM affordances using Qwen-VL.

        Returns:
            (vlm_regions, agreement_score)
        """
        if not await self.is_available():
            return [], 0.0

        start_time = time.time()
        dom_summary = self._summarize_dom_affordances(dom_affordances[:15])

        prompt = f"""Look at this screenshot and verify the UI elements.

DOM extracted these elements:
{dom_summary}

Tasks:
1. List all interactive elements you see (buttons, inputs, links)
2. For each: label, type, approximate position (x%, y%, width%, height%)
3. Rate how well DOM matches what you see (0.0 to 1.0)

Return JSON only:
{{"regions": [{{"label": "...", "element_type": "button", "bounding_box": {{"x": 10, "y": 20, "width": 15, "height": 5}}, "confidence": 0.9}}], "dom_agreement": 0.95}}"""

        try:
            result = await self._call_ollama(screenshot_b64, prompt)

            vlm_regions = [
                QwenVLMRegion(
                    label=r.get("label", ""),
                    bounding_box=r.get("bounding_box", {}),
                    confidence=r.get("confidence", 0.5),
                    element_type=r.get("element_type", "unknown"),
                    business_role=r.get("business_role"),
                )
                for r in result.get("regions", [])
            ]

            agreement = result.get("dom_agreement", 0.0)
            elapsed = (time.time() - start_time) * 1000
            self.total_time_ms += elapsed

            if self.verbose:
                print(f"  Qwen-VL: {len(vlm_regions)} regions, {agreement:.0%} agreement ({elapsed:.0f}ms)")

            return vlm_regions, agreement

        except Exception as e:
            if self.verbose:
                print(f"  Qwen-VL error: {e}")
            return [], 0.0

    async def ground_element_by_description(
        self,
        screenshot_b64: str,
        element_description: str,
    ) -> Optional[QwenVLMRegion]:
        """Find element by natural language description."""
        if not await self.is_available():
            return None

        prompt = f"""Find this UI element: "{element_description}"

Provide exact location as percentages (0-100).

Return JSON only:
{{"found": true, "label": "...", "element_type": "button", "bounding_box": {{"x": 10, "y": 20, "width": 15, "height": 5}}, "confidence": 0.9}}

If not found: {{"found": false}}"""

        try:
            result = await self._call_ollama(screenshot_b64, prompt)

            if not result.get("found"):
                return None

            return QwenVLMRegion(
                label=result.get("label", element_description),
                bounding_box=result.get("bounding_box", {}),
                confidence=result.get("confidence", 0.5),
                element_type=result.get("element_type", "unknown"),
            )

        except Exception as e:
            if self.verbose:
                print(f"  Qwen-VL grounding error: {e}")
            return None

    async def extract_invoice_fields(
        self,
        screenshot_b64: str,
        expected_fields: List[str],
    ) -> Dict[str, Any]:
        """
        Extract invoice fields from a screenshot image.

        For PDF invoices, consider using DoclingExtractor instead (more accurate).
        This method is best for extracting from screenshots of filled forms.
        """
        if not await self.is_available():
            return {}

        fields_str = ", ".join(expected_fields)

        prompt = f"""Extract these fields from this invoice/bill image:
{fields_str}

IMPORTANT RULES:
1. Only extract fields that are EXPLICITLY visible on the invoice
2. Use null for any field NOT present (do NOT copy values between fields)
3. po_number is DIFFERENT from invoice_number - only include if explicitly labeled as "PO", "Purchase Order", or similar
4. Extract ALL line items from the table

Return JSON with:
1. "fields" - object with field values (use null if not found)
2. "line_items" - array of items from the table with: description, quantity, unit_price, amount

Example:
{{
  "fields": {{"invoice_number": "INV-12345", "vendor_name": "Acme Corp", "total_amount": 1234.56, "po_number": null}},
  "line_items": [
    {{"description": "Widget A", "quantity": 2, "unit_price": 50.00, "amount": 100.00}}
  ],
  "confidence": 0.85
}}"""

        try:
            return await self._call_ollama(screenshot_b64, prompt)
        except Exception as e:
            if self.verbose:
                print(f"  Qwen-VL extraction error: {e}")
            return {}

    async def verify_screen(self, screenshot_bytes: bytes) -> Dict:
        """
        Verify UI elements from raw screenshot bytes.

        Returns dict with 'regions' list of detected form fields.
        """
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = """Analyze this bill/invoice form and identify KEY FORM FIELDS only.

Focus on:
- Text inputs (where you type)
- Dropdowns (Vendor, Customer)
- Date fields
- Amount fields
- Reference numbers

IGNORE buttons. Only data entry fields.

For each, provide: label, type, bounding box (x, y, width, height as %).

Return JSON:
{"regions": [{"label": "Vendor", "element_type": "dropdown", "bounding_box": {"x": 10, "y": 20, "width": 25, "height": 4}, "confidence": 0.95}]}

Limit to top 10 fields."""

        try:
            result = await self._call_ollama(screenshot_b64, prompt)
            return result
        except Exception as e:
            return {"regions": [], "error": str(e)}

    async def _call_ollama(self, image_b64: str, prompt: str) -> Dict:
        """Make API call to Ollama."""
        import httpx

        self.call_count += 1

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "format": "json",
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.status_code} - {response.text}")

            result = response.json()
            response_text = result.get("response", "{}")
            return self._parse_json_response(response_text)

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from VLM response (handles markdown wrappers)."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```json\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {"parse_error": text[:200]}

    def _summarize_dom_affordances(self, affordances: List[Dict]) -> str:
        """Summarize DOM affordances for prompt."""
        lines = []
        for aff in affordances:
            label = aff.get("label") or aff.get("ax_name") or "unlabeled"
            elem_type = aff.get("tag", "unknown")
            lines.append(f"- {elem_type}: '{label}'")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.total_time_ms / max(self.call_count, 1),
            "model": self.model,
            "cost": 0.0,
        }
