"""
Claude VLM Client: Anthropic Claude for high-quality UI understanding.

Requires: pip install anthropic
Requires: ANTHROPIC_API_KEY environment variable

Cost: ~$0.01 per image analysis
Speed: 2-3s per image
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import base64
import json
import os
import re
from pathlib import Path

from pdf_ingestion.vlm.vlm_types import VLMRegion

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeVLMClient:
    """
    Claude-based VLM for UI understanding and invoice extraction.

    Best for:
    - High-quality invoice field extraction
    - Complex reasoning about UI state
    - Fallback when local models fail

    Requires ANTHROPIC_API_KEY env var or pip install anthropic.
    """

    COST_PER_CALL = 0.01

    def __init__(self, model: str = "claude-sonnet-4-20250514", verbose: bool = True):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Either:\n"
                "  1. Set ANTHROPIC_API_KEY environment variable, or\n"
                '  2. Use VLMRouter(provider="qwen") for free local Ollama'
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.call_count = 0
        self.verbose = verbose
        self.last_audit_data: Optional[Dict] = None
        self.audit_history: List[Dict] = []

    def verify_dom_affordances(
        self,
        screenshot_path: str,
        dom_affordances: List[Dict],
        focus_region: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[VLMRegion], float]:
        """
        Verify DOM affordances using Claude vision.

        Returns:
            (vlm_regions, agreement_score)
        """
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        dom_summary = self._summarize_dom_affordances(dom_affordances)

        prompt = f"""Analyze this UI screenshot and verify the interactive elements.

DOM extracted these elements:
{dom_summary}

Your tasks:
1. Identify all interactive UI elements (buttons, inputs, links)
2. For each element, provide:
   - Label/text
   - Type (button/input/link/etc)
   - Bounding box (x, y, width, height) as percentages
   - Confidence (0-1)
3. Flag any elements DOM missed
4. Flag any DOM elements that don't visually exist

Return JSON:
{{
  "regions": [
    {{
      "label": "Save Bill",
      "element_type": "button",
      "bounding_box": {{"x": 10, "y": 20, "width": 100, "height": 40}},
      "confidence": 0.95,
      "business_role": "action_button"
    }}
  ],
  "dom_agreement": 0.97,
  "missing_from_dom": [],
  "false_positives": []
}}"""

        self.call_count += 1

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        raw_response = response.content[0].text
        result = self._parse_vlm_response(raw_response)

        vlm_regions = [
            VLMRegion(
                label=r["label"],
                bounding_box=r["bounding_box"],
                confidence=r["confidence"],
                element_type=r["element_type"],
                business_role=r.get("business_role"),
            )
            for r in result.get("regions", [])
        ]

        agreement = result.get("dom_agreement", 0.0)

        # Store audit data
        from datetime import datetime

        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "screenshot_path": screenshot_path,
            "raw_response": raw_response,
            "parsed_result": result,
            "model_used": self.model,
            "dom_affordances_count": len(dom_affordances),
            "vlm_regions_count": len(vlm_regions),
            "agreement_score": agreement,
        }
        self.last_audit_data = audit_data
        self.audit_history.append(audit_data)

        return vlm_regions, agreement

    def ground_element_by_description(
        self,
        screenshot_path: str,
        element_description: str,
    ) -> Optional[VLMRegion]:
        """Find element by natural language description."""
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""Find this UI element: "{element_description}"

Provide:
1. Exact bounding box (x, y, width, height as percentages)
2. Element type (button/input/link/etc)
3. Visible text/label
4. Confidence (0-1)

Return JSON:
{{
  "found": true,
  "label": "...",
  "element_type": "...",
  "bounding_box": {{"x": ..., "y": ..., "width": ..., "height": ...}},
  "confidence": 0.95
}}

If not found: {{"found": false}}"""

        self.call_count += 1

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        result = self._parse_vlm_response(response.content[0].text)

        if not result.get("found"):
            return None

        return VLMRegion(
            label=result["label"],
            bounding_box=result["bounding_box"],
            confidence=result["confidence"],
            element_type=result["element_type"],
            business_role=None,
        )

    def extract_invoice_fields(
        self,
        screenshot_path: str,
        expected_fields: List[str],
    ) -> Dict[str, Any]:
        """Extract structured data from an invoice screenshot."""
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""Extract these fields from this invoice/bill:
{', '.join(expected_fields)}

Return JSON with exact values. If field not found, use null.

Example:
{{
  "invoice_number": "INV-12345",
  "vendor_name": "Acme Corp",
  "total_amount": 1234.56,
  "invoice_date": "2024-01-15",
  "due_date": "2024-02-15"
}}"""

        self.call_count += 1

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return self._parse_vlm_response(response.content[0].text)

    async def verify_screen(self, screenshot_bytes: bytes) -> Dict:
        """Verify UI elements from raw screenshot bytes."""
        import asyncio

        image_data = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = """Analyze this bill/invoice form and identify the KEY FORM FIELDS.

Focus ONLY on data entry fields:
- Text input fields
- Dropdown/select fields (Vendor, Customer, etc.)
- Date fields (Invoice Date, Due Date, etc.)
- Amount/currency fields (Total, Subtotal, etc.)
- Reference number fields

IGNORE buttons. Only data entry fields.

For each form field:
- Label, Type, Bounding box (x, y, width, height as %), Confidence (0-1)

Return JSON:
{"regions": [{"label": "Vendor", "element_type": "dropdown", "bounding_box": {"x": 10, "y": 20, "width": 25, "height": 4}, "confidence": 0.95}]}

Limit to top 10 fields."""

        self.call_count += 1

        def _call_vlm():
            return self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _call_vlm)
            raw_response = response.content[0].text
            return self._parse_vlm_response(raw_response)
        except Exception as e:
            return {"regions": [], "error": str(e)}

    def get_last_audit_data(self) -> Optional[Dict]:
        """Get the most recent VLM audit data."""
        return self.last_audit_data

    def export_audit_history(self, filepath: str) -> int:
        """Export all VLM audit data to a JSONL file."""
        with open(filepath, "w") as f:
            for audit in self.audit_history:
                f.write(json.dumps(audit) + "\n")
        return len(self.audit_history)

    def _summarize_dom_affordances(self, affordances: List[Dict]) -> str:
        lines = []
        for aff in affordances[:20]:
            label = aff.get("label", "unlabeled")
            elem_type = aff.get("tag", "unknown")
            lines.append(f"- {elem_type}: '{label}'")
        return "\n".join(lines)

    def _parse_vlm_response(self, response_text: str) -> Dict:
        """Parse JSON from VLM response (handles markdown wrappers)."""
        # Try extracting from markdown code block
        json_match = re.search(r"```json\n?(.*?)\n?```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try without newline
        json_match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try bare JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            return {"regions": [], "dom_agreement": 0.0, "parse_error": str(e)}

    def get_call_count(self) -> int:
        return self.call_count


class ScreenGrounder:
    """
    Fuse DOM + VLM for robust element grounding.

    Strategy:
    1. High-confidence DOM elements -> use directly
    2. Low-confidence DOM -> verify with VLM
    3. Missing in DOM -> find with VLM
    """

    def __init__(self, vlm_client: ClaudeVLMClient):
        self.vlm = vlm_client
        self.agreement_threshold = 0.97

    def ground_element(
        self,
        element_id: str,
        screenshot_path: str,
        dom_confidence: float,
        dom_bbox: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, str]:
        """
        Ground element using DOM + VLM fusion.

        Returns:
            (final_bbox, final_confidence, source)
        """
        if dom_confidence >= 0.8:
            return dom_bbox, dom_confidence, "dom"

        vlm_region = self.vlm.ground_element_by_description(
            screenshot_path=screenshot_path,
            element_description=element_id,
        )

        if not vlm_region:
            return dom_bbox, dom_confidence * 0.5, "dom_unverified"

        agreement = self._compute_bbox_overlap(dom_bbox, vlm_region.bounding_box)

        if agreement >= self.agreement_threshold:
            return vlm_region.bounding_box, 0.95, "dom_vlm_verified"
        else:
            return vlm_region.bounding_box, vlm_region.confidence, "vlm_override"

    def _compute_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute IoU (Intersection over Union) for bounding boxes."""
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
