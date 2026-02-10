"""Shared data types for VLM modules."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class VLMRegion:
    """Detected UI region from a VLM provider."""

    label: str
    bounding_box: Dict[str, float]  # {x, y, width, height} as percentages
    confidence: float
    element_type: str  # "button", "input", "link", "dropdown", etc.
    business_role: Optional[str] = None  # "bill_number", "vendor_name", etc.
