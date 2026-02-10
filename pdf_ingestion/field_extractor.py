"""
Field Extractor: Extract structured fields from invoice text using regex patterns.

Provides confidence scoring and F1 evaluation against ground truth.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


@dataclass
class ExtractedField:
    """Single extracted field with provenance."""

    field_name: str
    value: Any
    confidence: float
    method: str  # "regex", "vlm", "manual"
    provenance: str
    raw_match: Optional[str] = None


class FieldExtractor:
    """
    Extract structured fields from invoice text using regex patterns.

    Supports confidence scoring and F1 evaluation.
    """

    PATTERNS = {
        "invoice_number": [
            r"Invoice\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"Bill\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"Reference\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"INV[-]?(\d+)",
        ],
        "invoice_date": [
            r"Invoice\s+Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ],
        "due_date": [
            r"Due\s+Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Payment\s+Due\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ],
        "vendor_name": [
            r"From\s*:?\s*([A-Za-z\s&.,]+)(?=\n|$)",
            r"Bill\s+From\s*:?\s*([A-Za-z\s&.,]+)(?=\n|$)",
            r"Vendor\s*:?\s*([A-Za-z\s&.,]+)(?=\n|$)",
        ],
        "total_amount": [
            r"Total\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"Amount\s+Due\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"Balance\s*:?\s*\$?\s*([\d,]+\.\d{2})",
        ],
        "subtotal": [
            r"Subtotal\s*:?\s*\$?\s*([\d,]+\.\d{2})",
        ],
        "tax": [
            r"Tax\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"Sales\s+Tax\s*:?\s*\$?\s*([\d,]+\.\d{2})",
        ],
        "po_number": [
            r"PO\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"Purchase\s+Order\s*:?\s*([A-Z0-9-]+)",
        ],
    }

    def __init__(self):
        self.extraction_count = 0

    def extract_fields(
        self,
        text: str,
        required_fields: Optional[List[str]] = None,
    ) -> Dict[str, ExtractedField]:
        """
        Extract all fields from invoice text.

        Args:
            text: Raw invoice text
            required_fields: Fields that must be extracted
        """
        self.extraction_count += 1
        extracted = {}

        if required_fields is None:
            required_fields = [
                "invoice_number",
                "vendor_name",
                "total_amount",
                "invoice_date",
            ]

        for field_name in required_fields:
            regex_result = self._extract_with_regex(text, field_name)
            if regex_result:
                extracted[field_name] = regex_result

        return extracted

    def _extract_with_regex(
        self, text: str, field_name: str
    ) -> Optional[ExtractedField]:
        """Extract field using regex patterns."""
        if field_name not in self.PATTERNS:
            return None

        patterns = self.PATTERNS[field_name]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                value = self._postprocess_value(field_name, value)
                confidence = self._compute_regex_confidence(pattern, field_name, value)

                return ExtractedField(
                    field_name=field_name,
                    value=value,
                    confidence=confidence,
                    method="regex",
                    provenance=f"regex:{pattern[:30]}",
                    raw_match=match.group(0),
                )

        return None

    def _postprocess_value(self, field_name: str, value: str) -> Any:
        """Clean and type-cast extracted value."""
        value = value.strip()

        if "amount" in field_name or field_name in ["subtotal", "tax", "total"]:
            value = value.replace(",", "")
            try:
                return float(value)
            except ValueError:
                return value

        if "date" in field_name:
            return self._parse_date(value)

        return re.sub(r"\s+", " ", value)

    def _parse_date(self, date_str: str) -> str:
        """Parse and standardize date format to ISO."""
        formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%m/%d/%y",
            "%m-%d-%y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return date_str

    def _compute_regex_confidence(
        self, pattern: str, field_name: str, value: Any
    ) -> float:
        """Compute confidence score for regex match."""
        confidence = 0.5

        if field_name.replace("_", " ") in pattern.lower():
            confidence += 0.2

        if self._validate_value(field_name, value):
            confidence += 0.2

        if len(pattern) > 30:
            confidence += 0.1

        return min(confidence, 1.0)

    def _validate_value(self, field_name: str, value: Any) -> bool:
        """Basic validation of extracted value."""
        if not value:
            return False

        if field_name == "invoice_number":
            return bool(re.match(r"^[A-Z0-9-]+$", str(value), re.IGNORECASE))

        if "amount" in field_name or field_name in ["subtotal", "tax", "total"]:
            try:
                return float(value) > 0
            except (ValueError, TypeError):
                return False

        if "date" in field_name:
            return bool(re.match(r"\d{4}-\d{2}-\d{2}", str(value)))

        if field_name == "vendor_name":
            return 2 <= len(str(value)) <= 100

        return True

    def validate_extraction(
        self, extracted: Dict[str, ExtractedField]
    ) -> Dict[str, bool]:
        """Validate extraction results."""
        validation = {}
        for field_name, field in extracted.items():
            validation[field_name] = (
                field.confidence >= 0.5
                and self._validate_value(field_name, field.value)
            )
        return validation

    def compute_extraction_f1(
        self,
        extracted: Dict[str, ExtractedField],
        ground_truth: Dict[str, Any],
    ) -> Tuple[float, Dict]:
        """
        Compute F1 score against ground truth.

        Returns:
            Tuple of (f1_score, per_field_details)
        """
        if not ground_truth:
            return 0.0, {}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        details = {}

        for field_name in ground_truth.keys():
            gt_value = ground_truth[field_name]
            if field_name in extracted:
                extracted_value = extracted[field_name].value
                if self._values_match(gt_value, extracted_value):
                    true_positives += 1
                    details[field_name] = "TP"
                else:
                    false_positives += 1
                    details[field_name] = (
                        f"FP (got: {extracted_value}, expected: {gt_value})"
                    )
            else:
                false_negatives += 1
                details[field_name] = "FN (not extracted)"

        for field_name in extracted.keys():
            if field_name not in ground_truth:
                false_positives += 1
                details[field_name] = "FP (extra field)"

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return f1, details

    def _values_match(self, gt_value: Any, extracted_value: Any) -> bool:
        """Check if ground truth and extracted values match."""
        gt_str = str(gt_value).strip().lower()
        ext_str = str(extracted_value).strip().lower()

        if gt_str == ext_str:
            return True

        if isinstance(gt_value, str) and isinstance(extracted_value, str):
            gt_clean = re.sub(r"[^a-z0-9]", "", gt_str)
            ext_clean = re.sub(r"[^a-z0-9]", "", ext_str)
            return gt_clean == ext_clean

        try:
            return abs(float(gt_value) - float(extracted_value)) < 0.01
        except (ValueError, TypeError):
            pass

        return False


class InvoiceSchema(BaseModel):
    """Pydantic model for quick invoice validation."""

    invoice_number: str = Field(..., description="Unique invoice identifier")
    vendor_name: str = Field(..., description="Name of vendor/supplier")
    invoice_date: str = Field(..., description="Invoice issue date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(None, description="Payment due date (YYYY-MM-DD)")
    total_amount: float = Field(..., gt=0, description="Total amount due")
    subtotal: Optional[float] = Field(None, ge=0, description="Subtotal before tax")
    tax: Optional[float] = Field(None, ge=0, description="Tax amount")
    po_number: Optional[str] = Field(None, description="Purchase order number")

    class Config:
        json_schema_extra = {
            "example": {
                "invoice_number": "INV-12345",
                "vendor_name": "Acme Corp",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15",
                "total_amount": 1234.56,
                "subtotal": 1100.00,
                "tax": 134.56,
                "po_number": "PO-67890",
            }
        }
