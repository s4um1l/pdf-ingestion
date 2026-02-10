"""
Extraction Service: Multi-strategy field detection wrapping DoclingExtractor.

Features:
- Column role detection for line item tables
- Multi-strategy field extraction (labeled -> pattern -> heuristic)
- Financial calculations with discount/shipping/tax support
- Per-field confidence scores
"""

import asyncio
import base64
import logging
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class ColumnRole(Enum):
    """Role of a column in a line items table."""

    UNKNOWN = "unknown"
    SKU = "sku"
    DESCRIPTION = "description"
    QUANTITY = "quantity"
    UNIT = "unit"
    UNIT_PRICE = "unit_price"
    LINE_TOTAL = "line_total"


class ExtractionStrategy(Enum):
    """How a field was extracted."""

    LABELED = "labeled"
    PATTERN = "pattern"
    HEURISTIC = "heuristic"
    CALCULATED = "calculated"
    MISSING = "missing"


@dataclass
class FieldResult:
    """Result of extracting a single field with confidence tracking."""

    value: Any
    confidence: float
    strategy: ExtractionStrategy
    source: str


# ==============================================================================
# HEADER MAPPINGS FOR COLUMN ROLE DETECTION
# ==============================================================================

HEADER_MAPPINGS = {
    ColumnRole.SKU: [
        "sku",
        "item #",
        "item_id",
        "product code",
        "code",
        "part #",
        "part no",
    ],
    ColumnRole.DESCRIPTION: [
        "description",
        "item",
        "product",
        "service",
        "name",
        "details",
    ],
    ColumnRole.QUANTITY: ["qty", "quantity", "qnty", "units", "count", "qty."],
    ColumnRole.UNIT: ["unit", "uom", "measure", "u/m"],
    ColumnRole.UNIT_PRICE: [
        "unit price",
        "price",
        "rate",
        "each",
        "unit cost",
        "cost",
        "unit_price",
    ],
    ColumnRole.LINE_TOTAL: [
        "total",
        "amount",
        "line total",
        "ext",
        "extended",
        "subtotal",
        "line_total",
        "line amount",
    ],
}


# ==============================================================================
# HELPERS
# ==============================================================================


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert to float, handling None, empty strings, dashes, currency."""
    if value is None or value == "" or value == "-" or value == "None":
        return default
    try:
        if isinstance(value, str):
            value = (
                value.replace("$", "")
                .replace(",", "")
                .replace("€", "")
                .replace("£", "")
            )
            value = value.replace("(", "-").replace(")", "").strip()
            if not value or value == "-":
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _extract_pattern_value(text: str, pattern: str) -> Optional[str]:
    """Extract a value using regex pattern."""
    if not text:
        return None
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


def _looks_like_invoice_number(value: str) -> bool:
    """Check if a value looks like an invoice number."""
    if not value:
        return False
    value_upper = value.upper().strip()
    invoice_patterns = [
        r"^INV[-\s]?\d",
        r"^INVOICE[-\s]?\d",
        r"^\d{4,}$",
        r"^[A-Z]{2,4}[-\s]?\d{4}",
        r"^\d+-\d+",
        r"^#\d+",
        r"^BILL[-\s]?\d",
        r"^\d{3,}-[A-Z0-9]+",
    ]
    for pattern in invoice_patterns:
        if re.match(pattern, value_upper):
            return True
    return False


def _looks_like_date(value: str) -> bool:
    if not value:
        return False
    date_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}",
    ]
    for pattern in date_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return True
    return False


def _looks_like_amount(value: str) -> bool:
    if not value:
        return False
    amount_patterns = [
        r"^\$[\d,]+\.?\d*$",
        r"^[\d,]+\.\d{2}$",
        r"^\$[\d,]+$",
    ]
    for pattern in amount_patterns:
        if re.match(pattern, value.strip()):
            return True
    return False


def _is_bill_to_company(candidate: str, raw_text: str) -> bool:
    """Check if candidate is likely the 'Bill To' company."""
    if not candidate:
        return False
    candidate_lower = candidate.lower()
    our_company_indicators = [
        "demo company",
        "your company",
        "my company",
        "acme corp",
        "test company",
    ]
    if any(ind in candidate_lower for ind in our_company_indicators):
        return True
    bill_to_patterns = [
        r"bill\s*to\s*:?\s*\n?\s*" + re.escape(candidate),
        r"ship\s*to\s*:?\s*\n?\s*" + re.escape(candidate),
        r"sold\s*to\s*:?\s*\n?\s*" + re.escape(candidate),
    ]
    for pattern in bill_to_patterns:
        if re.search(pattern, raw_text, re.IGNORECASE):
            return True
    return False


# ==============================================================================
# COLUMN ROLE DETECTION
# ==============================================================================


def _detect_column_role(header: str, sample_values: List[str]) -> ColumnRole:
    """Detect column role using header hints + value analysis."""
    header_lower = header.lower().strip()

    for role, keywords in HEADER_MAPPINGS.items():
        if any(kw in header_lower for kw in keywords):
            return role

    numeric_values = [_safe_float(v) for v in sample_values if _safe_float(v) > 0]

    if not numeric_values:
        non_empty = [v for v in sample_values if v and str(v).strip()]
        if non_empty and all(len(str(v)) > 10 for v in non_empty):
            return ColumnRole.DESCRIPTION
        return ColumnRole.UNKNOWN

    max_value = max(numeric_values)

    if all(v == int(v) and v < 100 for v in numeric_values):
        return ColumnRole.QUANTITY
    elif max_value > 1000:
        return ColumnRole.LINE_TOTAL
    else:
        return ColumnRole.UNIT_PRICE


# ==============================================================================
# MULTI-STRATEGY FIELD EXTRACTION
# ==============================================================================


def _extract_invoice_number(
    raw_text: str, raw_fields: Dict, result: Any
) -> FieldResult:
    """Extract invoice number using multi-strategy approach."""

    labeled_keys = [
        "invoice_number",
        "invoice_no",
        "invoice #",
        "inv_no",
        "invoice",
        "bill_number",
    ]
    for key in labeled_keys:
        value = raw_fields.get(key) or getattr(
            result, key.replace(" ", "_").replace("#", ""), None
        )
        if value and str(value).strip():
            return FieldResult(
                value=str(value).strip(),
                confidence=0.95,
                strategy=ExtractionStrategy.LABELED,
                source=f"field:{key}",
            )

    patterns = [
        (r"\b(INV-\d{4}-\d+)\b", 0.98),
        (r"INVOICE\s+(INV-[A-Z0-9\-]+)", 0.97),
        (r"\b([A-Z]{2,4}-\d{4}-[A-Z0-9]+)\b", 0.90),
        (r"Invoice\s+(?:Number|No\.?|#)\s*:?\s*([A-Z0-9][-A-Z0-9]+)", 0.85),
        (r"Invoice\s*:\s*([A-Z0-9][-A-Z0-9]+)", 0.80),
        (r"#\s*(\d{4,})", 0.70),
    ]

    for pattern, confidence in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if len(value) >= 3:
                return FieldResult(
                    value=value,
                    confidence=confidence,
                    strategy=ExtractionStrategy.PATTERN,
                    source=f"pattern:{pattern[:30]}...",
                )

    first_lines = raw_text.strip().split("\n")[:5]
    for line in first_lines:
        candidate = line.strip()
        if _looks_like_invoice_number(candidate) and len(candidate) >= 4:
            return FieldResult(
                value=candidate,
                confidence=0.60,
                strategy=ExtractionStrategy.HEURISTIC,
                source="first_line_heuristic",
            )

    return FieldResult(
        value="UNKNOWN",
        confidence=0.0,
        strategy=ExtractionStrategy.MISSING,
        source="not_found",
    )


def _extract_vendor_name(
    raw_text: str, raw_fields: Dict, result: Any
) -> FieldResult:
    """Extract vendor name using multi-strategy approach."""

    labeled_keys = [
        "vendor_name",
        "vendor",
        "from",
        "supplier",
        "company",
        "bill_from",
        "payee",
    ]
    for key in labeled_keys:
        value = raw_fields.get(key) or getattr(
            result, key.replace(" ", "_"), None
        )
        if value and str(value).strip():
            value_str = str(value).strip()
            if not _looks_like_invoice_number(
                value_str
            ) and not _is_bill_to_company(value_str, raw_text):
                return FieldResult(
                    value=value_str,
                    confidence=0.95,
                    strategy=ExtractionStrategy.LABELED,
                    source=f"field:{key}",
                )

    vendor_idx = raw_text.lower().find("vendor:")
    if vendor_idx >= 0:
        context = raw_text[vendor_idx : vendor_idx + 400]
        context_lines = context.split("\n")
        for line in context_lines[1:6]:
            candidate = line.strip()
            if not candidate or len(candidate) < 3:
                continue
            if re.match(r"^Date\s*:", candidate, re.IGNORECASE):
                continue
            if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", candidate):
                continue
            if re.match(
                r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d",
                candidate,
                re.IGNORECASE,
            ):
                continue
            if re.match(
                r"^\d+\s+[A-Z][a-z]+\s+(Street|St|Ave|Road|Rd|Blvd|Plaza|Drive|Dr)",
                candidate,
                re.IGNORECASE,
            ):
                continue
            if ":" in candidate and candidate.index(":") < 15:
                continue
            if not _looks_like_invoice_number(
                candidate
            ) and not _is_bill_to_company(candidate, raw_text):
                return FieldResult(
                    value=candidate,
                    confidence=0.95,
                    strategy=ExtractionStrategy.PATTERN,
                    source="vendor_label_nearby",
                )

    company_suffixes = r"(?:LLC|Inc\.?|Corp\.?|Ltd\.?|Co\.|L\.L\.C\.|Corporation|Company|Energy|Services|Consulting|Solutions|Technologies|Group|PLC)"
    company_pattern = rf"(?:^|\n)\s*([A-Za-z0-9][A-Za-z0-9\s&.,\'-]*?{company_suffixes})(?:\s*\n|$|,)"
    matches = re.finditer(company_pattern, raw_text, re.MULTILINE | re.IGNORECASE)

    for match in matches:
        candidate = match.group(1).strip()
        candidate = re.sub(r"[.,]+$", "", candidate).strip()
        if (
            len(candidate) > 3
            and not _looks_like_invoice_number(candidate)
            and not _is_bill_to_company(candidate, raw_text)
        ):
            return FieldResult(
                value=candidate,
                confidence=0.85,
                strategy=ExtractionStrategy.PATTERN,
                source="company_suffix_pattern",
            )

    general_patterns = [
        (
            r"(?:From|Supplier|Bill From|Payable to)[\s:]+([A-Za-z0-9][A-Za-z0-9\s&.,\'-]+?)(?:\n|$)",
            0.85,
        ),
    ]
    for pattern, confidence in general_patterns:
        match = re.search(pattern, raw_text, re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[.,]+$", "", candidate).strip()
            if (
                len(candidate) > 3
                and not _looks_like_invoice_number(candidate)
                and not _is_bill_to_company(candidate, raw_text)
            ):
                return FieldResult(
                    value=candidate,
                    confidence=confidence,
                    strategy=ExtractionStrategy.PATTERN,
                    source=f"pattern:{pattern[:30]}...",
                )

    first_lines = raw_text.strip().split("\n")[:10]
    for line in first_lines:
        candidate = line.strip()
        if (
            len(candidate) > 5
            and not _looks_like_invoice_number(candidate)
            and not _looks_like_date(candidate)
            and not _looks_like_amount(candidate)
            and not _is_bill_to_company(candidate, raw_text)
            and not candidate.lower().startswith(
                ("invoice", "bill to", "date", "due", "total", "ship to")
            )
        ):
            return FieldResult(
                value=candidate,
                confidence=0.50,
                strategy=ExtractionStrategy.HEURISTIC,
                source="first_line_heuristic",
            )

    return FieldResult(
        value="Unknown Vendor",
        confidence=0.0,
        strategy=ExtractionStrategy.MISSING,
        source="not_found",
    )


def _extract_date_field(
    raw_text: str, raw_fields: Dict, result: Any, field_name: str
) -> FieldResult:
    """Extract a date field (invoice_date or due_date)."""
    if field_name == "invoice_date":
        labeled_keys = ["invoice_date", "date", "bill_date", "invoice date"]
        patterns = [
            (
                r"(?:Invoice|Bill)\s+Date[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                0.90,
            ),
            (
                r"(?:Invoice|Bill)\s+Date[\s:]+(\w+\s+\d{1,2},?\s+\d{4})",
                0.85,
            ),
            (r"Date[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", 0.75),
        ]
    else:
        labeled_keys = ["due_date", "due", "payment_due", "due date"]
        patterns = [
            (r"Due\s+Date[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", 0.90),
            (r"Due\s+Date[\s:]+(\w+\s+\d{1,2},?\s+\d{4})", 0.85),
            (
                r"Payment\s+Due[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                0.85,
            ),
        ]

    for key in labeled_keys:
        value = raw_fields.get(key) or getattr(
            result, key.replace(" ", "_"), None
        )
        if value and str(value).strip():
            return FieldResult(
                value=str(value).strip(),
                confidence=0.95,
                strategy=ExtractionStrategy.LABELED,
                source=f"field:{key}",
            )

    for pattern, confidence in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return FieldResult(
                value=match.group(1).strip(),
                confidence=confidence,
                strategy=ExtractionStrategy.PATTERN,
                source=f"pattern:{pattern[:30]}...",
            )

    return FieldResult(
        value=None,
        confidence=0.0,
        strategy=ExtractionStrategy.MISSING,
        source="not_found",
    )


def _extract_po_number(
    raw_text: str, raw_fields: Dict, result: Any
) -> FieldResult:
    """Extract PO/Purchase Order number."""
    labeled_keys = [
        "po_number",
        "po",
        "purchase_order",
        "po_reference",
        "p.o.",
        "po #",
    ]
    for key in labeled_keys:
        value = raw_fields.get(key) or getattr(
            result,
            key.replace(" ", "_").replace(".", "").replace("#", ""),
            None,
        )
        if value and str(value).strip():
            return FieldResult(
                value=str(value).strip(),
                confidence=0.95,
                strategy=ExtractionStrategy.LABELED,
                source=f"field:{key}",
            )

    patterns = [
        (r"P\.?O\.?\s*Reference\s*:\s*([A-Z0-9][-A-Z0-9]+)", 0.95),
        (
            r"(?:P\.?O\.?|Purchase Order)\s*(?:Number|No\.?|#)?\s*:\s*([A-Z0-9][-A-Z0-9]+)",
            0.90,
        ),
        (
            r"(?:P\.?O\.?|Purchase Order)[\s#:-]+([A-Z0-9][-A-Z0-9]+)",
            0.85,
        ),
        (r"(?:PO|P/O)\s*:?\s*([A-Z0-9][-A-Z0-9]+)", 0.80),
    ]

    for pattern, confidence in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return FieldResult(
                value=match.group(1).strip(),
                confidence=confidence,
                strategy=ExtractionStrategy.PATTERN,
                source=f"pattern:{pattern[:30]}...",
            )

    return FieldResult(
        value=None,
        confidence=0.0,
        strategy=ExtractionStrategy.MISSING,
        source="not_found",
    )


# ==============================================================================
# FINANCIAL CALCULATIONS
# ==============================================================================


def _extract_financials(
    raw_fields: Dict, line_items: List[Dict], raw_text: str, result: Any
) -> Dict:
    """Extract financial totals with support for adjustments."""
    line_subtotal = (
        sum(li.get("total", 0) for li in line_items) if line_items else 0
    )

    discount = _safe_float(
        raw_fields.get("discount")
        or getattr(result, "discount", None)
        or _extract_pattern_value(raw_text, r"Discount[:\s]*-?\$?([\d,.]+)")
    )

    shipping = _safe_float(
        raw_fields.get("shipping")
        or raw_fields.get("freight")
        or getattr(result, "shipping", None)
        or _extract_pattern_value(
            raw_text, r"(?:Shipping|Freight)[:\s]*\$?([\d,.]+)"
        )
    )

    tax = _safe_float(
        raw_fields.get("tax")
        or raw_fields.get("tax_amount")
        or raw_fields.get("sales_tax")
        or getattr(result, "tax", None)
        or _extract_pattern_value(
            raw_text, r"(?:Tax|Sales Tax|VAT)[:\s]*\$?([\d,.]+)"
        )
    )

    explicit_subtotal = _safe_float(
        raw_fields.get("subtotal")
        or raw_fields.get("sub_total")
        or getattr(result, "subtotal", None)
        or _extract_pattern_value(raw_text, r"Subtotal[:\s]*\$?([\d,.]+)")
    )

    explicit_total = _safe_float(
        raw_fields.get("total")
        or raw_fields.get("total_amount")
        or raw_fields.get("amount_due")
        or raw_fields.get("balance_due")
        or getattr(result, "total", None)
        or _extract_pattern_value(
            raw_text,
            r"(?:Total|Amount Due|Balance Due|Grand Total)[:\s]*\$?([\d,.]+)",
        )
    )

    subtotal = explicit_subtotal if explicit_subtotal > 0 else line_subtotal
    calculated_total = subtotal - discount + shipping + tax

    total_source = "calculated"
    if explicit_total > 0:
        total = explicit_total
        total_source = "explicit"
    else:
        total = calculated_total if calculated_total > 0 else line_subtotal

    return {
        "subtotal": subtotal,
        "discount": discount,
        "shipping": shipping,
        "tax": tax,
        "total": total,
        "total_source": total_source,
    }


# ==============================================================================
# LINE ITEM NORMALIZATION
# ==============================================================================


def _normalize_line_items_with_roles(
    raw_line_items: List[Dict], tables: List = None
) -> List[Dict]:
    """Normalize raw line items using column role detection."""
    if not raw_line_items:
        return []

    column_roles = {}
    if tables and len(tables) > 0:
        for table in tables:
            if (
                isinstance(table, dict)
                and "headers" in table
                and "rows" in table
            ):
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                if len(rows) >= 1:
                    for i, header in enumerate(headers):
                        sample_values = [
                            row[i] if i < len(row) else "" for row in rows[:5]
                        ]
                        role = _detect_column_role(header, sample_values)
                        if role != ColumnRole.UNKNOWN:
                            column_roles[
                                header.lower().replace(" ", "_")
                            ] = role
                    break

    normalized = []
    for idx, li in enumerate(raw_line_items):
        if not isinstance(li, dict):
            continue

        normalized_li = {k.lower().replace(" ", "_"): v for k, v in li.items()}

        qty = _safe_float(
            normalized_li.get("qty")
            or normalized_li.get("quantity")
            or normalized_li.get("qnty")
            or normalized_li.get("units")
            or normalized_li.get("count")
            or 1
        )

        unit_price = _safe_float(
            normalized_li.get("unit_price")
            or normalized_li.get("price")
            or normalized_li.get("rate")
            or normalized_li.get("unit_cost")
            or normalized_li.get("each")
            or normalized_li.get("cost")
            or 0
        )

        total = _safe_float(
            normalized_li.get("total")
            or normalized_li.get("line_total")
            or normalized_li.get("ext")
            or normalized_li.get("extended")
            or normalized_li.get("amount")
            or (qty * unit_price if qty > 0 and unit_price > 0 else 0)
        )

        if total == 0 and qty > 0 and unit_price > 0:
            total = qty * unit_price

        description = str(
            normalized_li.get("description")
            or normalized_li.get("item")
            or normalized_li.get("product")
            or normalized_li.get("service")
            or normalized_li.get("name")
            or normalized_li.get("details")
            or ""
        ).strip()

        item_id = (
            normalized_li.get("item_id")
            or normalized_li.get("sku")
            or normalized_li.get("code")
            or normalized_li.get("part_#")
            or normalized_li.get("product_code")
        )

        normalized.append(
            {
                "line_id": f"L{idx+1}",
                "description": description,
                "quantity": qty,
                "unit_price": unit_price,
                "total": total,
                "gl_code": normalized_li.get("gl_code")
                or normalized_li.get("account"),
                "item_id": item_id,
            }
        )

    return normalized


# ==============================================================================
# MAIN NORMALIZATION FUNCTION
# ==============================================================================


def normalize_extraction_result(result: Any) -> Dict[str, Any]:
    """
    Normalize raw Docling output using multi-strategy field extraction.

    Args:
        result: Raw ExtractedInvoice from DoclingExtractor

    Returns:
        Normalized dict with confidence scores per field
    """
    raw_text = getattr(result, "raw_text", "") or ""
    raw_fields = getattr(result, "extracted_fields", {}) or {}
    raw_line_items = result.line_items or []
    tables = getattr(result, "tables", []) or []

    line_items = _normalize_line_items_with_roles(raw_line_items, tables)

    invoice_number = _extract_invoice_number(raw_text, raw_fields, result)
    vendor_name = _extract_vendor_name(raw_text, raw_fields, result)
    invoice_date = _extract_date_field(
        raw_text, raw_fields, result, "invoice_date"
    )
    due_date = _extract_date_field(raw_text, raw_fields, result, "due_date")
    po_number = _extract_po_number(raw_text, raw_fields, result)

    financials = _extract_financials(raw_fields, line_items, raw_text, result)

    return {
        "success": True,
        "invoice_number": invoice_number.value,
        "vendor_name": vendor_name.value,
        "vendor_address": raw_fields.get("vendor_address")
        or getattr(result, "vendor_address", None),
        "invoice_date": invoice_date.value,
        "due_date": due_date.value,
        "po_number": po_number.value,
        "subtotal": financials["subtotal"],
        "discount": financials["discount"],
        "shipping": financials["shipping"],
        "tax": financials["tax"],
        "total": financials["total"],
        "currency": raw_fields.get("currency")
        or getattr(result, "currency", None)
        or "USD",
        "line_items": line_items,
        "confidence": {
            "invoice_number": invoice_number.confidence,
            "vendor_name": vendor_name.confidence,
            "invoice_date": invoice_date.confidence,
            "due_date": due_date.confidence,
            "po_number": po_number.confidence,
            "total": 0.95
            if financials["total_source"] == "explicit"
            else 0.70,
        },
        "extraction_methods": {
            "invoice_number": invoice_number.strategy.value,
            "vendor_name": vendor_name.strategy.value,
            "invoice_date": invoice_date.strategy.value,
            "due_date": due_date.strategy.value,
            "po_number": po_number.strategy.value,
            "total": financials["total_source"],
        },
        "raw_text": raw_text,
        "tables": tables,
        "extraction_time_ms": getattr(result, "extraction_time_ms", 0),
        "source": getattr(result, "source", "docling"),
    }


# ==============================================================================
# SERVICE CLASS
# ==============================================================================

_extraction_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="docling_"
)


class ExtractionService:
    """
    High-level extraction service wrapping DoclingExtractor.

    Accepts base64-encoded PDF content and returns normalized results
    with multi-strategy field detection and per-field confidence scores.
    """

    def __init__(self):
        self._extractor = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Docling extractor."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                from pdf_ingestion.extractor import DoclingExtractor

                self._extractor = DoclingExtractor()
                self._initialized = True

            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                self._extractor = None
                self._initialized = True

    @property
    def is_available(self) -> bool:
        return self._extractor is not None and self._extractor.is_available

    async def extract(
        self,
        file_content_b64: str,
        filename: str = "invoice.pdf",
        content_type: str = "application/pdf",
    ) -> Dict[str, Any]:
        """
        Extract invoice data from base64-encoded PDF.

        Args:
            file_content_b64: Base64 encoded file content
            filename: Original filename
            content_type: MIME type

        Returns:
            Normalized extraction result dict
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        if self._extractor is None:
            return {
                "success": False,
                "error": "Extraction service not initialized",
                "extraction_time_ms": (time.time() - start_time) * 1000,
                "source": "error",
            }

        try:
            file_content = base64.b64decode(file_content_b64)
            ext = Path(filename).suffix.lower() if filename else ".pdf"
            if not ext:
                ext = ".pdf"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    _extraction_executor,
                    self._extract_sync,
                    tmp_path,
                )
                return result

            finally:
                import os

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "extraction_time_ms": (time.time() - start_time) * 1000,
                "source": "error",
            }

    def _extract_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronous extraction with multi-strategy normalization."""
        try:
            result = self._extractor.extract_invoice(file_path)
            return normalize_extraction_result(result)
        except Exception as e:
            logger.error(f"Sync extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": "error",
            }

    def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "name": "extraction",
            "healthy": self._initialized,
            "status": "ready"
            if self.is_available
            else ("fallback" if self._initialized else "not_initialized"),
            "details": {
                "docling_available": self.is_available,
                "initialized": self._initialized,
            },
        }
