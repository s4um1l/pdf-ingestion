"""
PDF Parser: Extract text and metadata from invoice PDFs using pypdf.

Lightweight alternative that works without Docling's ML models.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class PDFInvoice:
    """Structured representation of a parsed PDF invoice."""

    filepath: str
    raw_text: str
    num_pages: int
    metadata: Dict
    extracted_tables: List[List[str]]
    confidence: float


class PDFParser:
    """Parse invoice PDFs with pypdf."""

    def __init__(self):
        self.parse_count = 0

    def parse_invoice(self, pdf_path: str) -> PDFInvoice:
        """
        Extract text from PDF invoice.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFInvoice with raw text and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.parse_count += 1
        reader = PdfReader(str(pdf_path))

        raw_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            raw_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        metadata = {
            "producer": reader.metadata.get("/Producer", ""),
            "creator": reader.metadata.get("/Creator", ""),
            "creation_date": reader.metadata.get("/CreationDate", ""),
        }

        confidence = self._compute_extraction_confidence(raw_text, len(reader.pages))
        tables = self._extract_tables(raw_text)

        return PDFInvoice(
            filepath=str(pdf_path),
            raw_text=raw_text,
            num_pages=len(reader.pages),
            metadata=metadata,
            extracted_tables=tables,
            confidence=confidence,
        )

    def _compute_extraction_confidence(self, text: str, num_pages: int) -> float:
        """Estimate extraction quality."""
        if not text or len(text.strip()) < 50:
            return 0.0

        has_invoice_number = bool(re.search(r"invoice\s*#?", text, re.IGNORECASE))
        has_amount = bool(re.search(r"\$\s*[\d,]+\.\d{2}", text))
        has_date = bool(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text))

        confidence = 0.3

        if has_invoice_number:
            confidence += 0.3
        if has_amount:
            confidence += 0.2
        if has_date:
            confidence += 0.2

        if len(text) < 200:
            confidence *= 0.5

        return min(confidence, 1.0)

    def _extract_tables(self, text: str) -> List[List[str]]:
        """Extract tabular data using basic heuristics."""
        tables = []
        line_item_pattern = r"(.+?)\s+(\d+)\s+\$?([\d,]+\.\d{2})"
        matches = re.findall(line_item_pattern, text)

        if matches:
            tables.append(
                [["Description", "Quantity", "Amount"]] + list(matches)
            )

        return tables

    def extract_invoice_regions(self, pdf_path: str) -> Dict[str, str]:
        """Extract specific regions (header, line items, totals)."""
        invoice = self.parse_invoice(pdf_path)
        text = invoice.raw_text

        regions = {"header": "", "line_items": "", "totals": "", "footer": ""}
        lines = text.split("\n")

        current_region = "header"
        for line in lines:
            line_lower = line.lower()

            if any(kw in line_lower for kw in ["item", "description", "qty"]):
                current_region = "line_items"
            elif any(kw in line_lower for kw in ["total", "subtotal", "tax"]):
                current_region = "totals"
            elif any(kw in line_lower for kw in ["thank", "terms", "payment"]):
                current_region = "footer"

            regions[current_region] += line + "\n"

        return regions

    def validate_invoice_structure(self, pdf_path: str) -> Dict[str, bool]:
        """Validate that PDF has expected invoice structure."""
        invoice = self.parse_invoice(pdf_path)
        text = invoice.raw_text.lower()

        return {
            "has_invoice_number": bool(
                re.search(r"invoice\s*#?\s*:?\s*\d+", text)
            ),
            "has_vendor_name": bool(re.search(r"from\s*:|bill\s+from", text)),
            "has_date": bool(re.search(r"date\s*:", text)),
            "has_total": bool(re.search(r"total\s*:?\s*\$", text)),
            "has_line_items": len(invoice.extracted_tables) > 0,
            "is_multi_page": invoice.num_pages > 1,
            "confidence_acceptable": invoice.confidence >= 0.5,
        }


class ExtractionMetrics:
    """Track extraction quality metrics."""

    def __init__(self):
        self.total_extractions = 0
        self.successful_extractions = 0
        self.total_confidence = 0.0
        self.parse_times = []

    def record_extraction(self, success: bool, confidence: float, time_ms: float):
        self.total_extractions += 1
        if success:
            self.successful_extractions += 1
        self.total_confidence += confidence
        self.parse_times.append(time_ms)

    def get_success_rate(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions

    def get_average_confidence(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.total_confidence / self.total_extractions

    def get_p95_parse_time(self) -> float:
        if not self.parse_times:
            return 0.0
        return sorted(self.parse_times)[int(len(self.parse_times) * 0.95)]
