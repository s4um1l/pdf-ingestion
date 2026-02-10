"""
Docling Extractor: IBM Docling for PDF/invoice structured extraction.

Uses DocLayNet-trained models for layout analysis + table extraction.
Falls back to pypdf when Docling is unavailable.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedInvoice:
    """Structured invoice data extracted from a PDF."""

    invoice_number: Optional[str]
    vendor_name: Optional[str]
    vendor_address: Optional[str]
    invoice_date: Optional[str]
    due_date: Optional[str]
    po_number: Optional[str]
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    currency: str
    line_items: List[Dict[str, Any]]
    raw_text: str
    tables: List[List[List[str]]]  # List of tables, each table is rows of cells
    confidence: float
    extraction_time_ms: float
    source: str = "docling"


class DoclingExtractor:
    """
    IBM Docling-based document extractor.

    Features:
    - Layout-aware extraction (DocLayNet)
    - Table structure detection
    - Form field extraction
    - Multi-page support
    - Automatic fallback to pypdf when Docling is unavailable
    """

    def __init__(self, use_gpu: bool = False):
        self._docling_available = False
        self._converter = None
        self._use_gpu = use_gpu

        try:
            from docling.document_converter import DocumentConverter

            try:
                from docling.datamodel.base_models import InputFormat
            except ImportError:
                try:
                    from docling.document_converter import InputFormat
                except ImportError:
                    InputFormat = None

            try:
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
            except ImportError:
                pass

            self._converter = DocumentConverter()
            self._docling_available = True

        except ImportError:
            self._docling_available = False
        except FileNotFoundError:
            self._docling_available = False
        except Exception:
            self._docling_available = False

    @property
    def is_available(self) -> bool:
        """Check if Docling is available."""
        return self._docling_available

    def extract_invoice(self, pdf_path: str) -> ExtractedInvoice:
        """
        Extract structured invoice data from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedInvoice with extracted fields
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not self._docling_available:
            return self._fallback_extraction(pdf_path, start_time)

        try:
            result = None

            # Try method 1: Docling 1.0+ with DocumentConversionInput
            try:
                from docling.datamodel.document import DocumentConversionInput

                input_doc = DocumentConversionInput.from_paths([pdf_path])
                results = self._converter.convert(input_doc)
                result = next(iter(results))
            except ImportError:
                pass

            # Try method 2: Alternative import path
            if result is None:
                try:
                    from docling.document_converter import DocumentConversionInput

                    input_doc = DocumentConversionInput.from_paths([pdf_path])
                    results = self._converter.convert(input_doc)
                    result = next(iter(results))
                except ImportError:
                    pass

            # Try method 3: Direct path string (older Docling versions)
            if result is None:
                try:
                    results = self._converter.convert(str(pdf_path))
                    if hasattr(results, "__iter__") and not hasattr(results, "pages"):
                        result = next(iter(results))
                    else:
                        result = results
                except Exception:
                    pass

            # Try method 4: List of paths
            if result is None:
                try:
                    results = self._converter.convert([str(pdf_path)])
                    if hasattr(results, "__iter__") and not hasattr(results, "pages"):
                        result = next(iter(results))
                    else:
                        result = results
                except Exception:
                    pass

            if result is None:
                raise RuntimeError(
                    "Could not convert document with any Docling API version"
                )

            raw_text = self._extract_text_from_pages(result)
            tables = self._extract_tables_from_pages(result)
            fields = self._extract_invoice_fields(raw_text, tables)
            extraction_time = (time.time() - start_time) * 1000

            return ExtractedInvoice(
                invoice_number=fields.get("invoice_number"),
                vendor_name=fields.get("vendor_name"),
                vendor_address=fields.get("vendor_address"),
                invoice_date=fields.get("invoice_date"),
                due_date=fields.get("due_date"),
                po_number=fields.get("po_number"),
                subtotal=fields.get("subtotal"),
                tax=fields.get("tax"),
                total=fields.get("total"),
                currency=fields.get("currency", "USD"),
                line_items=fields.get("line_items", []),
                raw_text=raw_text,
                tables=tables,
                confidence=self._compute_confidence(fields),
                extraction_time_ms=extraction_time,
                source="docling",
            )

        except Exception:
            return self._fallback_extraction(pdf_path, start_time)

    def _extract_text_from_pages(self, result) -> str:
        """Extract text from Docling ConvertedDocument pages."""
        text_parts = []

        try:
            for page in result.pages:
                if not hasattr(page, "predictions") or not page.predictions:
                    continue

                layout = getattr(page.predictions, "layout", None)
                if not layout or not hasattr(layout, "clusters"):
                    continue

                clusters = sorted(
                    layout.clusters,
                    key=lambda c: (
                        c.bbox.t if hasattr(c.bbox, "t") else 0,
                        c.bbox.l if hasattr(c.bbox, "l") else 0,
                    ),
                )

                for cluster in clusters:
                    if cluster.label == "Table":
                        continue

                    if hasattr(cluster, "cells") and cluster.cells:
                        for cell in cluster.cells:
                            text = getattr(cell, "text", "").strip()
                            if text:
                                text_parts.append(text)
        except Exception:
            pass

        return "\n".join(text_parts)

    def _extract_tables_from_pages(self, result) -> List[List[List[str]]]:
        """Extract tables from Docling ConvertedDocument pages."""
        tables = []

        try:
            for page in result.pages:
                if not hasattr(page, "predictions") or not page.predictions:
                    continue

                table_structure = getattr(page.predictions, "tablestructure", None)
                if not table_structure or not hasattr(table_structure, "table_map"):
                    continue

                for table_id, table_element in table_structure.table_map.items():
                    if (
                        hasattr(table_element, "table_cells")
                        and table_element.table_cells
                    ):
                        num_rows = getattr(table_element, "num_rows", 0)
                        num_cols = getattr(table_element, "num_cols", 0)

                        if num_rows > 0 and num_cols > 0:
                            grid = [
                                ["" for _ in range(num_cols)] for _ in range(num_rows)
                            ]

                            for cell in table_element.table_cells:
                                row_idx = getattr(cell, "start_row_offset_idx", 0)
                                col_idx = getattr(cell, "start_col_offset_idx", 0)
                                text = getattr(cell, "text", "").strip()

                                if (
                                    0 <= row_idx < num_rows
                                    and 0 <= col_idx < num_cols
                                ):
                                    grid[row_idx][col_idx] = text

                            grid = [
                                row
                                for row in grid
                                if any(cell.strip() for cell in row)
                            ]

                            if len(grid) > 1 and any(
                                any(cell for cell in row) for row in grid
                            ):
                                tables.append(grid)
        except Exception:
            pass

        return tables

    def _extract_invoice_fields(self, text: str, tables: List) -> Dict[str, Any]:
        """Extract invoice fields from text and tables."""
        fields = {}
        lines = text.strip().split("\n")

        # Invoice number patterns
        inv_patterns = [
            r"\b(INV-\d{4}-\d+)\b",
            r"INVOICE\s+(INV-[A-Z0-9\-]+)",
            r"INVOICE\s*\n\s*(INV-[A-Z0-9\-]+)",
            r"\b([A-Z]{2,4}-\d{4}-[A-Z0-9]+)\b",
            r"invoice\s*#\s*:\s*([A-Z0-9\-]+)",
            r"invoice\s+number\s*:\s*([A-Z0-9\-]+)",
            r"inv[#:\s]+([A-Z0-9\-]+)",
            r"bill\s*#\s*:\s*([A-Z0-9\-]+)",
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_number"] = match.group(1).strip()
                break

        # PO number
        po_patterns = [
            r"p\.?o\.?\s*reference\s*:\s*([A-Z0-9\-]+)",
            r"p\.?o\.?\s*(?:number|#)?\s*:\s*([A-Z0-9\-]+)",
            r"purchase\s+order\s*(?:reference|ref|number|#)?\s*:\s*([A-Z0-9\-]+)",
            r"order\s*(?:reference|ref|number|#)?\s*:\s*([A-Z0-9\-]+)",
        ]
        for pattern in po_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["po_number"] = match.group(1).strip()
                break

        # Dates
        date_patterns = [
            r"date\s*:\s*(\d{4}-\d{2}-\d{2})",
            r"date\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_date"] = match.group(1)
                break

        # Due date
        due_patterns = [
            r"due\s*:\s*(\d{4}-\d{2}-\d{2})",
            r"due\s*date\s*:\s*(\d{4}-\d{2}-\d{2})",
            r"due\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]
        for pattern in due_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["due_date"] = match.group(1)
                break

        # Amounts
        subtotal_match = re.search(
            r"subtotal\s*:\s*\$?\s*([\d,]+\.?\d*)", text, re.IGNORECASE
        )
        if subtotal_match:
            fields["subtotal"] = float(subtotal_match.group(1).replace(",", ""))

        total_match = re.search(
            r"(?<![a-z])total\s*:\s*\$?\s*([\d,]+\.?\d*)", text, re.IGNORECASE
        )
        if total_match:
            fields["total"] = float(total_match.group(1).replace(",", ""))

        amount_due_match = re.search(
            r"amount\s+due\s*:\s*\$?\s*([\d,]+\.?\d*)", text, re.IGNORECASE
        )
        if amount_due_match:
            fields["total"] = float(amount_due_match.group(1).replace(",", ""))

        tax_match = re.search(
            r"(?<![a-z])tax\s*:\s*\$?\s*([\d,]+\.?\d*)", text, re.IGNORECASE
        )
        if tax_match:
            fields["tax"] = float(tax_match.group(1).replace(",", ""))

        # Vendor name extraction
        def is_bill_to_company(candidate: str) -> bool:
            if not candidate:
                return False
            candidate_lower = candidate.lower()
            if any(
                ind in candidate_lower
                for ind in [
                    "demo company",
                    "your company",
                    "my company",
                    "acme corp",
                    "test company",
                ]
            ):
                return True
            bill_to_match = re.search(
                r"bill\s*to\s*:?\s*\n?\s*" + re.escape(candidate),
                text,
                re.IGNORECASE,
            )
            if bill_to_match:
                return True
            return False

        company_suffixes = [
            "LLC",
            "Inc",
            "Inc.",
            "PLC",
            "Corp",
            "Corp.",
            "Ltd",
            "Ltd.",
            "Co.",
            "Company",
            "L.L.C.",
            "Corporation",
            "Energy",
            "Services",
            "Solutions",
            "Technologies",
            "Group",
        ]

        # Strategy 1: "Vendor:" label
        vendor_section_match = re.search(
            r"Vendor\s*:\s*\n((?:[^\n]+\n){1,5})", text, re.IGNORECASE
        )
        if vendor_section_match:
            vendor_lines = vendor_section_match.group(1).strip().split("\n")
            for candidate in vendor_lines:
                candidate = candidate.strip()
                if not candidate:
                    continue
                if re.match(r"^Date\s*:", candidate, re.IGNORECASE):
                    continue
                if re.match(
                    r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d",
                    candidate,
                    re.IGNORECASE,
                ):
                    continue
                if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", candidate):
                    continue
                if re.match(
                    r"^\d+\s+[A-Z][a-z]+\s+(Street|St|Ave|Road|Rd|Blvd|Plaza|Drive|Dr)",
                    candidate,
                    re.IGNORECASE,
                ):
                    continue
                if not is_bill_to_company(candidate):
                    fields["vendor_name"] = candidate
                    break

        # Strategy 2: Company name patterns
        if "vendor_name" not in fields:
            for line in lines:
                line = line.strip()
                if any(suffix in line for suffix in company_suffixes):
                    if not is_bill_to_company(line):
                        fields["vendor_name"] = line
                        break

        # Strategy 3: "Vendor:" context
        if "vendor_name" not in fields:
            vendor_idx = text.lower().find("vendor")
            if vendor_idx >= 0:
                context = text[vendor_idx : vendor_idx + 200]
                context_lines = context.split("\n")
                for ctx_line in context_lines[1:5]:
                    ctx_line = ctx_line.strip()
                    if ctx_line and len(ctx_line) > 2:
                        if not any(
                            x in ctx_line.lower()
                            for x in ["date", "invoice", "bill to", "due", ":"]
                        ):
                            if any(
                                suffix in ctx_line for suffix in company_suffixes
                            ) or (len(ctx_line) > 3 and ctx_line[0].isalnum()):
                                if not is_bill_to_company(ctx_line):
                                    fields["vendor_name"] = ctx_line
                                    break

        # Strategy 4: From/Seller patterns
        if "vendor_name" not in fields:
            vendor_patterns = [
                r"from\s*:\s*\n?\s*([A-Za-z0-9][A-Za-z0-9\s&\.,\-]+)",
                r"bill\s+from\s*:\s*\n?\s*([A-Za-z0-9][A-Za-z0-9\s&\.,\-]+)",
                r"seller\s*:\s*\n?\s*([A-Za-z0-9][A-Za-z0-9\s&\.,\-]+)",
            ]
            for pattern in vendor_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if not is_bill_to_company(candidate):
                        fields["vendor_name"] = candidate
                        break

        # Line items from tables
        fields["line_items"] = self._parse_line_items(tables)

        # Currency
        currency_match = re.search(r"currency\s*:\s*([A-Z]{3})", text, re.IGNORECASE)
        fields["currency"] = currency_match.group(1) if currency_match else "USD"

        return fields

    def _parse_line_items(self, tables: List[List[List[str]]]) -> List[Dict]:
        """Parse line items from extracted tables."""
        line_items = []

        for table in tables:
            if len(table) < 2:
                continue

            header = [h.lower() for h in table[0]]
            has_qty = any("qty" in h or "quantity" in h for h in header)
            has_amount = any("amount" in h or "price" in h or "total" in h for h in header)

            if has_qty or has_amount:
                for row in table[1:]:
                    if len(row) >= len(header):
                        item = {}
                        for i, h in enumerate(header):
                            if i < len(row):
                                item[h] = row[i]

                        desc = (
                            str(
                                item.get(
                                    "description",
                                    item.get("item", item.get("product", "")),
                                )
                            )
                            .strip()
                            .lower()
                        )

                        all_values_lower = " ".join(
                            str(v).lower() for v in item.values()
                        )
                        is_summary_row = any(
                            kw in all_values_lower
                            for kw in [
                                "subtotal",
                                "tax:",
                                "total:",
                                "amount due",
                                "grand total",
                            ]
                        )

                        has_description = bool(desc) and desc not in ["", "-", "n/a"]
                        has_qty_value = any(
                            str(item.get(k, ""))
                            .strip()
                            .replace("$", "")
                            .replace(",", "")
                            .replace(".", "")
                            .isdigit()
                            for k in ["qty", "quantity"]
                        )

                        if (has_description or has_qty_value) and not is_summary_row:
                            line_items.append(item)

        # Deduplicate
        seen = set()
        unique_items = []
        for item in line_items:
            key = (
                str(item.get("description", item.get("item", ""))).strip().lower(),
                str(item.get("quantity", item.get("qty", ""))).strip(),
                str(
                    item.get("unit_price", item.get("price", item.get("unit price", "")))
                ).strip(),
            )
            if key not in seen and any(k for k in key):
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def _compute_confidence(self, fields: Dict) -> float:
        """Compute extraction confidence score."""
        confidence = 0.0

        if fields.get("invoice_number"):
            confidence += 0.3
        if fields.get("total"):
            confidence += 0.25
        if fields.get("invoice_date"):
            confidence += 0.15
        if fields.get("vendor_name"):
            confidence += 0.15
        if fields.get("po_number"):
            confidence += 0.1
        if fields.get("line_items"):
            confidence += 0.05

        return min(confidence, 1.0)

    def _fallback_extraction(
        self, pdf_path: Path, start_time: float
    ) -> ExtractedInvoice:
        """Fallback to basic pypdf extraction when Docling is unavailable."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text() + "\n"

            fields = self._extract_invoice_fields(raw_text, [])
            extraction_time = (time.time() - start_time) * 1000

            return ExtractedInvoice(
                invoice_number=fields.get("invoice_number"),
                vendor_name=fields.get("vendor_name"),
                vendor_address=fields.get("vendor_address"),
                invoice_date=fields.get("invoice_date"),
                due_date=fields.get("due_date"),
                po_number=fields.get("po_number"),
                subtotal=fields.get("subtotal"),
                tax=fields.get("tax"),
                total=fields.get("total"),
                currency=fields.get("currency", "USD"),
                line_items=[],
                raw_text=raw_text,
                tables=[],
                confidence=self._compute_confidence(fields) * 0.7,
                extraction_time_ms=extraction_time,
                source="pypdf_fallback",
            )

        except Exception as e:
            extraction_time = (time.time() - start_time) * 1000
            return ExtractedInvoice(
                invoice_number=None,
                vendor_name=None,
                vendor_address=None,
                invoice_date=None,
                due_date=None,
                po_number=None,
                subtotal=None,
                tax=None,
                total=None,
                currency="USD",
                line_items=[],
                raw_text=f"Error: {e}",
                tables=[],
                confidence=0.0,
                extraction_time_ms=extraction_time,
                source="error",
            )

    def to_dict(self, invoice: ExtractedInvoice) -> Dict[str, Any]:
        """Convert ExtractedInvoice to dict for JSON serialization."""
        return {
            "invoice_number": invoice.invoice_number,
            "vendor_name": invoice.vendor_name,
            "vendor_address": invoice.vendor_address,
            "invoice_date": invoice.invoice_date,
            "due_date": invoice.due_date,
            "po_number": invoice.po_number,
            "subtotal": invoice.subtotal,
            "tax": invoice.tax,
            "total": invoice.total,
            "currency": invoice.currency,
            "line_items": invoice.line_items,
            "confidence": invoice.confidence,
            "extraction_time_ms": invoice.extraction_time_ms,
            "source": invoice.source,
        }


# Singleton
_extractor: Optional[DoclingExtractor] = None


def get_extractor() -> DoclingExtractor:
    """Get or create the singleton DoclingExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = DoclingExtractor()
    return _extractor


def extract_invoice(pdf_path: str) -> ExtractedInvoice:
    """
    Extract structured data from an invoice PDF (synchronous).

    Args:
        pdf_path: Path to PDF file

    Returns:
        ExtractedInvoice with all extracted fields
    """
    extractor = get_extractor()
    return extractor.extract_invoice(pdf_path)


async def extract_invoice_async(pdf_path: str) -> ExtractedInvoice:
    """
    Extract structured data from an invoice PDF (async).

    Runs the extraction in a thread pool to avoid blocking the event loop.

    Args:
        pdf_path: Path to PDF file

    Returns:
        ExtractedInvoice with all extracted fields
    """
    extractor = get_extractor()
    return await asyncio.to_thread(extractor.extract_invoice, pdf_path)
