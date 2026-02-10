"""
PDF Ingestion Service - Extract structured data from invoice PDFs.

Standalone library for parsing invoices using IBM Docling (primary)
with pypdf fallback. Optional VLM integration via Ollama (Qwen-VL)
or Claude for screenshot-based extraction.

Usage:
    from pdf_ingestion import extract_invoice, extract_invoice_async

    result = extract_invoice("path/to/invoice.pdf")
    print(result.vendor_name, result.total, result.line_items)

VLM Usage:
    from pdf_ingestion.vlm import VLMRouter

    router = VLMRouter(provider="qwen")  # free, local via Ollama
    fields = await router.extract_invoice_fields(screenshot_b64, ["vendor_name", "total"])
"""

from pdf_ingestion.extractor import (
    DoclingExtractor,
    ExtractedInvoice,
    extract_invoice,
    extract_invoice_async,
    get_extractor,
)
from pdf_ingestion.field_extractor import FieldExtractor, ExtractedField
from pdf_ingestion.pdf_parser import PDFParser, PDFInvoice
from pdf_ingestion.schema import (
    Invoice,
    LineItem,
    InvoiceStatus,
    InvoiceValidationResult,
    validate_invoice,
)
from pdf_ingestion.extraction_service import ExtractionService

# VLM types are always available (no external deps)
from pdf_ingestion.vlm.vlm_types import VLMRegion

__version__ = "1.1.0"

__all__ = [
    # Core extraction
    "extract_invoice",
    "extract_invoice_async",
    "get_extractor",
    "DoclingExtractor",
    "ExtractedInvoice",
    # Field extraction
    "FieldExtractor",
    "ExtractedField",
    # PDF parsing
    "PDFParser",
    "PDFInvoice",
    # Schema & validation
    "Invoice",
    "LineItem",
    "InvoiceStatus",
    "InvoiceValidationResult",
    "validate_invoice",
    # Service wrapper
    "ExtractionService",
    # VLM types
    "VLMRegion",
]
