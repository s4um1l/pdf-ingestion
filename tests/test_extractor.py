"""Tests for the core DoclingExtractor."""

import pytest
from pathlib import Path

from pdf_ingestion.extractor import DoclingExtractor, ExtractedInvoice


class TestDoclingExtractor:
    """Test the main extractor (uses pypdf fallback when Docling is not installed)."""

    @pytest.fixture
    def extractor(self):
        return DoclingExtractor()

    def test_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.extract_invoice("/nonexistent/path.pdf")

    def test_extract_returns_extracted_invoice(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert isinstance(result, ExtractedInvoice)

    def test_extract_has_raw_text(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.raw_text is not None
        assert len(result.raw_text) > 0

    def test_extract_has_confidence(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert 0.0 <= result.confidence <= 1.0

    def test_extract_has_source(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.source in ("docling", "pypdf_fallback", "error")

    def test_extract_has_extraction_time(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.extraction_time_ms > 0

    def test_to_dict(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        d = extractor.to_dict(result)
        assert isinstance(d, dict)
        assert "invoice_number" in d
        assert "vendor_name" in d
        assert "total" in d
        assert "confidence" in d
        assert "source" in d

    def test_extract_detects_invoice_number(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.invoice_number is not None
        assert "INV" in result.invoice_number

    def test_extract_detects_vendor(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.vendor_name is not None

    def test_extract_detects_total(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.total is not None
        assert result.total > 0

    def test_extract_detects_dates(self, extractor, sample_invoice_pdf):
        result = extractor.extract_invoice(str(sample_invoice_pdf))
        assert result.invoice_date is not None


class TestExtractInvoiceFields:
    """Test the regex field extraction directly."""

    @pytest.fixture
    def extractor(self):
        return DoclingExtractor()

    def test_invoice_number_extraction(self, extractor):
        text = "INVOICE INV-2026-000001\nVendor: Test Corp\nDate: 2026-01-15"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("invoice_number") == "INV-2026-000001"

    def test_total_extraction(self, extractor):
        text = "Subtotal: $500.00\nTax: $50.00\nTotal: $550.00"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields["total"] == 550.00
        assert fields["subtotal"] == 500.00
        assert fields["tax"] == 50.00

    def test_vendor_extraction_from_label(self, extractor):
        text = "Vendor:\nSpringfield Energy LLC\n123 Main St\nDate: 2026-01-15"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("vendor_name") == "Springfield Energy LLC"

    def test_vendor_extraction_from_suffix(self, extractor):
        text = "Some Header\nGlobal Services LLC\nInvoice #: 12345"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("vendor_name") == "Global Services LLC"

    def test_po_number_extraction(self, extractor):
        text = "PO Reference: PO-2026-000001\nInvoice Date: 2026-01-15"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("po_number") == "PO-2026-000001"

    def test_date_extraction(self, extractor):
        text = "Invoice Date: 2026-01-15\nDue: 2026-02-15"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("invoice_date") == "2026-01-15"
        assert fields.get("due_date") == "2026-02-15"

    def test_amount_due_overrides_total(self, extractor):
        text = "Subtotal: $100.00\nTotal: $110.00\nAmount Due: $110.00"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields["total"] == 110.00

    def test_bill_to_not_used_as_vendor(self, extractor):
        text = "Vendor:\nReal Vendor Inc\nBill To:\nDemo Company\n123 Fake St"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("vendor_name") == "Real Vendor Inc"

    def test_currency_default_usd(self, extractor):
        text = "Total: $100.00"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("currency") == "USD"

    def test_currency_detection(self, extractor):
        text = "Currency: EUR\nTotal: 100.00"
        fields = extractor._extract_invoice_fields(text, [])
        assert fields.get("currency") == "EUR"


class TestConfidenceScoring:
    """Test confidence computation."""

    @pytest.fixture
    def extractor(self):
        return DoclingExtractor()

    def test_all_fields_present(self, extractor):
        fields = {
            "invoice_number": "INV-001",
            "total": 100.0,
            "invoice_date": "2026-01-01",
            "vendor_name": "Test Corp",
            "po_number": "PO-001",
            "line_items": [{"desc": "item"}],
        }
        assert extractor._compute_confidence(fields) == 1.0

    def test_no_fields_present(self, extractor):
        assert extractor._compute_confidence({}) == 0.0

    def test_partial_fields(self, extractor):
        fields = {"invoice_number": "INV-001", "total": 100.0}
        confidence = extractor._compute_confidence(fields)
        assert 0.5 <= confidence <= 0.6


class TestLineItemParsing:
    """Test table-based line item extraction."""

    @pytest.fixture
    def extractor(self):
        return DoclingExtractor()

    def test_basic_line_items(self, extractor):
        tables = [
            [
                ["Description", "Qty", "Unit Price", "Total"],
                ["Widget A", "10", "$5.00", "$50.00"],
                ["Widget B", "5", "$10.00", "$50.00"],
            ]
        ]
        items = extractor._parse_line_items(tables)
        assert len(items) == 2

    def test_skip_summary_rows(self, extractor):
        tables = [
            [
                ["Description", "Qty", "Unit Price", "Total"],
                ["Widget A", "10", "$5.00", "$50.00"],
                ["Subtotal", "", "", "$50.00"],
            ]
        ]
        items = extractor._parse_line_items(tables)
        assert len(items) == 1

    def test_deduplication(self, extractor):
        tables = [
            [
                ["Description", "Qty", "Price"],
                ["Widget A", "10", "$5.00"],
                ["Widget A", "10", "$5.00"],
            ]
        ]
        items = extractor._parse_line_items(tables)
        assert len(items) == 1

    def test_empty_tables(self, extractor):
        assert extractor._parse_line_items([]) == []

    def test_no_qty_or_amount_table(self, extractor):
        tables = [
            [
                ["Name", "Category"],
                ["Widget A", "Hardware"],
            ]
        ]
        items = extractor._parse_line_items(tables)
        assert len(items) == 0
