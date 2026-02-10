"""Tests for the FieldExtractor regex-based extraction."""

import pytest

from pdf_ingestion.field_extractor import FieldExtractor, ExtractedField


class TestFieldExtractor:
    @pytest.fixture
    def extractor(self):
        return FieldExtractor()

    def test_extract_invoice_number(self, extractor):
        text = "Invoice #: INV-12345\nDate: 01/15/2026"
        fields = extractor.extract_fields(text, ["invoice_number"])
        assert "invoice_number" in fields
        assert fields["invoice_number"].value == "INV-12345"

    def test_extract_total_amount(self, extractor):
        text = "Total: $1,234.56\nDue Date: 02/15/2026"
        fields = extractor.extract_fields(text, ["total_amount"])
        assert "total_amount" in fields
        assert fields["total_amount"].value == 1234.56

    def test_extract_date(self, extractor):
        text = "Invoice Date: 01/15/2026\nAmount: $500.00"
        fields = extractor.extract_fields(text, ["invoice_date"])
        assert "invoice_date" in fields
        assert fields["invoice_date"].value == "2026-01-15"

    def test_extract_vendor(self, extractor):
        text = "From: Acme Corp\nInvoice #: 12345"
        fields = extractor.extract_fields(text, ["vendor_name"])
        assert "vendor_name" in fields
        assert "Acme" in fields["vendor_name"].value

    def test_extract_all_default_fields(self, extractor):
        text = """
        Invoice #: INV-001
        Date: 01/15/2026
        From: Test Vendor
        Total: $500.00
        """
        fields = extractor.extract_fields(text)
        assert len(fields) >= 3  # At least invoice_number, total, date

    def test_confidence_scoring(self, extractor):
        text = "Invoice Date: 01/15/2026"
        fields = extractor.extract_fields(text, ["invoice_date"])
        assert fields["invoice_date"].confidence >= 0.5

    def test_missing_field_not_in_result(self, extractor):
        text = "Some random text with no invoice data"
        fields = extractor.extract_fields(text, ["po_number"])
        assert "po_number" not in fields

    def test_validation(self, extractor):
        text = "Invoice #: INV-001\nTotal: $500.00"
        fields = extractor.extract_fields(text, ["invoice_number", "total_amount"])
        validation = extractor.validate_extraction(fields)
        assert validation.get("invoice_number") is True
        assert validation.get("total_amount") is True

    def test_extraction_count(self, extractor):
        text = "Invoice #: INV-001"
        extractor.extract_fields(text)
        extractor.extract_fields(text)
        assert extractor.extraction_count == 2


class TestF1Score:
    @pytest.fixture
    def extractor(self):
        return FieldExtractor()

    def test_perfect_match(self, extractor):
        text = "Invoice #: INV-001\nTotal: $500.00"
        extracted = extractor.extract_fields(text, ["invoice_number", "total_amount"])
        ground_truth = {"invoice_number": "INV-001", "total_amount": 500.00}
        f1, details = extractor.compute_extraction_f1(extracted, ground_truth)
        assert f1 == 1.0

    def test_empty_extraction(self, extractor):
        f1, details = extractor.compute_extraction_f1({}, {"invoice_number": "INV-001"})
        assert f1 == 0.0

    def test_empty_ground_truth(self, extractor):
        f1, details = extractor.compute_extraction_f1({}, {})
        assert f1 == 0.0
