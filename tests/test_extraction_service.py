"""Tests for the ExtractionService multi-strategy normalization."""

import pytest

from pdf_ingestion.extraction_service import (
    _safe_float,
    _looks_like_invoice_number,
    _looks_like_date,
    _looks_like_amount,
    _detect_column_role,
    ColumnRole,
    _extract_invoice_number,
    _extract_vendor_name,
    ExtractionStrategy,
    normalize_extraction_result,
)


class TestSafeFloat:
    def test_basic_float(self):
        assert _safe_float("123.45") == 123.45

    def test_currency(self):
        assert _safe_float("$1,234.56") == 1234.56

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_empty(self):
        assert _safe_float("") == 0.0

    def test_dash(self):
        assert _safe_float("-") == 0.0

    def test_default(self):
        assert _safe_float("bad", default=99.0) == 99.0

    def test_negative_parens(self):
        assert _safe_float("($50.00)") == -50.00


class TestPatternDetection:
    def test_invoice_number_patterns(self):
        assert _looks_like_invoice_number("INV-12345") is True
        assert _looks_like_invoice_number("INVOICE-001") is True
        assert _looks_like_invoice_number("12345678") is True
        assert _looks_like_invoice_number("#12345") is True
        assert _looks_like_invoice_number("Hello World") is False

    def test_date_patterns(self):
        assert _looks_like_date("01/15/2026") is True
        assert _looks_like_date("2026-01-15") is True
        assert _looks_like_date("Jan 15") is True
        assert _looks_like_date("Hello") is False

    def test_amount_patterns(self):
        assert _looks_like_amount("$1,234.56") is True
        assert _looks_like_amount("1234.56") is True
        assert _looks_like_amount("Hello") is False


class TestColumnRoleDetection:
    def test_description_header(self):
        assert _detect_column_role("Description", []) == ColumnRole.DESCRIPTION

    def test_qty_header(self):
        assert _detect_column_role("Qty", []) == ColumnRole.QUANTITY

    def test_price_header(self):
        assert _detect_column_role("Price", []) == ColumnRole.UNIT_PRICE

    def test_total_header(self):
        assert _detect_column_role("Amount", []) == ColumnRole.LINE_TOTAL

    def test_value_based_quantity(self):
        role = _detect_column_role("Col1", ["1", "5", "10", "2"])
        assert role == ColumnRole.QUANTITY

    def test_value_based_large_total(self):
        role = _detect_column_role("Col1", ["1500.00", "2300.00", "9999.00"])
        assert role == ColumnRole.LINE_TOTAL


class TestInvoiceNumberExtraction:
    def test_labeled_field(self):
        result = _extract_invoice_number(
            "raw text",
            {"invoice_number": "INV-001"},
            type("R", (), {"invoice_number": None})(),
        )
        assert result.value == "INV-001"
        assert result.strategy == ExtractionStrategy.LABELED

    def test_pattern_match(self):
        result = _extract_invoice_number(
            "INVOICE INV-2026-000001\nMore text",
            {},
            type("R", (), {})(),
        )
        assert "INV-2026-000001" in result.value
        assert result.strategy == ExtractionStrategy.PATTERN

    def test_not_found(self):
        result = _extract_invoice_number(
            "no invoice here",
            {},
            type("R", (), {})(),
        )
        assert result.strategy == ExtractionStrategy.MISSING


class TestVendorNameExtraction:
    def test_labeled_field(self):
        result = _extract_vendor_name(
            "raw text",
            {"vendor_name": "Springfield Energy LLC"},
            type("R", (), {"vendor_name": None})(),
        )
        assert result.value == "Springfield Energy LLC"
        assert result.strategy == ExtractionStrategy.LABELED

    def test_company_suffix_pattern(self):
        result = _extract_vendor_name(
            "Global Services LLC\n123 Main St",
            {},
            type("R", (), {})(),
        )
        assert "Global Services LLC" in result.value

    def test_not_found(self):
        result = _extract_vendor_name(
            "123 456 789",
            {},
            type("R", (), {})(),
        )
        assert result.strategy in (ExtractionStrategy.MISSING, ExtractionStrategy.HEURISTIC)


class TestNormalization:
    def test_normalize_basic_result(self):
        mock_result = type(
            "MockResult",
            (),
            {
                "raw_text": "INVOICE INV-2026-001\nVendor:\nTest Corp LLC\nDate: 2026-01-15\nTotal: $500.00",
                "extracted_fields": {},
                "line_items": [],
                "tables": [],
                "vendor_name": None,
                "invoice_number": None,
                "vendor_address": None,
                "invoice_date": None,
                "due_date": None,
                "po_number": None,
                "subtotal": None,
                "tax": None,
                "total": 500.0,
                "currency": "USD",
                "extraction_time_ms": 100,
                "source": "docling",
                "discount": None,
                "shipping": None,
            },
        )()

        result = normalize_extraction_result(mock_result)
        assert result["success"] is True
        assert "invoice_number" in result
        assert "vendor_name" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], dict)
