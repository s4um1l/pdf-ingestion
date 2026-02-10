"""Tests for the Pydantic invoice schema and validation."""

import pytest
from pydantic import ValidationError

from pdf_ingestion.schema import Invoice, LineItem, InvoiceStatus, validate_invoice


class TestInvoiceSchema:
    def test_valid_invoice(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=500.00,
        )
        assert inv.invoice_number == "INV-001"
        assert inv.status == InvoiceStatus.DRAFT

    def test_invalid_date_format(self):
        with pytest.raises(ValidationError):
            Invoice(
                invoice_number="INV-001",
                vendor_name="Test Corp",
                invoice_date="15-01-2026",  # Wrong format
                total_amount=500.00,
            )

    def test_negative_amount(self):
        with pytest.raises(ValidationError):
            Invoice(
                invoice_number="INV-001",
                vendor_name="Test Corp",
                invoice_date="2026-01-15",
                total_amount=-100.00,
            )

    def test_with_line_items(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=500.00,
            line_items=[
                LineItem(description="Widget", quantity=10, unit_price=50.00, amount=500.00)
            ],
        )
        assert len(inv.line_items) == 1

    def test_line_item_amount_validation(self):
        with pytest.raises(ValidationError):
            LineItem(description="Widget", quantity=10, unit_price=50.00, amount=999.99)

    def test_us_date_format_accepted(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="01/15/2026",  # MM/DD/YYYY - common US format
            total_amount=500.00,
        )
        assert inv.invoice_date == "2026-01-15"  # Normalized to ISO

    def test_us_date_due_date(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            due_date="02/15/2026",
            total_amount=500.00,
        )
        assert inv.due_date == "2026-02-15"

    def test_optional_fields(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=500.00,
            due_date="2026-02-15",
            subtotal=450.00,
            tax=50.00,
            po_number="PO-001",
        )
        assert inv.due_date == "2026-02-15"
        assert inv.po_number == "PO-001"


class TestValidation:
    def test_valid_invoice_passes(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=500.00,
            extraction_confidence=0.9,
        )
        result = validate_invoice(inv)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_low_confidence_warning(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=500.00,
            extraction_confidence=0.5,
        )
        result = validate_invoice(inv)
        assert result.valid is True
        assert any("confidence" in w.lower() for w in result.warnings)

    def test_due_before_invoice_warning(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-02-15",
            due_date="2026-01-15",
            total_amount=500.00,
            extraction_confidence=0.9,
        )
        result = validate_invoice(inv)
        assert any("due date" in w.lower() for w in result.warnings)

    def test_total_mismatch_warning(self):
        inv = Invoice(
            invoice_number="INV-001",
            vendor_name="Test Corp",
            invoice_date="2026-01-15",
            total_amount=600.00,
            subtotal=450.00,
            tax=50.00,
            extraction_confidence=0.9,
        )
        result = validate_invoice(inv)
        assert any("mismatch" in w.lower() for w in result.warnings)
