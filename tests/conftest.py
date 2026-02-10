"""Shared fixtures for tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_invoice_pdf(tmp_path):
    """
    Generate a minimal but realistic invoice PDF for testing.

    Uses reportlab if available, otherwise creates a simple PDF with pypdf.
    """
    pdf_path = tmp_path / "test_invoice.pdf"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, height - 72, "INVOICE")

        c.setFont("Helvetica", 10)
        c.drawString(72, height - 100, "Invoice Number: INV-2026-000042")
        c.drawString(72, height - 115, "Date: 2026-01-15")
        c.drawString(72, height - 130, "Due: 2026-02-15")

        # Vendor
        c.setFont("Helvetica-Bold", 10)
        c.drawString(72, height - 160, "Vendor:")
        c.setFont("Helvetica", 10)
        c.drawString(72, height - 175, "Springfield Energy Services LLC")
        c.drawString(72, height - 190, "456 Power Ave, Springfield, IL 62701")

        # Bill To
        c.drawString(350, height - 160, "Bill To:")
        c.drawString(350, height - 175, "Demo Company Inc")
        c.drawString(350, height - 190, "789 Corporate Blvd, Suite 100")

        # PO Reference
        c.drawString(72, height - 220, "PO Reference: PO-2026-000015")

        # Line items header
        y = height - 260
        c.setFont("Helvetica-Bold", 10)
        c.drawString(72, y, "Description")
        c.drawString(300, y, "Qty")
        c.drawString(360, y, "Unit Price")
        c.drawString(460, y, "Total")

        # Line items
        c.setFont("Helvetica", 10)
        y -= 20
        c.drawString(72, y, "Consulting Services - January 2026")
        c.drawString(300, y, "40")
        c.drawString(360, y, "$150.00")
        c.drawString(460, y, "$6,000.00")

        y -= 20
        c.drawString(72, y, "Software License - Annual")
        c.drawString(300, y, "1")
        c.drawString(360, y, "$2,400.00")
        c.drawString(460, y, "$2,400.00")

        # Totals
        y -= 40
        c.drawString(360, y, "Subtotal: $8,400.00")
        y -= 20
        c.drawString(360, y, "Tax: $672.00")
        y -= 20
        c.setFont("Helvetica-Bold", 10)
        c.drawString(360, y, "Total: $9,072.00")

        # Footer
        y -= 60
        c.setFont("Helvetica", 8)
        c.drawString(72, y, "Payment Terms: Net 30")
        c.drawString(72, y - 15, "Thank you for your business.")

        c.save()

    except ImportError:
        # Fallback: create a simple text-content PDF with pypdf
        from pypdf import PdfWriter
        from io import BytesIO

        writer = PdfWriter()
        # Unfortunately pypdf can't create pages with text content easily.
        # Create a minimal empty PDF. The extractor will still run,
        # just won't extract fields. Tests that need rich content
        # should install reportlab.
        writer.add_blank_page(width=612, height=792)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # Mark this as a minimal PDF
        pytest.skip("reportlab not installed - install it for full PDF tests")

    return pdf_path


@pytest.fixture
def sample_invoice_text():
    """Return sample invoice text for testing field extraction."""
    return """
INVOICE INV-2026-000042

Vendor:
Springfield Energy Services LLC
456 Power Ave
Springfield, IL 62701

Bill To:
Demo Company Inc
789 Corporate Blvd, Suite 100

Invoice Date: 2026-01-15
Due Date: 2026-02-15
PO Reference: PO-2026-000015

Description                    Qty    Unit Price    Total
Consulting Services            40     $150.00       $6,000.00
Software License               1      $2,400.00     $2,400.00

Subtotal: $8,400.00
Tax: $672.00
Total: $9,072.00

Payment Terms: Net 30
Thank you for your business.
"""
