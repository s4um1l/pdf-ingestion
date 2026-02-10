"""
Invoice Schema: Pydantic models for invoice validation.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class InvoiceStatus(str, Enum):
    """Invoice processing status."""

    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    POSTED = "posted"


class LineItem(BaseModel):
    """Individual line item on invoice."""

    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)

    @validator("amount")
    def validate_amount(cls, v, values):
        """Validate amount = quantity x unit_price."""
        if "quantity" in values and "unit_price" in values:
            expected = values["quantity"] * values["unit_price"]
            if abs(v - expected) > 0.01:
                raise ValueError(f"Amount mismatch: {v} != {expected}")
        return v


class Invoice(BaseModel):
    """Complete invoice model with validation."""

    # Required fields
    invoice_number: str = Field(
        ..., description="Unique invoice identifier", min_length=1
    )
    vendor_name: str = Field(
        ..., description="Name of vendor/supplier", min_length=2
    )
    invoice_date: str = Field(..., description="Invoice issue date (YYYY-MM-DD)")
    total_amount: float = Field(..., gt=0, description="Total amount due")

    # Optional fields
    due_date: Optional[str] = Field(
        None, description="Payment due date (YYYY-MM-DD)"
    )
    subtotal: Optional[float] = Field(None, ge=0, description="Subtotal before tax")
    tax: Optional[float] = Field(None, ge=0, description="Tax amount")
    po_number: Optional[str] = Field(None, description="Purchase order number")

    # Metadata
    status: InvoiceStatus = InvoiceStatus.DRAFT
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Line items
    line_items: List[LineItem] = Field(default_factory=list)

    # Provenance
    source_file: Optional[str] = None
    extraction_method: Optional[str] = None

    @validator("invoice_date", "due_date")
    def validate_date_format(cls, v):
        """Validate and normalize date format to YYYY-MM-DD."""
        if v is None:
            return v

        # Accept multiple common date formats and normalize to YYYY-MM-DD
        formats = [
            "%Y-%m-%d",      # 2026-01-15
            "%m/%d/%Y",      # 01/15/2026
            "%m-%d-%Y",      # 01-15-2026
            "%d/%m/%Y",      # 15/01/2026 (handled by trying m/d first)
            "%m/%d/%y",      # 01/15/26
            "%B %d, %Y",    # January 15, 2026
            "%b %d, %Y",    # Jan 15, 2026
            "%d %B %Y",     # 15 January 2026
            "%d %b %Y",     # 15 Jan 2026
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(v, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD or MM/DD/YYYY")

    @validator("total_amount")
    def validate_total(cls, v, values):
        """Validate total = subtotal + tax (if both present)."""
        if "subtotal" in values and "tax" in values:
            if values["subtotal"] is not None and values["tax"] is not None:
                expected = values["subtotal"] + values["tax"]
                if abs(v - expected) > 0.01:
                    pass  # Allow mismatch, validated separately
        return v

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
                "status": "draft",
                "extraction_confidence": 0.95,
                "source_file": "invoice_12345.pdf",
                "extraction_method": "pdf_extract",
            }
        }


class InvoiceValidationResult(BaseModel):
    """Result of invoice validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    def add_error(self, error: str):
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)


def validate_invoice(invoice: Invoice) -> InvoiceValidationResult:
    """
    Comprehensive invoice validation.

    Returns:
        InvoiceValidationResult with errors, warnings, and confidence
    """
    result = InvoiceValidationResult(
        valid=True, confidence=invoice.extraction_confidence
    )

    if not invoice.invoice_number:
        result.add_error("Invoice number is required")

    if not invoice.vendor_name:
        result.add_error("Vendor name is required")

    if invoice.total_amount <= 0:
        result.add_error("Total amount must be positive")

    if invoice.due_date and invoice.invoice_date:
        inv_date = datetime.strptime(invoice.invoice_date, "%Y-%m-%d")
        due_date = datetime.strptime(invoice.due_date, "%Y-%m-%d")
        if due_date < inv_date:
            result.add_warning("Due date is before invoice date")

    if invoice.subtotal and invoice.tax:
        expected_total = invoice.subtotal + invoice.tax
        if abs(invoice.total_amount - expected_total) > 0.01:
            result.add_warning(
                f"Total amount mismatch: {invoice.total_amount} != {expected_total}"
            )

    if invoice.extraction_confidence < 0.7:
        result.add_warning("Low extraction confidence - manual review recommended")

    return result
