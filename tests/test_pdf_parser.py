"""Tests for the lightweight PDFParser."""

import pytest
from pdf_ingestion.pdf_parser import PDFParser, PDFInvoice, ExtractionMetrics


class TestPDFParser:
    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse_invoice("/nonexistent/file.pdf")

    def test_parse_returns_pdf_invoice(self, parser, sample_invoice_pdf):
        result = parser.parse_invoice(str(sample_invoice_pdf))
        assert isinstance(result, PDFInvoice)

    def test_parse_has_text(self, parser, sample_invoice_pdf):
        result = parser.parse_invoice(str(sample_invoice_pdf))
        assert len(result.raw_text) > 0

    def test_parse_has_metadata(self, parser, sample_invoice_pdf):
        result = parser.parse_invoice(str(sample_invoice_pdf))
        assert "producer" in result.metadata

    def test_parse_count_increments(self, parser, sample_invoice_pdf):
        parser.parse_invoice(str(sample_invoice_pdf))
        parser.parse_invoice(str(sample_invoice_pdf))
        assert parser.parse_count == 2

    def test_regions(self, parser, sample_invoice_pdf):
        regions = parser.extract_invoice_regions(str(sample_invoice_pdf))
        assert "header" in regions
        assert "line_items" in regions
        assert "totals" in regions

    def test_structure_validation(self, parser, sample_invoice_pdf):
        result = parser.validate_invoice_structure(str(sample_invoice_pdf))
        assert isinstance(result, dict)
        assert "has_invoice_number" in result
        assert "has_total" in result


class TestExtractionMetrics:
    def test_record_and_get(self):
        m = ExtractionMetrics()
        m.record_extraction(True, 0.9, 100.0)
        m.record_extraction(True, 0.8, 200.0)
        m.record_extraction(False, 0.3, 300.0)

        assert m.total_extractions == 3
        assert m.get_success_rate() == pytest.approx(2 / 3)
        assert m.get_average_confidence() == pytest.approx(2.0 / 3)

    def test_empty_metrics(self):
        m = ExtractionMetrics()
        assert m.get_success_rate() == 0.0
        assert m.get_average_confidence() == 0.0
        assert m.get_p95_parse_time() == 0.0
