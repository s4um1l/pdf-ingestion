# PDF Ingestion Service

Extract structured data from invoice PDFs and screenshots. Uses IBM Docling for layout-aware extraction with automatic fallback to pypdf + regex. Includes VLM (Vision-Language Model) integration via Ollama/Qwen-VL (free, local) or Claude for screenshot-based extraction.

## Features

- **Layout-aware extraction** via IBM Docling (DocLayNet models)
- **Table detection** for line item parsing
- **Multi-strategy field extraction**: labeled fields, regex patterns, heuristics
- **Automatic fallback** to pypdf when Docling isn't installed
- **Confidence scoring** per extracted field
- **Pydantic validation** for extracted invoices
- **Async support** for non-blocking extraction
- **VLM integration** - extract from screenshots using Qwen-VL (Ollama, free) or Claude
- **VLM Router** - auto-fallback between providers with cost tracking
- **No API keys required** for PDF extraction (runs entirely locally)
- **Ollama support** - free local VLM via Qwen2.5-VL

## Install

```bash
# Core (pypdf fallback only - no ML models needed)
pip install -e .

# With Docling for full layout-aware extraction (~2GB model download)
pip install -e ".[docling]"

# With VLM support (Ollama/Qwen-VL + Claude)
pip install -e ".[vlm]"

# Ollama only (free, no API keys)
pip install -e ".[ollama]"

# With test dependencies
pip install -e ".[test]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```python
from pdf_ingestion import extract_invoice

result = extract_invoice("path/to/invoice.pdf")

print(f"Vendor:    {result.vendor_name}")
print(f"Invoice #: {result.invoice_number}")
print(f"Date:      {result.invoice_date}")
print(f"Total:     ${result.total:,.2f}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Source:    {result.source}")  # "docling" or "pypdf_fallback"

# Line items (when tables are detected)
for item in result.line_items:
    print(f"  - {item}")
```

### Async Usage

```python
import asyncio
from pdf_ingestion import extract_invoice_async

async def main():
    result = await extract_invoice_async("invoice.pdf")
    print(result.vendor_name, result.total)

asyncio.run(main())
```

### Using the ExtractionService (base64 input)

```python
import asyncio
import base64
from pdf_ingestion import ExtractionService

async def main():
    service = ExtractionService()
    await service.initialize()

    with open("invoice.pdf", "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    result = await service.extract(content_b64, "invoice.pdf")
    print(result["vendor_name"])
    print(result["total"])
    print(result["confidence"])  # Per-field confidence dict

asyncio.run(main())
```

### Schema Validation

```python
from pdf_ingestion import Invoice, validate_invoice

invoice = Invoice(
    invoice_number="INV-001",
    vendor_name="Acme Corp",
    invoice_date="2026-01-15",
    total_amount=1234.56,
    extraction_confidence=0.92,
)

result = validate_invoice(invoice)
print(f"Valid: {result.valid}")
print(f"Warnings: {result.warnings}")
```

### Field Extraction from Raw Text

```python
from pdf_ingestion import FieldExtractor

extractor = FieldExtractor()
fields = extractor.extract_fields(raw_text)

for name, field in fields.items():
    print(f"{name}: {field.value} (confidence: {field.confidence:.0%})")
```

### VLM Extraction from Screenshots (Ollama/Qwen-VL)

Extract invoice fields from screenshots using a local VLM - completely free, no API keys needed.

```bash
# Setup Ollama (one-time)
brew install ollama && ollama serve
ollama pull qwen2.5vl:7b
```

```python
import asyncio
from pdf_ingestion.vlm import VLMRouter

async def main():
    router = VLMRouter(provider="qwen")  # Free, local

    # Extract from screenshot (base64 or file path)
    fields = await router.extract_invoice_fields(
        "/path/to/screenshot.png",
        ["invoice_number", "vendor_name", "total_amount", "invoice_date"]
    )
    print(fields)

asyncio.run(main())
```

### VLM with Auto-Fallback

```python
import asyncio
from pdf_ingestion.vlm import VLMRouter

async def main():
    # Auto mode: tries Qwen (free) first, falls back to Claude
    router = VLMRouter(provider="auto")

    # Find a UI element by description
    region = await router.ground_element_by_description(
        screenshot_b64,
        "the Save button"
    )
    if region:
        print(f"Found: {region.label} at {region.bounding_box} ({region.confidence:.0%})")

    # Check cost
    print(router.get_stats())

asyncio.run(main())
```

### VLM with Claude (highest quality)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from pdf_ingestion.vlm import VLMRouter

router = VLMRouter(provider="claude")
fields = await router.extract_invoice_fields(screenshot_b64, ["vendor_name", "total"])
```

## Extracted Fields

| Field | Type | Description |
|-------|------|-------------|
| `invoice_number` | str | Invoice/bill number |
| `vendor_name` | str | Supplier name |
| `vendor_address` | str | Supplier address |
| `invoice_date` | str | Issue date |
| `due_date` | str | Payment due date |
| `po_number` | str | Purchase order reference |
| `subtotal` | float | Pre-tax amount |
| `tax` | float | Tax amount |
| `total` | float | Total amount due |
| `currency` | str | Currency code (default: USD) |
| `line_items` | list | Parsed table rows |
| `confidence` | float | 0.0 - 1.0 extraction confidence |
| `source` | str | "docling" or "pypdf_fallback" |

## Run Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_extractor.py

# Run a specific test
pytest tests/test_extractor.py::TestExtractInvoiceFields::test_invoice_number_extraction
```

## Project Structure

```
pdf-ingestion/
  pdf_ingestion/
    __init__.py            # Public API
    extractor.py           # DoclingExtractor + pypdf fallback
    field_extractor.py     # Regex field extraction + F1 scoring
    pdf_parser.py          # Lightweight pypdf parser
    schema.py              # Pydantic Invoice models + validation
    extraction_service.py  # Async service + multi-strategy normalization
    vlm/                   # Vision-Language Model integration
      __init__.py          # VLM public API
      vlm_types.py         # Shared VLMRegion dataclass
      vlm_router.py        # Unified router (auto/qwen/claude)
      qwen_vlm_client.py   # Ollama/Qwen-VL client (free, local)
      claude_vlm_client.py # Claude VLM client (API, paid)
  tests/
    conftest.py            # Shared fixtures (sample PDF generation)
    test_extractor.py      # Core extractor tests
    test_field_extractor.py
    test_pdf_parser.py
    test_schema.py
    test_extraction_service.py
    test_vlm.py            # VLM router + client tests
  examples/
    sample_invoice.pdf     # Sample invoice for manual testing
  pyproject.toml
  README.md
```

## How It Works

1. **Docling path** (when installed): Converts PDF using DocLayNet models, extracts text by layout position, detects tables, then applies regex patterns to extract fields.

2. **Fallback path** (pypdf only): Extracts raw text with pypdf, applies the same regex patterns. No table detection. Confidence is reduced by 30%.

3. **Field extraction** uses multiple strategies in priority order:
   - **Labeled**: Explicit field labels ("Vendor:", "Invoice #:")
   - **Pattern**: Regex pattern matching on raw text
   - **Heuristic**: Position-based inference (company name suffixes, first non-date line, etc.)

4. **Confidence scoring**: Each field contributes to overall confidence. Invoice number (30%), total (25%), date (15%), vendor (15%), PO number (10%), line items (5%).

5. **VLM path** (screenshot extraction): Sends screenshot to a Vision-Language Model (Qwen-VL via Ollama or Claude) with structured prompts. The model returns JSON with extracted fields, bounding boxes, and confidence scores. Best for extracting from filled form screenshots rather than raw PDFs.

## VLM Provider Comparison

| Provider | Cost | Speed | Setup | Best For |
|----------|------|-------|-------|----------|
| **Qwen-VL** (Ollama) | $0.00 | 3-10s | `ollama pull qwen2.5vl:7b` | Local dev, free usage |
| **Claude** (API) | ~$0.01/call | 2-3s | API key only | High quality, production |
| **Auto** | varies | varies | both | Best effort fallback |
