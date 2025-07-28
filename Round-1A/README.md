# Overview
Round-1A extracts document outlines (headings) from PDFs, generating JSON files per document that describe the hierarchical structure of the PDF content, helping downstream tasks understanding document sections.

## Project Structure
```bash
Round_1A/
├── input/                 # Place your input PDF files here
├── output/                # JSON outline files are saved here
├── pdf_utils/             # Shared PDF processing utilities
│   ├── __init__.py
│   ├── reader.py
│   ├── heading_detect.py
│   ├── text_extract.py
│   └── embedding_utils.py
├── process_pdfs.py        # Main script to extract outlines
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image build configuration
├── output_schema.json     # JSON schema for validation (optional)
```
# Setup
 Make sure your input PDFs are placed inside Round_1A/input/ directory.

 Manual Docker Build and Run Instructions
## Build Docker Image

### Open a terminal in the Round_1A directory, then run:

```bash
export DOCKER_BUILDKIT=1
docker build -t round1a-outline-extractor .
```
## Run Docker Container

### To process all PDFs from input/ and save JSON outlines in output/, run:

```bash
docker run --rm \
  -v "$(pwd)/input":/app/input:ro \
  -v "$(pwd)/output":/app/output \
  round1a-outline-extractor \
  /app/input /app/output
```
## Replace /app/input and /app/output if you mount custom paths.

### Process a Single PDF

### If you want to process a single PDF file:

```bash
docker run --rm \
  -v "$(pwd)/input":/app/input:ro \
  -v "$(pwd)/output":/app/output \
  round1a-outline-extractor \
  /app/input/yourfile.pdf /app/output/yourfile.json
```
## Outputs
The container writes JSON files to the mounted output folder named after each PDF, e.g.: sample_document.pdf → sample_document.json.

The JSON structure contains:

title: extracted document title
 outline: list of headings with their levels and page numbers
