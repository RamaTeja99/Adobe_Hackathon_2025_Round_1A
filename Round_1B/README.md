# Overview
Round-1B consumes JSON outlines from Round-1A and ranks/Presents document sections relevant to a persona’s task, using a hybrid semantic and lexical ranking and extractive summarization.

## Project Structure
```bash
Round_1B/
├── input/                 # Contains challenge1b_input.json and outlines/
│   ├── challenge1b_input.json
│   └── outlines/          # JSON outlines from Round_1A here
├── output/                # Results JSON saved here
├── pdf_utils/             # Shared utilities (embedding, text extraction)
├── persona_miner/
│   ├── __init__.py
│   ├── section_indexer.py
│   ├── hybrid_ranker.py
│   ├── text_refiner.py
│   └── output_generator.py
├── analyze_collection.py  # Main entrypoint script
├── requirements.txt
├── Dockerfile
```
## Manual Docker Build and Run Instructions
### Build Docker Image
```bash
export DOCKER_BUILDKIT=1
docker build -t round1b-persona-miner .
```
## Run Docker Container
``` bash
docker run --rm \
  -v "$(pwd)/input":/app/input:ro \
  -v "$(pwd)/output":/app/output \
  round1b-persona-miner
```
Automatically uses /app/input/challenge1b_input.json as input

Outputs to /app/output/challenge1b_output.json

## Custom Input/Output Paths (Optional)
```bash
docker run --rm \
  -v "$(pwd)/input":/app/input:ro \
  -v "$(pwd)/output":/app/output \
  round1b-persona-miner \
  /app/input/challenge1b_input.json /app/output/challenge1b_output.json
```
# Output
Ranked and refined sections output in JSON file challenge1b_output.json in the output/ directory.
