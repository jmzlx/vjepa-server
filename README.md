# V-JEPA Server

V-JEPA 2 inference server for OpenScope ATC visual analysis.

## Installation

```bash
uv sync
```

## Usage

```bash
# Start server locally (Mac with MPS)
VJEPA_DEVICE_MAP=mps uv run uvicorn vjepa_server.server:app --host 127.0.0.1 --port 8001

# Start server on GPU cluster
./scripts/start_cluster.sh
```

## API

- `POST /analyze` - Analyze video frames, returns embeddings
- `GET /health` - Health check and model status
- `GET /memory` - GPU memory statistics

## Configuration

Environment variables:
- `VJEPA_MODEL` - Model name (default: `facebook/vjepa2-vitl-fpc64-256`)
- `VJEPA_DEVICE_MAP` - Device map (`auto`, `cuda`, `mps`, `cpu`)
- `VJEPA_DTYPE` - Torch dtype (`float16`, `float32`)
- `VJEPA_HOST` - Server bind address
- `VJEPA_PORT` - Server port
