#!/bin/bash
# Start V-JEPA server on DGX Spark cluster
#
# This script configures and starts the V-JEPA inference server
# on both DGX Spark nodes connected via NVLink.
#
# Usage:
#   ./start_cluster.sh [--model MODEL] [--port PORT]
#
# Environment variables:
#   VJEPA_MODEL: Model name (default: facebook/vjepa2-vitl-fpc64-256)
#   VJEPA_PORT: Server port (default: 8001)
#   VJEPA_HOST: Bind address (default: 0.0.0.0)
#   VJEPA_DTYPE: Torch dtype (default: float16)

set -e

# Default configuration
VJEPA_MODEL="${VJEPA_MODEL:-facebook/vjepa2-vitl-fpc64-256}"
VJEPA_PORT="${VJEPA_PORT:-8001}"
VJEPA_HOST="${VJEPA_HOST:-0.0.0.0}"
VJEPA_DTYPE="${VJEPA_DTYPE:-float16}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            VJEPA_MODEL="$2"
            shift 2
            ;;
        --port)
            VJEPA_PORT="$2"
            shift 2
            ;;
        --host)
            VJEPA_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== V-JEPA Server Startup ==="
echo "Model: $VJEPA_MODEL"
echo "Host: $VJEPA_HOST"
echo "Port: $VJEPA_PORT"
echo "Dtype: $VJEPA_DTYPE"
echo ""

# Check CUDA availability
echo "Checking CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
echo ""

# Check NVLink topology (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Topology:"
    nvidia-smi topo -m 2>/dev/null || echo "  (topology info not available)"
    echo ""

    echo "GPU Memory:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Export environment variables
export VJEPA_MODEL
export VJEPA_PORT
export VJEPA_HOST
export VJEPA_DTYPE
export VJEPA_DEVICE_MAP="auto"

# Enable memory-efficient attention if available
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Start server
echo "Starting V-JEPA server..."
exec uvicorn vjepa_server.server:app \
    --host "$VJEPA_HOST" \
    --port "$VJEPA_PORT" \
    --log-level info
