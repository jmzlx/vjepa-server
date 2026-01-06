"""FastAPI server for V-JEPA 2 inference.

This server exposes V-JEPA 2 model for visual analysis of ATC situations.
Designed to run on DGX Spark cluster with NVLink-connected GPUs.

Usage:
    uvicorn vjepa_server.server:app --host 0.0.0.0 --port 8001
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import ModelConfig, VJEPAModel
from .schemas import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    VideoRequest,
)

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model instance
model: VJEPAModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, unload on shutdown."""
    global model

    # Load model on startup
    logger.info("Starting V-JEPA server...")

    config = ModelConfig(
        model_name=os.getenv("VJEPA_MODEL", "facebook/vjepa2-vitl-fpc64-256"),
        torch_dtype=os.getenv("VJEPA_DTYPE", "float16"),
        device_map=os.getenv("VJEPA_DEVICE_MAP", "auto"),
    )

    model = VJEPAModel(config)

    try:
        model.load()
        logger.info("V-JEPA model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue running - health endpoint will report model not loaded

    yield

    # Cleanup on shutdown
    if model is not None:
        model.unload()
    logger.info("V-JEPA server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="V-JEPA Server",
    description="V-JEPA 2 inference server for OpenScope ATC visual analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests from agent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health and model status."""
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    memory_allocated = 0.0

    if device_count > 0:
        for i in range(device_count):
            memory_allocated += torch.cuda.memory_allocated(i) / 1e9

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and model.is_loaded,
        device_count=device_count,
        model_name=model.config.model_name if model else "",
        memory_allocated_gb=memory_allocated,
    )


@app.post("/analyze", response_model=AnalysisResponse, responses={500: {"model": ErrorResponse}})
async def analyze(request: VideoRequest) -> AnalysisResponse:
    """Analyze video frames with V-JEPA.

    Expects video tensor of shape (T, C, H, W) where:
    - T = number of frames (8-16 recommended)
    - C = 3 (RGB channels)
    - H, W = 224, 224

    Returns embeddings that capture visual understanding of the scene.
    """
    if model is None or not model.is_loaded:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check server logs.",
        )

    try:
        result = model.encode_from_array(
            frames=request.frames,
            return_pooled=request.return_pooled,
        )

        return AnalysisResponse(
            embeddings=result["embeddings"],
            embedding_dim=result["embedding_dim"],
            num_frames=result["num_frames"],
            inference_time_ms=result["inference_time_ms"],
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@app.get("/memory", response_model=dict)
async def memory_stats() -> dict:
    """Get GPU memory statistics."""
    if model is None:
        return {"error": "Model not initialized"}
    return model.get_memory_stats()


def main():
    """Run the server."""
    import uvicorn

    host = os.getenv("VJEPA_HOST", "0.0.0.0")
    port = int(os.getenv("VJEPA_PORT", "8001"))

    logger.info(f"Starting V-JEPA server on {host}:{port}")

    uvicorn.run(
        "vjepa_server.server:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
