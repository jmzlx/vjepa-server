"""Pydantic schemas for V-JEPA server API."""

from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    """Request containing video frames for V-JEPA analysis.

    V-JEPA expects video tensors of shape (T, C, H, W) where:
    - T = number of frames (typically 8-16)
    - C = 3 (RGB channels)
    - H, W = 224, 224 (standard ViT input size)

    Frames should be normalized to [0, 1] range.
    """

    frames: list[list[list[list[float]]]] = Field(
        ...,
        description="Video frames as nested list: [T][C][H][W], normalized to [0, 1]",
    )
    return_pooled: bool = Field(
        default=True,
        description="Return mean-pooled embedding instead of per-frame embeddings",
    )


class AnalysisResponse(BaseModel):
    """Response containing V-JEPA embeddings."""

    embeddings: list[float] = Field(
        ...,
        description="V-JEPA embeddings (1024-dim for ViT-L)",
    )
    embedding_dim: int = Field(
        ...,
        description="Dimension of embedding vector",
    )
    num_frames: int = Field(
        ...,
        description="Number of frames processed",
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    device_count: int = Field(default=0, description="Number of GPUs available")
    model_name: str = Field(default="", description="Loaded model name")
    memory_allocated_gb: float = Field(default=0.0, description="GPU memory allocated in GB")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
