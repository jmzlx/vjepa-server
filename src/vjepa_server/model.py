"""V-JEPA 2 model wrapper with NVLink distribution support.

This module handles loading V-JEPA 2 and distributing it across
multiple GPUs connected via NVLink for optimal inference performance.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for V-JEPA model loading."""

    model_name: str = "facebook/vjepa2-vitl-fpc64-256"
    torch_dtype: str = "float16"
    device_map: str = "auto"  # Automatically distribute across NVLink GPUs
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True


class VJEPAModel:
    """Wrapper for V-JEPA 2 model with multi-GPU support.

    The model is loaded with device_map="auto" which leverages the
    accelerate library to distribute model weights across available
    GPUs connected via NVLink.

    Attributes:
        config: Model configuration
        model: The loaded V-JEPA model
        processor: Image/video processor for the model
        device: Primary device for inference
    """

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the model wrapper.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self.model = None
        self.processor = None
        self.device = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load V-JEPA 2 model with automatic multi-GPU distribution.

        Uses accelerate's device_map="auto" to distribute the model
        across all available GPUs via NVLink.
        """
        if self._is_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading V-JEPA model: {self.config.model_name}")
        start_time = time.time()

        try:
            from transformers import AutoModel, AutoProcessor

            # Determine torch dtype
            dtype = getattr(torch, self.config.torch_dtype)

            # Load model with automatic device mapping
            # This distributes layers across NVLink-connected GPUs
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Load processor for input normalization
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Set model to evaluation mode (PyTorch's .eval() method)
            # This disables dropout and batch norm tracking for inference
            self.model.train(False)

            # Determine primary device
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")

            self._is_loaded = True
            load_time = time.time() - start_time

            logger.info(
                f"Model loaded in {load_time:.2f}s, "
                f"device_map: {self.config.device_map}, "
                f"dtype: {self.config.torch_dtype}"
            )

            # Log device distribution
            self._log_device_distribution()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _log_device_distribution(self) -> None:
        """Log how the model is distributed across devices."""
        if not torch.cuda.is_available():
            return

        device_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(
                f"  GPU {i}: {props.name}, "
                f"Memory: {props.total_memory / 1e9:.1f}GB total, "
                f"{allocated:.2f}GB allocated"
            )

    def encode(self, video_tensor: torch.Tensor, return_pooled: bool = True) -> torch.Tensor:
        """Encode video frames to V-JEPA embeddings.

        Args:
            video_tensor: Input tensor of shape (T, C, H, W) or (B, T, C, H, W)
            return_pooled: If True, return mean-pooled embedding across frames

        Returns:
            Embeddings tensor of shape (embedding_dim,) if pooled,
            or (T, embedding_dim) if not pooled
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure 5D input: (B, T, C, H, W)
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension

        # Move to appropriate device and dtype
        video_tensor = video_tensor.to(self.device, dtype=self.model.dtype)

        with torch.no_grad():
            # Get encoder outputs
            # V-JEPA returns hidden states from vision transformer
            outputs = self.model(video_tensor, return_dict=True)

            # Extract embeddings from last hidden state
            # Shape: (B, T, num_patches, hidden_dim) or similar
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                # Fallback: use first output
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs

            # Pool across spatial dimensions if needed
            if embeddings.dim() > 2:
                # Global average pooling over spatial/temporal dimensions
                embeddings = embeddings.mean(dim=list(range(1, embeddings.dim() - 1)))

            # Squeeze batch dimension for single video
            if embeddings.size(0) == 1:
                embeddings = embeddings.squeeze(0)

        return embeddings

    def encode_from_array(
        self,
        frames: np.ndarray | list,
        return_pooled: bool = True,
    ) -> dict[str, Any]:
        """Encode video frames from numpy array or list.

        Args:
            frames: Video frames as numpy array (T, C, H, W) or nested list
            return_pooled: If True, return mean-pooled embedding

        Returns:
            Dict with embeddings and metadata
        """
        start_time = time.time()

        # Convert to tensor
        if isinstance(frames, list):
            frames = np.array(frames, dtype=np.float32)
        tensor = torch.from_numpy(frames)

        # Encode
        embeddings = self.encode(tensor, return_pooled=return_pooled)

        # Convert back to numpy
        embeddings_np = embeddings.cpu().numpy()

        inference_time = (time.time() - start_time) * 1000

        return {
            "embeddings": embeddings_np.tolist(),
            "embedding_dim": embeddings_np.shape[-1],
            "num_frames": frames.shape[0],
            "inference_time_ms": inference_time,
        }

    def get_memory_stats(self) -> dict[str, float]:
        """Get GPU memory statistics."""
        stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                stats[f"gpu_{i}_allocated_gb"] = allocated
        return stats

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        logger.info("Model unloaded")
