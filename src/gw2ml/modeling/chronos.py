"""Chronos Foundation Model for Time Series Forecasting."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from darts import TimeSeries
from darts.models import Chronos2Model
from .base import BaseModel
from gw2ml.utils import get_logger

logger = get_logger("modeling.chronos")


def _get_device() -> str:
    """Determine the best available device for inference.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    # ROCm uses CUDA API through HIP
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "cuda"
    return "cpu"


def _get_accelerator() -> str:
    """Get the PyTorch Lightning accelerator string.

    Returns:
        Accelerator string for pl_trainer_kwargs
    """
    device = _get_device()
    if device == "cuda":
        return "gpu"
    elif device == "mps":
        return "mps"
    return "cpu"


class Chronos2(BaseModel):
    """
    Chronos Foundation Model by AutoGluon/Amazon

    This model supports CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon), and CPU.
    The model is automatically moved to the best available device after initialization.

    Args:
        input_chunk_length: Number of past values to use (default: 168)
        output_chunk_length: Number of future values to predict (default: 168)
        model_hub_name: HuggingFace model name. Options:
            - 'autogluon/chronos-2-synth' (120M params, default)
            - 'autogluon/chronos-2-small' (28M params, faster/less memory)
        batch_size: Batch size for inference (default: 32)
        use_mixed_precision: Use 16-bit mixed precision for less memory (default: True)
        device: Force specific device ('cuda', 'mps', 'cpu', or None for auto)
    """

    @property
    def name(self) -> str:
        return "Chronos2"

    @property
    def default_params(self) -> Dict[str, Any]:
        # chronos-2-synth (120M) is more accurate but uses more memory
        # chronos-2-small (28M) is faster and uses less memory
        return {"model_hub_name": "autogluon/chronos-2-synth"}

    def __init__(
            self,
            input_chunk_length: int = 168,
            output_chunk_length: int = 168,
            batch_size: int = 32,
            use_mixed_precision: bool = True,
            device: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        # Merge default params with provided kwargs
        self.params: Dict[str, Any] = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "batch_size": batch_size,
            **self.default_params,
            **kwargs,
        }
        self._model: Optional[Chronos2Model] = None
        self._device = device or _get_device()
        self._use_mixed_precision = use_mixed_precision
        self._model_on_device = False

        # Optimize for Tensor Cores on NVIDIA GPUs
        if self._device == "cuda" and torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')

    def build_model(self, **kwargs: Any) -> Chronos2Model:
        build_params = {**self.params, **kwargs}

        input_chunk_length = build_params.pop("input_chunk_length")
        output_chunk_length = build_params.pop("output_chunk_length")
        batch_size = build_params.pop("batch_size", 32)

        # Remove parameters not accepted by Chronos2Model
        build_params.pop("model", None)
        build_params.pop("epochs", None)
        model_hub_name = build_params.pop("model_hub_name", "autogluon/chronos-2-synth")

        # Configure PyTorch Lightning trainer for GPU support
        if "pl_trainer_kwargs" not in build_params:
            build_params["pl_trainer_kwargs"] = {}

        pl_kwargs = build_params["pl_trainer_kwargs"]

        # Set accelerator based on available hardware
        accelerator = _get_accelerator()
        if "accelerator" not in pl_kwargs:
            pl_kwargs["accelerator"] = accelerator

        # Configure devices
        if accelerator in ("gpu", "mps") and "devices" not in pl_kwargs:
            pl_kwargs["devices"] = 1

        # Enable mixed precision for memory savings (CUDA and MPS)
        if self._use_mixed_precision and "precision" not in pl_kwargs:
            if accelerator == "gpu":
                pl_kwargs["precision"] = "16-mixed"
            # Note: MPS has limited precision support, skip for now

        # Disable progress bars to avoid log spam
        if "enable_progress_bar" not in pl_kwargs:
            pl_kwargs["enable_progress_bar"] = False

        logger.info(
            f"Building Chronos2Model: hub={model_hub_name}, "
            f"device={self._device}, accelerator={accelerator}, "
            f"batch_size={batch_size}"
        )

        return Chronos2Model(
            model_name=model_hub_name,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            batch_size=batch_size,
            **build_params
        )

    def _move_to_device(self) -> None:
        """Move the model to the configured device after initialization."""
        if self._model is None or self._model_on_device:
            return

        if self._device != "cpu" and hasattr(self._model, 'model'):
            try:
                self._model.model = self._model.model.to(self._device)
                self._model_on_device = True
                logger.info(f"Moved Chronos2 model to {self._device}")
            except Exception as e:
                logger.warning(f"Failed to move model to {self._device}: {e}")
                self._device = "cpu"

    def cleanup(self) -> None:
        """Release GPU memory and cleanup resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_on_device = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Chronos2 cleanup completed")

    def fit(self, series: TimeSeries, **kwargs: Any) -> Chronos2:
        """Fit the Chronos2 model (zero-shot mode by default).

        For foundation models like Chronos2, this initializes the model and
        prepares it for inference. The model is automatically moved to the
        best available device (GPU/MPS/CPU) after initialization.

        Args:
            series: Time series data
            **kwargs: Additional arguments passed to the underlying model

        Returns:
            self for method chaining
        """
        # Rebuild model to ensure a fresh start or consistent state,
        # matching the behavior of other models in the pipeline.
        self._model_on_device = False

        # MPS and CUDA work best with float32. Ensure series is float32.
        if series.dtype == np.float64:
            logger.debug("Casting series to float32 for Chronos2 (GPU compatibility)")
            series = series.astype(np.float32)

        self._model = self.build_model()
        logger.info(f"Fitting Chronos2 (n={series.shape[0]}, device={self._device})")

        # Chronos is a foundation model. 'fit' initializes internal state.
        # epochs=0 means zero-shot (no fine-tuning).
        epochs = kwargs.pop("epochs", 0)
        verbose = kwargs.pop("verbose", False)

        self._model.fit(series, epochs=epochs, verbose=verbose, **kwargs)

        # CRITICAL: Move model to GPU after fit() with epochs=0
        # PyTorch Lightning only moves the model to GPU during training,
        # but with epochs=0 no training happens, so model stays on CPU.
        self._move_to_device()

        self._context_series = series
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        """Generate predictions for n steps ahead.

        Args:
            n: Number of steps to predict
            **kwargs: Additional arguments (e.g., series to predict from)

        Returns:
            TimeSeries with predictions
        """
        if self._model is None:
            self._model = self.build_model()
            self._move_to_device()

        series = kwargs.pop("series", getattr(self, "_context_series", None))
        if series is not None and series.dtype == np.float64:
            logger.debug("Casting series to float32 for Chronos2 predict")
            series = series.astype(np.float32)

        logger.debug(f"Generating prediction with Chronos2 (n={n}, device={self._device})")
        return self._model.predict(n=n, series=series, **kwargs)

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[float, int],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> TimeSeries:
        """Walk-forward backtesting.

        For foundation models, retrain=False is recommended as retraining
        at each step is computationally expensive and unnecessary.

        Args:
            series: Full time series for backtesting
            start: Where to start (ratio 0.0-1.0 or absolute index)
            forecast_horizon: Steps ahead to forecast at each point
            stride: Steps between forecasts (default: 1)
            retrain: Whether to retrain at each step (default: False)
            verbose: Print progress (default: False)
            **kwargs: Additional arguments

        Returns:
            TimeSeries with historical forecasts
        """
        if self._model is None:
            self._model = self.build_model()
            self._move_to_device()

        if series.dtype == np.float64:
            logger.debug("Casting series to float32 for Chronos2 historical_forecasts")
            series = series.astype(np.float32)

        logger.info(
            f"Computing historical forecasts with Chronos2 "
            f"(horizon={forecast_horizon}, stride={stride}, retrain={retrain}, device={self._device})"
        )

        # For foundation models, retrain=False uses zero-shot capability
        return self._model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=verbose,
            **kwargs,
        )

    def __repr__(self) -> str:
        model_hub = self.params.get("model_hub_name", "unknown")
        return (
            f"Chronos2(model={model_hub}, device={self._device}, "
            f"input_chunk={self.params.get('input_chunk_length')}, "
            f"output_chunk={self.params.get('output_chunk_length')})"
        )


__all__ = ["Chronos2", "_get_device", "_get_accelerator"]
