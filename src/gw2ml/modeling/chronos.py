"""Chronos Foundation Model for Time Series Forecasting."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from darts import TimeSeries
from darts.models import Chronos2Model
from .base import BaseModel
from gw2ml.utils import get_logger

logger = get_logger("modeling.chronos")


class Chronos2(BaseModel):
    """
    Chronos Foundation Model by AutoGluon/Amazon
    
    Args:
       input_chunk_length: Number of past values to use
       output_chunk_length: Number of future values to predict
       model_hub_name: HuggingFace model name
    """

    @property
    def name(self) -> str:
        return "Chronos2"

    @property
    def default_params(self) -> Dict[str, Any]:
        # smaller: autogluon/chronos-2-small -> 28m, synt is 120m
        return {"model_hub_name": "autogluon/chronos-2-synth"}

    def __init__(
            self,
            input_chunk_length: int = 168,
            output_chunk_length: int = 168,
            **kwargs: Any,
    ) -> None:
        # Merge default params with provided kwargs
        self.params: Dict[str, Any] = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            **self.default_params,
            **kwargs,
        }
        self._model: Optional[Chronos2Model] = None

    def build_model(self, **kwargs: Any) -> Chronos2Model:
        build_params = {**self.params, **kwargs}

        input_chunk_length = build_params.pop("input_chunk_length")
        output_chunk_length = build_params.pop("output_chunk_length")
        # Remove parameters not accepted by Chronos2Model
        build_params.pop("model", None)
        build_params.pop("epochs", None)
        model_hub_name = build_params.pop("model_hub_name", "autogluon/chronos-2-synth")

        logger.info(f"Building Chronos2Model with hub name: {model_hub_name}")
        return Chronos2Model(
            model_name=model_hub_name,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            **build_params
        )

    def fit(self, series: TimeSeries, **kwargs: Any) -> Chronos2:
        # Rebuild model to ensure a fresh start or consistent state,
        # matching the behavior of other models in the pipeline.
        
        # MPS doesn't support float64. Ensure series is float32 if on MPS or in general for Chronos.
        if series.dtype == np.float64:
            logger.info("Casting series to float32 for Chronos2 (MPS compatibility)")
            series = series.astype(np.float32)

        self._model = self.build_model()
        logger.info(f"Fitting Chronos2 (n={series.shape[0]}) with kwargs {kwargs}")

        # Chronos is a foundation model. 'fit' can be used for fine-tuning.
        # Darts requires fit() to be called to initialize internal state, 
        # but we can use epochs=0 for zero-shot.
        epochs = kwargs.pop("epochs", self.params.get("epochs", 0))
        logger.info(f"Fitting Chronos2 model (epochs={epochs})")
        self._model.fit(series, epochs=epochs,verbose=True, **kwargs)
        self._context_series = series
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        if self._model is None:
            self._model = self.build_model()
        
        series = kwargs.pop("series", getattr(self, "_context_series", None))
        if series is not None and series.dtype == np.float64:
            logger.info("Casting series to float32 for Chronos2 predict (MPS compatibility)")
            series = series.astype(np.float32)

        logger.info(f"Generating prediction with Chronos2 (n={n})")
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
        """Walk-forward backtesting."""
        if self._model is None:
            self._model = self.build_model()

        if series.dtype == np.float64:
            logger.info("Casting series to float32 for Chronos2 historical_forecasts (MPS compatibility)")
            series = series.astype(np.float32)

        logger.info(f"Computing historical forecasts with Chronos2 (horizon={forecast_horizon}, retrain={retrain})")
        # If retrain is False and we have a foundation model, we can use it zero-shot
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
        pass
        # TODO: implement


__all__ = ["Chronos2"]
