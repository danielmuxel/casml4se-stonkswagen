"""Prophet Model for Time Series Forecasting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from darts import TimeSeries
from darts.models import ARIMA
from darts.models import Chronos2Model
from .base import BaseModel


class Chronos2(BaseModel):
    """
    Prophet by Meta
    
    Args:
       #TODO: doc.
    """

    @property
    def name(self) -> str:
        return "Chronos2"

    @property
    def default_params(self) -> Dict[str, Any]:
        # smaller: autogluon/chronos-2-small -> 28m, synt is 120m
        return {"model": "Chronos2", "model_hub_name": "autogluon/chronos-2-synth"}

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
        model_hub_name = build_params.pop("model_hub_name", "autogluon/chronos-2-synth")

        return Chronos2Model(
            model_name=model_hub_name,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            **build_params
        )

    def fit(self, series: TimeSeries, **kwargs: Any) -> Chronos2:
        self._model = self.build_model()
        self._model.fit(series)
        return self

    def predict(self, n: int, **kwargs: Any) -> TimeSeries:
        if self._model is None:
            raise ValueError("Model must be fitted first. Call fit().")
        return self._model.predict(n=n, **kwargs)

    def __repr__(self) -> str:
        pass
        # TODO: implement


__all__ = ["Chronos2"]
