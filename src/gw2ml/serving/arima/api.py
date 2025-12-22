from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from gw2ml.modeling import load_model
from gw2ml.pipeline.data_preparation import build_pipeline_context, load_latest_validation
from gw2ml.pipeline.feature_hooks import apply_augmentors

router = APIRouter(prefix="/models/arima", tags=["arima"])


class PredictionRow(BaseModel):
    item_id: int = Field(..., ge=0)
    fetched_at: datetime
    buy_unit_price: float | None = Field(None, ge=0)
    sell_unit_price: float | None = Field(None, ge=0)
    buy_quantity: int | None = Field(None, ge=0)
    sell_quantity: int | None = Field(None, ge=0)


class PredictionRequest(BaseModel):
    model_key: str = "arima"
    rows: list[PredictionRow] = Field(default_factory=list)
    fetch_latest: bool = False
    lookback_days: int | None = None
    rows_per_item: int = Field(200, ge=1)
    item_ids: list[int] | None = None


class PredictionResponse(BaseModel):
    model_key: str
    generated_at: datetime
    predictions: list[dict[str, Any]]


def _rows_to_frame(rows: list[PredictionRow]) -> pd.DataFrame:
    frame = pd.DataFrame([row.model_dump() for row in rows])
    if frame.empty:
        return frame
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["fetched_at"]).reset_index(drop=True)
    return frame


@router.post("/predict", response_model=PredictionResponse)
async def handle_predict(payload: PredictionRequest) -> PredictionResponse:
    context = build_pipeline_context(
        payload.model_key,
        run_date=datetime.now(UTC),
        lookback_days=payload.lookback_days or 30,
    )

    if payload.fetch_latest:
        inference_df, _ = load_latest_validation(
            context,
            item_ids=payload.item_ids,
            lookback_days=payload.lookback_days,
            rows_per_item=payload.rows_per_item,
        )
    else:
        inference_df = _rows_to_frame(payload.rows)
        if inference_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one row or set fetch_latest=true.",
            )

    inference_df = apply_augmentors(inference_df, context, "inference")

    try:
        model = load_model(payload.model_key)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No artifact available for model_key={payload.model_key}.",
        ) from exc

    prediction_values = model.predict(inference_df).reset_index(drop=True)
    records = [
        {
            "item_id": int(row["item_id"]),
            "prediction": float(prediction_values.iloc[idx]),
            "fetched_at": row.get("fetched_at"),
        }
        for idx, row in inference_df.reset_index(drop=True).iterrows()
    ]

    return PredictionResponse(
        model_key=payload.model_key,
        generated_at=datetime.now(UTC),
        predictions=records,
    )

