from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gw2ml.pipelines.config import DEFAULT_CONFIG, merge_config
from gw2ml.pipelines.forecast import forecast_item
from gw2ml.pipelines.train import train_items

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALLOWED_CONFIG_KEYS = {"data", "split", "forecast", "metric", "models"}


def _sanitize_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cfg:
        return {}
    return {key: value for key, value in cfg.items() if key in ALLOWED_CONFIG_KEYS}


class TrainRequest(BaseModel):
    item_ids: List[int] = Field(..., min_items=1, description="List of item IDs to train models for")
    override_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional config overrides (data/split/forecast/metric/models); paths/DB/S3 are ignored",
    )


class ForecastRequest(BaseModel):
    item_id: int = Field(..., description="Item ID to forecast")
    override_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional config overrides (data/forecast/metric/models); paths/DB/S3 are ignored",
    )


app = FastAPI(title="GW2ML API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(req: TrainRequest) -> Dict[str, Any]:
    logger.info(f"POST /train - item_ids={req.item_ids}")
    try:
        sanitized = _sanitize_config(req.override_config)
        results = train_items(req.item_ids, override_config=merge_config(DEFAULT_CONFIG, sanitized))
        logger.info(f"Training complete - processed {len(results)} item(s)")
        return {"items": results}
    except Exception as exc:  # pragma: no cover - surface errors to client
        logger.error(f"Training failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:
    logger.info(f"POST /forecast - item_id={req.item_id}")
    try:
        sanitized = _sanitize_config(req.override_config)
        result = forecast_item(req.item_id, override_config=merge_config(DEFAULT_CONFIG, sanitized))
        logger.info(f"Forecast complete for item {req.item_id}")
        return result
    except FileNotFoundError as exc:
        logger.warning(f"Forecast failed - artifacts not found: {exc}")
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.error(f"Forecast failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# Run with: uv run fastapi dev apps/api/main.py --host 0.0.0.0 --port 8000

