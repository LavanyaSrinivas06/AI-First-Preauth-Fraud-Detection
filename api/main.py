import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from api.schemas import TransactionIn, PredictionOut
from api.service import load_artifacts, run_inference, is_ready
from api.utils import latency_ms

APP_VERSION = "0.1.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
        logger.info("Artifacts loaded successfully")
    except Exception as e:
        logger.exception("Startup artifact loading failed: %s", e)
    yield


app = FastAPI(title="FPN Inference API", version=APP_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "preprocess_loaded": bool(is_ready()), "version": APP_VERSION}


@app.post("/predict", response_model=PredictionOut)
def predict(req: TransactionIn, response: Response):
    start = time.perf_counter()
    try:
        out = run_inference(req.model_dump())
        return JSONResponse(content=out, headers={"X-Latency-ms": f"{latency_ms(start):.2f}"})
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Prediction failed"},
            headers={"X-Latency-ms": f"{latency_ms(start):.2f}"},
        )
