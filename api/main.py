import logging
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from api.config import APP_VERSION, LOG_DIR, INFERENCE_LOG_PATH
from api.schemas import TransactionIn, PredictionOut
from api.service import load_artifacts, run_inference, is_ready
from api.utils import latency_ms


def setup_logging() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        # Avoid duplicate handlers in reload/tests
        return

    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        INFERENCE_LOG_PATH,
        maxBytes=2_000_000,
        backupCount=3,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


setup_logging()
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
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
