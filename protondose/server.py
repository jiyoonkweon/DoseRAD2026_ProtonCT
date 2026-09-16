"""Inference server for the DoseRAD2026 Grand Challenge `invoke` API.

Run with `python -m protondose.server`; the container entry point.

Only /invoke is timed, so weight loading, CUDA context creation, cuDNN
autotuning and the per-energy input channels all happen before /health returns.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Response, status
from uvicorn.config import LOGGING_CONFIG

from . import challenge_io, inference

MODEL_DIR = Path(os.environ.get("DOSERAD_MODEL_DIR", "/opt/app/model"))


def init_model():
    t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    prop = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  device: {prop.name}  "
          f"vram={prop.total_memory / 2 ** 30:.1f} GiB  sm={prop.major}.{prop.minor}",
          flush=True)

    model = inference.build_model(MODEL_DIR, device="cuda")
    challenge_io.selftest_parallel_deflate()
    inference.warmup(model)
    print(f"[init_model] ready in {time.time() - t0:.1f}s", flush=True)
    return model


MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODELS["dose"] = init_model()
    yield
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if "dose" in MODELS:
        return Response(status_code=status.HTTP_200_OK)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


@app.post("/invoke")
async def invoke():
    t0 = time.time()
    challenge_io.run(MODELS["dose"])
    print(f"[invoke] completed in {time.time() - t0:.2f}s", flush=True)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
