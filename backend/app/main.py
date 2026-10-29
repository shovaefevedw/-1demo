"""PhotoDepth 3D local server."""

from __future__ import annotations

import logging

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODEL_IDS
from app.routers.api import router
from app.services.device import preferred_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="PhotoDepth 3D", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Point-Count", "X-Elapsed-Ms"],
)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": preferred_device(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch": torch.__version__,
        "models": list(MODEL_IDS),
    }
