from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AppConfig, load_config
from .generation import LyricGenerator
from .importer import SUPPORTED_SUFFIXES, prepare_dataset, write_uploaded_file


class PredictRequest(BaseModel):
    context: str = Field(min_length=1)
    continue_: bool = Field(default=True, alias="continue")
    mode: str | None = None
    strictness: str | None = None
    correction: bool = False


class PredictResponse(BaseModel):
    text: str
    accepted: bool
    confidence: float
    reason: str
    corrected_context: str | None = None


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path or os.environ.get("LYRICPREDICT_CONFIG", "configs/default.yaml"))
    app = FastAPI(title="LyricPredict", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    generators: dict[str, LyricGenerator] = {}

    def get_generator(mode: str | None = None) -> LyricGenerator:
        del mode
        key = "matching"
        if key not in generators:
            generators[key] = LyricGenerator(config, mode=key)
        return generators[key]

    @app.post("/api/import")
    async def import_lyrics(files: Annotated[list[UploadFile], File()]):
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        saved = []
        for uploaded in files:
            suffix = Path(uploaded.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {uploaded.filename}")
            data = await uploaded.read()
            saved.append(str(write_uploaded_file(config.paths.raw_dir, uploaded.filename or "lyrics.txt", data)))
        stats = prepare_dataset(config.paths.raw_dir, config.paths.processed_dir, config.training.validation_ratio)
        generators.clear()
        return {"saved": saved, "stats": asdict(stats)}

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        prediction = get_generator(request.mode).predict(
            request.context,
            strictness=request.strictness,
            correction=request.correction,
        )
        return PredictResponse(
            text=prediction.text,
            accepted=prediction.accepted,
            confidence=prediction.confidence,
            reason=prediction.reason,
            corrected_context=prediction.corrected_context,
        )

    web_dir = config.paths.web_dir
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def index():
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
