"""
Body Measurement API — FastAPI microservice.

Endpoints:
    GET  /health   - Health check (GPU, models)
    POST /measure  - Synchronous: upload image, get measurements back
    POST /run      - Async: upload image, get job_id, poll for results
    GET  /jobs/{job_id} - Poll async job status
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.pipeline import pipeline
from api.schemas import (
    EXPOSED_MEASUREMENTS,
    STANDARD_LABELS,
    AsyncJobResponse,
    HealthResponse,
    JobStatusResponse,
    MeasureResponse,
    MeasurementValues,
)

# ---------------------------------------------------------------------------
# In-memory job store (simple dict — replace with Redis/DB for production)
# ---------------------------------------------------------------------------
_jobs: Dict[str, dict] = {}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB


def _validate_image(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")


def _filter_measurements(raw: dict) -> Dict[str, MeasurementValues]:
    """Keep only the 8 exposed measurements, attach standard labels."""
    out = {}
    for key in EXPOSED_MEASUREMENTS:
        value = raw.get(key)
        if value is None:
            continue
        out[key] = MeasurementValues(
            value_cm=round(float(value), 2),
            label=STANDARD_LABELS[key],
        )
    return out


# ---------------------------------------------------------------------------
# App lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    lightweight = os.environ.get("LIGHTWEIGHT_MODE", "false").lower() == "true"
    pipeline.load_models(lightweight=lightweight)
    yield


app = FastAPI(
    title="Body Measurement Service",
    description="GPU-accelerated body measurement extraction from 2D photographs.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_mem_used = None
    gpu_mem_total = None
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_used = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1)
        gpu_mem_total = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1)
    return HealthResponse(
        status="healthy" if pipeline.is_loaded else "degraded",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        models_loaded=pipeline.is_loaded,
    )


# ---------------------------------------------------------------------------
# POST /measure — synchronous
# ---------------------------------------------------------------------------

@app.post("/measure", response_model=MeasureResponse)
async def measure(
    image: UploadFile = File(...),
    height_cm: float = Form(..., ge=100, le=250),
    gender: str = Form("neutral"),
):
    _validate_image(image)
    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")

    t0 = time.time()
    try:
        result = pipeline.process_image(image_bytes, height_cm)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    elapsed = round(time.time() - t0, 2)

    filtered = _filter_measurements(result["measurements"])

    return MeasureResponse(
        status="success",
        num_persons=result["num_persons"],
        measurements=filtered,
        processing_time_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# POST /run — async job submission
# ---------------------------------------------------------------------------

@app.post("/run", response_model=AsyncJobResponse)
async def run_async(
    image: UploadFile = File(...),
    height_cm: float = Form(..., ge=100, le=250),
    gender: str = Form("neutral"),
):
    _validate_image(image)
    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "queued", "measurements": None, "errors": None}

    import asyncio
    loop = asyncio.get_event_loop()

    async def _run():
        _jobs[job_id]["status"] = "processing"
        try:
            result = await loop.run_in_executor(
                None, pipeline.process_image, image_bytes, height_cm
            )
            _jobs[job_id]["measurements"] = _filter_measurements(result["measurements"])
            _jobs[job_id]["status"] = "completed"
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["errors"] = str(e)

    asyncio.ensure_future(_run())

    return AsyncJobResponse(status="accepted", job_id=job_id)


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — poll async job
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        measurements=job["measurements"],
        errors=job["errors"],
    )
