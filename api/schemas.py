"""Pydantic models for the Body Measurement API."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    models_loaded: bool


class MeasureRequest(BaseModel):
    height_cm: float = Field(..., ge=100, le=250, description="Person's height in cm")
    gender: str = Field("neutral", pattern="^(neutral|male|female)$")


class MeasurementValues(BaseModel):
    value_cm: float
    label: str


class MeasureResponse(BaseModel):
    status: str
    num_persons: int
    measurements: Dict[str, MeasurementValues]
    processing_time_seconds: float
    errors: Optional[str] = None


class AsyncJobResponse(BaseModel):
    status: str
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    measurements: Optional[Dict[str, MeasurementValues]] = None
    errors: Optional[str] = None


# Standard label mapping for the 8 requested measurements
STANDARD_LABELS = {
    "arm_length": "A",
    "waist_girth": "B",
    "bust_girth": "C",
    "hip_girth": "D",
    "inside_leg_height": "E",
    "shoulder_to_crotch": "F",
    "shoulder_width": "G",
    "thigh_girth": "H",
}

# The 8 measurements we expose
EXPOSED_MEASUREMENTS = list(STANDARD_LABELS.keys())
