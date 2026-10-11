"""
api/main.py — FastAPI backend for TechSight.

Serves technology trends, forecasts, and analytics data
to the Streamlit dashboard via REST endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import API_HOST, API_PORT, CORS_ORIGINS

# Import mock data for demo mode
from mock.mock_data import (
    get_technologies, get_yearly_indicators,
    get_trend_scores, get_forecasts,
)

app = FastAPI(
    title="TechSight API",
    description="Cloud-Based Technology Trend Aggregation and Forecasting API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class TechnologyResponse(BaseModel):
    id: int
    name: str
    cluster_id: int
    top_terms: str
    document_count: int
    description: str


class IndicatorResponse(BaseModel):
    technology_id: int
    technology_name: str
    year: int
    patent_count: int
    paper_count: int
    total_count: int
    avg_citations: Optional[float]
    is_complete: bool


class TrendScoreResponse(BaseModel):
    technology_id: int
    technology_name: str
    score: float
    label: str
    growth_rate: float
    consistency: float
    acceleration: float
    activity_volume: float
    citation_signal: float


class ForecastResponse(BaseModel):
    technology_name: str
    year: int
    series_type: str
    predicted_value: float
    lower_bound: float
    upper_bound: float
    is_future: bool


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "project": "TechSight",
        "description": "Technology Trend Aggregation and Forecasting API",
    }


@app.get("/api/technologies", response_model=List[TechnologyResponse])
def list_technologies():
    """Get all discovered technology topics."""
    df = get_technologies()
    return df.to_dict("records")


@app.get("/api/indicators", response_model=List[IndicatorResponse])
def list_indicators(
    technology_name: Optional[str] = Query(None, description="Filter by technology"),
    year_from: Optional[int] = Query(None, description="Start year"),
    year_to: Optional[int] = Query(None, description="End year"),
):
    """Get yearly patent/paper indicators for technologies."""
    df = get_yearly_indicators()

    if technology_name:
        df = df[df["technology_name"] == technology_name]
    if year_from:
        df = df[df["year"] >= year_from]
    if year_to:
        df = df[df["year"] <= year_to]

    return df.to_dict("records")


@app.get("/api/trend-scores", response_model=List[TrendScoreResponse])
def list_trend_scores():
    """Get computed trend scores for all technologies."""
    df = get_trend_scores()
    return df.to_dict("records")


@app.get("/api/forecasts", response_model=List[ForecastResponse])
def list_forecasts(
    technology_name: Optional[str] = Query(None, description="Filter by technology"),
):
    """Get forecast predictions."""
    df = get_forecasts()

    if technology_name:
        df = df[df["technology_name"] == technology_name]

    return df.to_dict("records")


@app.get("/api/summary")
def get_summary():
    """Get high-level summary statistics."""
    techs = get_technologies()
    indicators = get_yearly_indicators()
    scores = get_trend_scores()

    return {
        "total_technologies": len(techs),
        "total_documents": int(techs["document_count"].sum()),
        "year_range": f"{int(indicators['year'].min())}–{int(indicators['year'].max())}",
        "trend_scores": scores[["technology_name", "score", "label"]].to_dict("records"),
        "data_sources": ["OpenAlex (Research Papers)", "USPTO (Patents)"],
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
