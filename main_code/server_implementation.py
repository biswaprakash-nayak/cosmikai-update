# This is the api server implementation using FASTAPI

# last updated: 5-April-2026
# updated by: Biswaprakash Nayak
# changes made: added logging

# imports
# asyncio for running sync inference in a worker thread with timeout handling
# json for storing nested prediction fields (candidate details, scores) in SQLite
# logging for server logs
# time for uptime and timing inference requests
# contextlib for async lifespan management of the FastAPI server
# datetime for timestamping DB records
# pathlib for handling file paths
# aiosqlite for async SQLite access
# fastapi for the web server and API handling
# pydantic for request/response data validation and modeling

import asyncio
import json
import logging
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import aiosqlite
import torch
from fastapi import FastAPI, HTTPException
from fastapi import Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# internal imports
from model_inference import DEFAULT_WEIGHTS_PATH, predict_star_transit, resolve_torch_device
from star_details_service import (
    StarDetailsFetchError,
    StarDetailsResponse,
    fetch_star_details_from_mast,
)

# logging setup
LOG = logging.getLogger("cosmikai.main_code.server")
# basic logging config can be improved as needed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# constants used
# DB_PATH stores inference outputs
# the REQUEST_TIMEOUT_SECONDS is the max timeout time (None = no limit)
# the DEFAULT_THRESHOLD is the score threshold for transit detection
# the DEFAULT_K_CANDIDATES is the default number of top BLS candidates to consider
DB_PATH = Path(__file__).resolve().parent / "stars_cache.db"
REQUEST_TIMEOUT_SECONDS = None
DEFAULT_THRESHOLD = 0.5
DEFAULT_K_CANDIDATES = 15
INFERENCE_HEARTBEAT_SECONDS = 10.0

# internal state tracking
class _State:
    # server uptime tracking 
    start_time: float = 0.0
    progress_lock: Lock = Lock()
    progress: dict[str, dict] = {}

_state = _State()

# creates the SQLite table for storage
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS star_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_name     TEXT NOT NULL,
    mission         TEXT NOT NULL,
    author          TEXT NOT NULL,
    threshold       REAL NOT NULL,
    k_candidates    INTEGER NOT NULL,
    best_score      REAL NOT NULL,
    verdict         TEXT NOT NULL,
    best_candidate  TEXT NOT NULL,
    num_candidates  INTEGER NOT NULL,
    device          TEXT NOT NULL,
    all_scores      TEXT NOT NULL,
    folded_lightcurve TEXT,
    candidate_rank  INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(target_name, mission, author, candidate_rank)
);
"""

# input schema for the /predict API endpoint
class PredictRequest(BaseModel):
    star_name: str | None = Field(default=None)
    target_name: str | None = Field(default=None)
    mission: str = Field(..., min_length=1)
    author: str = Field(default="None")
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    k_candidates: int = Field(default=DEFAULT_K_CANDIDATES, ge=1, le=100)
    cutoff_mode: str = Field(default="relative")
    confidence_drop_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    elbow_plus_extra: int = Field(default=1, ge=0, le=10)
    force_rerun: bool = Field(default=False)

# output schema for the /predict API endpoint
class PredictionResponse(BaseModel):
    star_name: str
    mission: str
    score: float
    percentage: float
    period_days: float | None
    verdict: str
    transit_depth_estimate: float | None
    duration_estimate: float | None
    num_datapoints: int | None
    cached: bool
    processing_time_seconds: float
    timestamp: str
    folded_lightcurve: list[float] | None = None

# schema for a stored prediction in history
class HistoryItem(BaseModel):
    id: int
    star_name: str
    mission: str
    score: float
    percentage: float
    period_days: float | None
    transit_depth_estimate: float | None = None
    verdict: str
    num_datapoints: int | None
    timestamp: str
    folded_lightcurve: list[float] | None = None

# response schema for the /history API endpoint
class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int

# response schema for the /stats API endpoint
class StatsResponse(BaseModel):
    total_analyzed: int
    average_score: float
    detection_rate: float
    missions: dict[str, int]
    above_threshold: int
    below_threshold: int

# response schema for the /health API endpoint
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_auprc: float
    total_predictions: int
    uptime_seconds: float

# initializes the SQLite database and creates the necessary table if it doesn't exist
async def _init_db() -> None:
    # ensures if the file exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # creates the table if it doesn't exist
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()

# gets cached predictions for the same target/mission/author combination
# input:
# target_name: the name of the target star
# mission: the name of the mission
# author: the name of the data author
# output:
# a list of dictionaries with cached prediction data, or empty list if not found
async def _get_cached_predictions(target_name: str, mission: str, author: str) -> list[dict]:
    # queries the database for all records matching the target_name, mission, and author, ordered by candidate_rank
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM star_predictions
            WHERE target_name = ? AND mission = ? AND author = ?
            ORDER BY candidate_rank ASC
            """,
            (target_name, mission, author),
        ) as cur:
            rows = await cur.fetchall()
    if not rows:
        return []
    # converts the rows to python dicts from json
    results = []
    for row in rows:
        data = dict(row)
        data["best_candidate"] = json.loads(data["best_candidate"])
        data["all_scores"] = json.loads(data["all_scores"])
        results.append(data)
    return results

# creates new prediction records in cache/storage (one per detected planet)
# input:
# payloads: a list of dictionaries, each containing a detected candidate with its data
async def _upsert_predictions_batch(payloads: list[dict]) -> None:
    if not payloads:
        return
    # this gets the datetime in ISO format
    now_iso = datetime.now(timezone.utc).isoformat()
    # inserts the data into the database
    async with aiosqlite.connect(DB_PATH) as db:
        first = payloads[0]
        await db.execute(
            """
            DELETE FROM star_predictions
            WHERE target_name = ? AND mission = ? AND author = ?
            """,
            (first["target_name"], first["mission"], first["author"]),
        )
        for rank, payload in enumerate(payloads):
            await db.execute(
                """
                INSERT INTO star_predictions (
                    target_name, mission, author, threshold, k_candidates,
                    best_score, verdict, best_candidate, num_candidates, device,
                    all_scores, folded_lightcurve, candidate_rank, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_name, mission, author, candidate_rank)
                DO UPDATE SET
                    threshold = excluded.threshold,
                    k_candidates = excluded.k_candidates,
                    best_score = excluded.best_score,
                    verdict = excluded.verdict,
                    best_candidate = excluded.best_candidate,
                    num_candidates = excluded.num_candidates,
                    device = excluded.device,
                    all_scores = excluded.all_scores,
                    folded_lightcurve = excluded.folded_lightcurve,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["target_name"],
                    payload["mission"],
                    payload["author"],
                    payload["threshold"],
                    payload["k_candidates"],
                    payload["best_score"],
                    payload["verdict"],
                    json.dumps(payload["best_candidate"]),
                    payload["num_candidates"],
                    payload["device"],
                    json.dumps(payload["all_scores"]),
                    json.dumps(payload.get("folded_lightcurve", [])),
                    rank,
                    now_iso,
                    now_iso,
                ),
            )
        await db.commit()

# calls on the inference pipeline 
# input:
# body: a PredictRequest object containing the inference parameters
# output:
# a list of dictionaries containing the inference results (multiple candidates)
def _run_pipeline_sync(body: PredictRequest, progress_callback=None) -> list[dict]:
    # this is the main function that runs the entire inference pipeline
    target_name = (body.target_name or body.star_name or "").strip()
    return predict_star_transit(
        target_name=target_name,
        mission=body.mission,
        author=body.author,
        threshold=body.threshold,
        k_candidates=body.k_candidates,
        cutoff_mode=body.cutoff_mode,
        confidence_drop_fraction=body.confidence_drop_fraction,
        elbow_plus_extra=body.elbow_plus_extra,
        model_weights_path=DEFAULT_WEIGHTS_PATH,
        device=None,
        progress_callback=progress_callback,
    )

# defines the FastAPI app, handling the startup and shutdown
# on startup, it initializes the database and logs that the server is ready
# on shutdown, it logs that the server is shutting down and closes it
# calls on the app to create the API endpoints and handle requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 
    _state.start_time = _time.monotonic()
    await _init_db()
    
    # Check and log CUDA availability
    device = resolve_torch_device()
    LOG.info("Server starting - Using device: %s", device)
    LOG.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        LOG.info("GPU: %s", torch.cuda.get_device_name(0))
        LOG.info("GPU Memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    
    LOG.info("Server ready. SQLite cache at %s", DB_PATH)
    try:
        yield
    except asyncio.CancelledError:
        LOG.info("Server received shutdown signal.")
    finally:
        # shutdown: 
        LOG.info("Server shutting down.")

# initializes the FastAPI app
app = FastAPI(
    # name of the app
    title="CosmiKAI API",
    # umm version number i guess, not sure if this is really needed
    version="1.0.0",
    # the lifespan function (startup and shutdown handlers)
    lifespan=lifespan,
)

# allow frontend to call this API from different origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# defines the /health endpoint to check server status and uptime
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM star_predictions") as cur:
            row = await cur.fetchone()
            total_predictions = int(row[0]) if row and row[0] is not None else 0
    # returns the health response with model status, uptime, and total predictions made
    return HealthResponse(
        status="ok",
        model_loaded=Path(DEFAULT_WEIGHTS_PATH).exists(),
        model_version="main_code_v1",
        model_auprc=0.0,
        total_predictions=total_predictions,
        uptime_seconds=round(_time.monotonic() - _state.start_time, 2),
    )


@app.get("/api/star-details", response_model=StarDetailsResponse)
async def star_details(star_name: str = QueryParam(..., min_length=1)) -> StarDetailsResponse:
    clean_name = star_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="star_name is required.")

    try:
        return await asyncio.to_thread(fetch_star_details_from_mast, clean_name)
    except StarDetailsFetchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

# defines the /predict endpoint to run inference on the target
# input:
# body: a PredictRequest object containing the inference parameters
# output:
# a PredictionResponse object containing the inference results and metadata
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(body: PredictRequest) -> PredictionResponse:
    target_name = (body.target_name or body.star_name or "").strip()
    # cleans the input parameters (removes leading/trailing whitespace)
    mission = body.mission.strip()
    # author is optional, set to "None" if none is given
    author = body.author.strip() if body.author else "None"
    # basic request validation checks if the required parameters are present, if not, raises a 400 error
    if not target_name or not mission:
        raise HTTPException(status_code=400, detail="star_name and mission are required.")
    LOG.info("PREDICT START star=%s mission=%s force_rerun=%s", target_name, mission, body.force_rerun)
    progress_key = f"{target_name}:{mission}:{_time.monotonic():.6f}"
    last_stage: str | None = None
    last_pct_bucket: int = -1

    def _progress_callback(stage: str, percent: int, message: str) -> None:
        nonlocal last_stage, last_pct_bucket
        pct = int(max(0, min(100, percent)))
        pct_bucket = pct // 10
        with _state.progress_lock:
            _state.progress[progress_key] = {
                "stage": stage,
                "percent": pct,
                "message": message,
                "updated": _time.monotonic(),
            }
        if stage != last_stage or pct_bucket > last_pct_bucket:
            LOG.info(
                "Progress update star=%s mission=%s stage=%s %s%% (%s)",
                target_name,
                mission,
                stage,
                pct,
                message,
            )
            last_stage = stage
            last_pct_bucket = pct_bucket
    # checks the cache for existing predictions (returns best one first)
    if not body.force_rerun:
        cached_list = await _get_cached_predictions(target_name, mission, author)
        if cached_list:
            cached = cached_list[0]  # return the best one
            period_days = float(cached["best_candidate"].get("period")) if cached.get("best_candidate") else None
            duration_estimate = float(cached["best_candidate"].get("duration")) if cached.get("best_candidate") else None
            depth = float(cached["best_candidate"].get("depth", 0.0)) if cached.get("best_candidate") else 0.0
            folded = None
            if cached.get("folded_lightcurve"):
                folded = json.loads(cached["folded_lightcurve"]) if isinstance(cached["folded_lightcurve"], str) else cached["folded_lightcurve"]
            return PredictionResponse(
                star_name=cached["target_name"],
                mission=cached["mission"],
                score=float(cached["best_score"]),
                percentage=round(float(cached["best_score"]) * 100.0, 2),
                period_days=period_days,
                verdict="TRANSIT_DETECTED" if cached["best_score"] >= body.threshold else "NO_TRANSIT",
                transit_depth_estimate=abs(depth),
                duration_estimate=duration_estimate,
                num_datapoints=None,
                cached=True,
                processing_time_seconds=0.0,
                timestamp=cached["updated_at"],
                folded_lightcurve=folded,
            )
    # running inference (incase not cached or force_rerun is True)
    # defines the request and start time
    t0 = _time.monotonic()
    request = PredictRequest(
        star_name=target_name,
        mission=mission,
        author=author,
        threshold=body.threshold,
        k_candidates=body.k_candidates,
        cutoff_mode=body.cutoff_mode,
        confidence_drop_fraction=body.confidence_drop_fraction,
        elbow_plus_extra=body.elbow_plus_extra,
        force_rerun=body.force_rerun,
    )
    # runs the inference pipeline in a worker thread
    # logs heartbeat so long requests don't look frozen
    try:
        task = asyncio.create_task(asyncio.to_thread(_run_pipeline_sync, request, _progress_callback))
        while True:
            elapsed = _time.monotonic() - t0
            if REQUEST_TIMEOUT_SECONDS is not None:
                remaining = REQUEST_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    task.cancel()
                    raise asyncio.TimeoutError
                wait_window = min(INFERENCE_HEARTBEAT_SECONDS, remaining)
            else:
                wait_window = INFERENCE_HEARTBEAT_SECONDS

            try:
                results = await asyncio.wait_for(asyncio.shield(task), timeout=wait_window)
                break
            except asyncio.TimeoutError:
                with _state.progress_lock:
                    progress = _state.progress.get(progress_key)
                if progress:
                    LOG.info(
                        "Inference still running star=%s mission=%s elapsed=%.1fs stage=%s %s%% (%s)",
                        target_name,
                        mission,
                        elapsed,
                        progress.get("stage", "unknown"),
                        progress.get("percent", 0),
                        progress.get("message", ""),
                    )
                else:
                    LOG.info(
                        "Inference still running for star=%s mission=%s (elapsed=%.1fs)",
                        target_name,
                        mission,
                        elapsed,
                    )
                continue
    # handles the timeout error if the inference takes too long and returns a 408 HTTP error
    except asyncio.TimeoutError:
        timeout_detail = (
            f"{int(REQUEST_TIMEOUT_SECONDS)} seconds"
            if REQUEST_TIMEOUT_SECONDS is not None
            else "the configured limit"
        )
        with _state.progress_lock:
            _state.progress.pop(progress_key, None)
        raise HTTPException(
            status_code=408,
            detail=f"Inference timed out after {timeout_detail}.",
        )
    # handles user interruption (Ctrl+C) gracefully
    except asyncio.CancelledError:
        LOG.warning("Inference cancelled by user")
        with _state.progress_lock:
            _state.progress.pop(progress_key, None)
        raise HTTPException(status_code=499, detail="Request cancelled by user")
    # these catch other errors
    except ValueError as exc:
        with _state.progress_lock:
            _state.progress.pop(progress_key, None)
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        with _state.progress_lock:
            _state.progress.pop(progress_key, None)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        with _state.progress_lock:
            _state.progress.pop(progress_key, None)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}")
    
    # results is now a list of detected candidates
    # prepare for storage and return the best one
    stored_batch = []
    for result in results:
        stored = {
            "target_name": result["target_name"],
            "mission": result["mission"],
            "author": author,
            "threshold": float(body.threshold),
            "k_candidates": int(body.k_candidates),
            "best_score": float(result["best_score"]),
            "verdict": result["verdict"],
            "best_candidate": result["best_candidate"],
            "num_candidates": int(result["num_candidates"]),
            "device": result["device"],
            "all_scores": [float(s) for s in result["all_scores"]],
            "folded_lightcurve": result.get("folded_lightcurve", []),
        }
        stored_batch.append(stored)
    
    # store all detections to the database
    await _upsert_predictions_batch(stored_batch)
    
    # return the best detection (first in list)
    best_result = results[0]
    period_days = float(best_result["best_candidate"].get("period")) if best_result.get("best_candidate") else None
    duration_estimate = float(best_result["best_candidate"].get("duration")) if best_result.get("best_candidate") else None
    depth = float(best_result["best_candidate"].get("depth", 0.0)) if best_result.get("best_candidate") else 0.0
    # returns the final response 
    LOG.info("PREDICT END star=%s mission=%s elapsed=%.2fs", target_name, mission, _time.monotonic() - t0)
    with _state.progress_lock:
        _state.progress.pop(progress_key, None)
    return PredictionResponse(
        star_name=best_result["target_name"],
        mission=best_result["mission"],
        score=float(best_result["best_score"]),
        percentage=round(float(best_result["best_score"]) * 100.0, 2),
        period_days=period_days,
        verdict=best_result["verdict"],
        transit_depth_estimate=abs(depth),
        duration_estimate=duration_estimate,
        num_datapoints=None,
        cached=False,
        processing_time_seconds=round(_time.monotonic() - t0, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
        folded_lightcurve=best_result.get("folded_lightcurve"),
    )

# defines the /history endpoint to list past predictions with pagination
# input:
# limit: the number of records to return (default 50, max 200)
# offset: the number of records to skip for pagination (default 0)
# output:
# a HistoryResponse object containing the list of past predictions and pagination metadata
@app.get("/api/history", response_model=HistoryResponse)
async def history(
    limit: int = QueryParam(default=50, ge=1, le=200),
    offset: int = QueryParam(default=0, ge=0),
    sort: str = QueryParam(default="timestamp"),
    order: str = QueryParam(default="desc"),
) -> HistoryResponse:
    # defines allowed sorting options 
    allowed_sort = {
        "timestamp": "updated_at",
        "score": "best_score",
        "star_name": "target_name",
        "mission": "mission",
        "period_days": "updated_at",
    }
    # validates the sort and order parameters
    sort_col = allowed_sort.get(sort, "updated_at")
    direction = "DESC" if order.lower() == "desc" else "ASC"
    # queries the database 
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT COUNT(*) AS n FROM star_predictions") as cur:
            row = await cur.fetchone()
            total = int(row["n"]) if row and row["n"] is not None else 0
        query = f"SELECT * FROM star_predictions ORDER BY {sort_col} {direction} LIMIT ? OFFSET ?"
        async with db.execute(query, (limit, offset)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    # converts the json output into the HistoryItem format for the response
    items: list[HistoryItem] = []
    for row in rows:
        best_candidate = json.loads(row["best_candidate"])
        period_days = float(best_candidate.get("period")) if best_candidate else None
        depth = float(best_candidate.get("depth", 0.0)) if best_candidate else None
        folded = None
        if row.get("folded_lightcurve"):
            folded = json.loads(row["folded_lightcurve"]) if isinstance(row["folded_lightcurve"], str) else row["folded_lightcurve"]
        items.append(
            HistoryItem(
                id=int(row["id"]),
                star_name=row["target_name"],
                mission=row["mission"],
                score=float(row["best_score"]),
                percentage=round(float(row["best_score"]) * 100.0, 2),
                period_days=period_days,
                transit_depth_estimate=abs(depth) if depth is not None else None,
                verdict=row["verdict"],
                num_datapoints=None,
                timestamp=row["updated_at"],
                folded_lightcurve=folded,
            )
        )
    # returns the history response
    return HistoryResponse(items=items, total=total)

# defines the /stats endpoint for dashboard telemetry
@app.get("/api/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    # queries the database for the stats
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT COUNT(*) AS n FROM star_predictions") as cur:
            row = await cur.fetchone()
            total = int(row["n"]) if row and row["n"] is not None else 0
        if total == 0:
            return StatsResponse(
                total_analyzed=0,
                average_score=0.0,
                detection_rate=0.0,
                missions={},
                above_threshold=0,
                below_threshold=0,
            )
        async with db.execute("SELECT AVG(best_score) AS avg_score FROM star_predictions") as cur:
            row = await cur.fetchone()
            avg_score = float(row["avg_score"]) if row and row["avg_score"] is not None else 0.0
        async with db.execute(
            "SELECT COUNT(*) AS n FROM star_predictions WHERE best_score >= ?",
            (DEFAULT_THRESHOLD,),
        ) as cur:
            row = await cur.fetchone()
            above = int(row["n"]) if row and row["n"] is not None else 0
        async with db.execute(
            "SELECT mission, COUNT(*) AS n FROM star_predictions GROUP BY mission"
        ) as cur:
            missions = {row["mission"]: int(row["n"]) for row in await cur.fetchall()}
    # returns the stats response
    return StatsResponse(
        total_analyzed=total,
        average_score=round(avg_score, 4),
        detection_rate=round(above / total, 4),
        missions=missions,
        above_threshold=above,
        below_threshold=total - above,
    )
