# This is the api server implementation using FASTAPI

# last updated: 28-March-2026
# updated by: Biswaprakash Nayak
# changes made: made this code and added all the neccesary functions.

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

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi import Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# internal imports
from model_inference import DEFAULT_WEIGHTS_PATH, predict_star_transit

# logging setup
LOG = logging.getLogger("cosmikai.main_code.server")
# basic logging config can be improved as needed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# constants used
# DB_PATH stores inference outputs
# the REQUEST_TIMEOUT_SECONDS is the max timeout time
# the DEFAULT_THRESHOLD is the score threshold for transit detection
# the DEFAULT_K_CANDIDATES is the default number of top BLS candidates to consider
DB_PATH = Path(__file__).resolve().parent / "stars_cache.db"
REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_K_CANDIDATES = 15

# internal state tracking
class _State:
    # server uptime tracking 
    start_time: float = 0.0

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
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(target_name, mission, author)
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

# schema for a stored prediction in history
class HistoryItem(BaseModel):
    id: int
    star_name: str
    mission: str
    score: float
    percentage: float
    period_days: float | None
    verdict: str
    num_datapoints: int | None
    timestamp: str

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

# gets cached prediction for the same target/mission/author combination, returns None if not found
# input:
# target_name: the name of the target star
# mission: the name of the mission
# author: the name of the data author
# output:
# a dictionary with the cached prediction data if found, or None if not found
async def _get_cached_prediction(target_name: str, mission: str, author: str) -> dict | None:
    # queries the database for a record matching the target_name, mission, and author
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM star_predictions
            WHERE target_name = ? AND mission = ? AND author = ?
            """,
            (target_name, mission, author),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    # converts the entire thing to a python dict from the json
    data = dict(row)
    data["best_candidate"] = json.loads(data["best_candidate"])
    data["all_scores"] = json.loads(data["all_scores"])
    # returns the data
    return data

# creates a new prediction record in cache/ storage
# input:
# payload: a dictionary containing the prediction data to be stored
async def _upsert_prediction(payload: dict) -> None:
    # this gets the datetime in ISO format
    now_iso = datetime.now(timezone.utc).isoformat()
    # inserts the data into the database
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO star_predictions (
                target_name, mission, author, threshold, k_candidates,
                best_score, verdict, best_candidate, num_candidates, device,
                all_scores, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_name, mission, author)
            DO UPDATE SET
                threshold = excluded.threshold,
                k_candidates = excluded.k_candidates,
                best_score = excluded.best_score,
                verdict = excluded.verdict,
                best_candidate = excluded.best_candidate,
                num_candidates = excluded.num_candidates,
                device = excluded.device,
                all_scores = excluded.all_scores,
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
                now_iso,
                now_iso,
            ),
        )
        await db.commit()

# calls on the inference pipeline 
# input:
# body: a PredictRequest object containing the inference parameters
# output:
# a dictionary containing the inference results
def _run_pipeline_sync(body: PredictRequest) -> dict:
    # this is the main function that runs the entire inference pipeline
    target_name = (body.target_name or body.star_name or "").strip()
    return predict_star_transit(
        target_name=target_name,
        mission=body.mission,
        author=body.author,
        threshold=body.threshold,
        k_candidates=body.k_candidates,
        model_weights_path=DEFAULT_WEIGHTS_PATH,
        device=None,
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
    LOG.info("Server ready. SQLite cache at %s", DB_PATH)
    yield
    # shutdown: 
    LOG.info("Server shutdown.")

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
    # checks the cache for existing predictions
    if not body.force_rerun:
        cached = await _get_cached_prediction(target_name, mission, author)
        if cached:
            period_days = float(cached["best_candidate"].get("period")) if cached.get("best_candidate") else None
            duration_estimate = float(cached["best_candidate"].get("duration")) if cached.get("best_candidate") else None
            depth = float(cached["best_candidate"].get("depth", 0.0)) if cached.get("best_candidate") else 0.0
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
        force_rerun=body.force_rerun,
    )
    # runs the inference pipeline in a worker thread
    # can increase the timeout if needed
    try:
        # runs the synchronous inference function in a separate thread and waits for the result
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_pipeline_sync, request),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    # handles the timeout error if the inference takes too long and returns a 408 HTTP error
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Inference timed out after {int(REQUEST_TIMEOUT_SECONDS)} seconds.",
        )
    # these catch other errors
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}")
    # define the results 
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
    }
    # adds to the database
    await _upsert_prediction(stored)
    # extracts the period, duration, and depth from the best candidate for the response
    period_days = float(result["best_candidate"].get("period")) if result.get("best_candidate") else None
    duration_estimate = float(result["best_candidate"].get("duration")) if result.get("best_candidate") else None
    depth = float(result["best_candidate"].get("depth", 0.0)) if result.get("best_candidate") else 0.0
    # returns the final response 
    return PredictionResponse(
        star_name=result["target_name"],
        mission=result["mission"],
        score=float(result["best_score"]),
        percentage=round(float(result["best_score"]) * 100.0, 2),
        period_days=period_days,
        verdict=result["verdict"],
        transit_depth_estimate=abs(depth),
        duration_estimate=duration_estimate,
        num_datapoints=None,
        cached=False,
        processing_time_seconds=round(_time.monotonic() - t0, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
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
        items.append(
            HistoryItem(
                id=int(row["id"]),
                star_name=row["target_name"],
                mission=row["mission"],
                score=float(row["best_score"]),
                percentage=round(float(row["best_score"]) * 100.0, 2),
                period_days=period_days,
                verdict=row["verdict"],
                num_datapoints=None,
                timestamp=row["updated_at"],
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
