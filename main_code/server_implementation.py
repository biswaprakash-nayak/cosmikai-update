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

#from __future__ import annotations         # can help with older Python versions, uncomment if needed

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
    target_name: str = Field(..., min_length=1)
    mission: str = Field(..., min_length=1)
    author: str = Field(default="None")
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    k_candidates: int = Field(default=DEFAULT_K_CANDIDATES, ge=1, le=100)
    force_rerun: bool = Field(default=False)

# output schema for the /predict API endpoint
class PredictionResponse(BaseModel):
    target_name: str
    mission: str
    author: str
    threshold: float
    best_score: float
    verdict: str
    best_candidate: dict
    num_candidates: int
    device: str
    all_scores: list[float]
    cached: bool
    processing_time_seconds: float
    timestamp: str

# schema for a stored prediction 
class HistoryItem(BaseModel):
    id: int
    target_name: str
    mission: str
    author: str
    threshold: float
    k_candidates: int
    best_score: float
    verdict: str
    best_candidate: dict
    num_candidates: int
    device: str
    all_scores: list[float]
    created_at: str
    updated_at: str

# response schema for the /history API endpoint
class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int

# response schema for the /health API endpoint
class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    db_path: str
    weights_path: str

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
    # calls the predict_star_transit function from model_inference.py 
    return predict_star_transit(
        target_name=body.target_name,
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
    # just to check if the api is running
    return HealthResponse(
        status="ok",
        uptime_seconds=round(_time.monotonic() - _state.start_time, 2),
        db_path=str(DB_PATH),
        weights_path=str(DEFAULT_WEIGHTS_PATH),
    )

# defines the /predict endpoint to run inference on the target
# input:
# body: a PredictRequest object containing the inference parameters
# output:
# a PredictionResponse object containing the inference results and metadata
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(body: PredictRequest) -> PredictionResponse:
    # edit user inputs (remove whitespace)
    target_name = body.target_name.strip()
    mission = body.mission.strip()
    # author is optional, set to "None" if none is given
    author = body.author.strip() if body.author else "None"
    # basic request validation
    if not target_name or not mission:
        raise HTTPException(status_code=400, detail="target_name and mission are required.")
    # checks cache if it exists unless force_rerun is True
    if not body.force_rerun:
        cached = await _get_cached_prediction(target_name, mission, author)
        if cached:
            return PredictionResponse(
                target_name=cached["target_name"],
                mission=cached["mission"],
                author=cached["author"],
                threshold=body.threshold,
                best_score=cached["best_score"],
                verdict="TRANSIT_DETECTED" if cached["best_score"] >= body.threshold else "NO_TRANSIT",
                best_candidate=cached["best_candidate"],
                num_candidates=cached["num_candidates"],
                device=cached["device"],
                all_scores=cached["all_scores"],
                cached=True,
                processing_time_seconds=0.0,
                timestamp=cached["updated_at"],
            )
    # running inference (incase not cached or force_rerun is True)
    # defines the request and start time
    t0 = _time.monotonic()
    request = PredictRequest(
        target_name=target_name,
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
    # return API response for frontend
    return PredictionResponse(
        target_name=result["target_name"],
        mission=result["mission"],
        author=author,
        threshold=float(body.threshold),
        best_score=float(result["best_score"]),
        verdict=result["verdict"],
        best_candidate=result["best_candidate"],
        num_candidates=int(result["num_candidates"]),
        device=result["device"],
        all_scores=[float(s) for s in result["all_scores"]],
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
) -> HistoryResponse:
    # list stored predictions
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT COUNT(*) AS n FROM star_predictions") as cur:
            total = int((await cur.fetchone())["n"])
        async with db.execute(
            """
            SELECT * FROM star_predictions
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    # convert the json to dict and create HistoryItem objects for the response
    items = []
    for row in rows:
        row["best_candidate"] = json.loads(row["best_candidate"])
        row["all_scores"] = json.loads(row["all_scores"])
        items.append(HistoryItem(**row))
    # return the response with the list of history items 
    return HistoryResponse(items=items, total=total, limit=limit, offset=offset)
