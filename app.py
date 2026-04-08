"""
FastAPI server — Micro-Grid Energy Arbitrator OpenEnv
Team RauResh — IIT Mandi

Endpoints:
  GET  /          → metadata + health
  GET  /health    → liveness probe
  POST /reset     → start episode, returns GridObservation
  POST /step      → advance one hour, returns EpisodeResult
  GET  /state     → full internal state
  GET  /tasks     → task descriptions
  POST /grade     → grade completed episode
"""

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import MicroGridEnv
from env.models import ActionType, GridAction, LoadTier
from graders.graders import GRADER_REGISTRY, get_grader

app = FastAPI(
    title="Micro-Grid Energy Arbitrator OpenEnv",
    description="Team RauResh — IIT Mandi | Solar + Battery micro-grid management benchmark",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, MicroGridEnv] = {}


class ResetRequest(BaseModel):
    task_id:   str          = "task_easy"
    seed:      Optional[int]= 42
    max_steps: Optional[int]= 24


class StepRequest(BaseModel):
    session_id:  str
    action_type: str          # buy_energy | sell_energy | store_energy | idle
    quantity_kw: float = 0.0
    shed_tier:   Optional[str] = None   # deferrable | essential | critical


class GradeRequest(BaseModel):
    session_id: str


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name":    "microgrid-energy-arbitrator-v1",
        "version": "1.0.0",
        "team":    "RauResh — IIT Mandi",
        "status":  "ok",
        "tasks":   list(GRADER_REGISTRY.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    if req.task_id not in GRADER_REGISTRY:
        raise HTTPException(400, f"Unknown task_id {req.task_id!r}. Valid: {list(GRADER_REGISTRY)}")
    session_id = str(uuid.uuid4())
    env = MicroGridEnv(task_id=req.task_id, max_steps=req.max_steps or 24, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session {req.session_id!r} not found. Call /reset first.")
    try:
        atype = ActionType(req.action_type.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid action_type {req.action_type!r}")

    shed = None
    if req.shed_tier:
        try:
            shed = LoadTier(req.shed_tier.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid shed_tier {req.shed_tier!r}")

    action = GridAction(action_type=atype, quantity_kw=req.quantity_kw, shed_tier=shed)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return result.model_dump()


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session {session_id!r} not found.")
    return env.state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {tid: g.describe() for tid, g in GRADER_REGISTRY.items()}


@app.post("/grade")
def grade(req: GradeRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session {req.session_id!r} not found.")
    s      = env.state()
    grader = get_grader(s.get("task_id", "task_easy"))
    result = grader.grade(s)
    return {
        "session_id":     req.session_id,
        "score":          result.score,
        "uptime_score":   result.uptime_score,
        "economic_score": result.economic_score,
        "reserve_score":  result.reserve_score,
        "blackout_score": result.blackout_score,
        "passed":         result.passed,
        "feedback":       result.feedback,
        "episode_state":  s,
    }
