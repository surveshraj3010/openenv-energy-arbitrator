"""
Task definitions and graders for the Micro-Grid Energy Arbitrator.

Team RauResh — IIT Mandi
-----------------------------------------------------------
Scoring formula (team's original design):

  Score = w_uptime   * uptime_score
        + w_economic * economic_score
        + w_reserve  * reserve_score
        + w_blackout * blackout_score    (hard penalty)

All weights sum to 1.0. Graders are fully deterministic:
  same episode state → same score, every time.

Difficulty progression:
  Easy   → generous budget, high initial SoC, sunny weather
  Medium → tighter budget, intermittent solar, must time purchases
  Hard   → storm scenario, critical low SoC, load-shed decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GradeResult:
    score:          float   # 0.0 – 1.0
    uptime_score:   float
    economic_score: float
    reserve_score:  float
    blackout_score: float
    passed:         bool
    feedback:       str


# ── Shared scoring helpers ────────────────────────────────────────────────────

def _score_uptime(blackout_count: int, max_hours: int) -> float:
    """
    Full marks for zero blackouts. Each blackout costs proportionally.
    One blackout on a 24-hour episode → 0.0 (binary for safety).
    """
    if blackout_count == 0:
        return 1.0
    return 0.0   # any blackout = failed uptime


def _score_economic(
    net_cost_usd: float,
    budget_usd:   float,
    revenue_usd:  float,
) -> float:
    """
    Measures how well the agent managed costs vs. a budget ceiling.
    Extra revenue earns bonus up to 1.0 cap.
    Formula: 1 - (net_cost / budget), clipped to [0, 1].
    """
    if net_cost_usd <= 0:
        # Agent made money (net revenue) — full marks + bonus
        return min(1.0, 1.0 + abs(net_cost_usd) / max(budget_usd, 0.01))
    ratio = net_cost_usd / max(budget_usd, 0.01)
    return max(0.0, round(1.0 - ratio, 4))


def _score_reserve(hours_above_reserve: int, total_hours: int) -> float:
    """Fraction of hours the battery stayed above the 20% SoC reserve floor."""
    if total_hours == 0:
        return 0.0
    return round(min(1.0, hours_above_reserve / total_hours), 4)


def _score_blackout(blackout_count: int) -> float:
    """Hard safety gate — zero tolerance."""
    return 1.0 if blackout_count == 0 else 0.0


def _weighted_score(
    uptime:   float,
    economic: float,
    reserve:  float,
    blackout: float,
    weights:  Dict[str, float],
) -> float:
    total = (
        uptime   * weights["uptime"]
        + economic * weights["economic"]
        + reserve  * weights["reserve"]
        + blackout * weights["blackout"]
    )
    return round(min(1.0, max(0.0, total)), 4)


def _build_feedback(
    uptime_s: float,
    economic_s: float,
    reserve_s: float,
    blackout_count: int,
    net_cost: float,
) -> str:
    parts = []
    if blackout_count > 0:
        parts.append(f"BLACKOUT occurred ({blackout_count}x) — grid failure")
    if economic_s < 0.5:
        parts.append(f"High net cost ${net_cost:.2f} — improve arbitrage timing")
    if reserve_s < 0.6:
        parts.append("Battery reserve low too often — buy energy earlier")
    return "; ".join(parts) if parts else "Good grid management"


def grade_episode(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> GradeResult:
    blackout_count      = state.get("blackout_count", 0)
    net_cost            = state.get("net_cost_usd", 0.0)
    revenue             = state.get("total_revenue_usd", 0.0)
    hours_above_reserve = state.get("hours_above_reserve", 0)
    total_hours         = state.get("hour", 24)

    u_score = _score_uptime(blackout_count, total_hours)
    e_score = _score_economic(net_cost, config["budget_usd"], revenue)
    r_score = _score_reserve(hours_above_reserve, total_hours)
    b_score = _score_blackout(blackout_count)

    final   = _weighted_score(u_score, e_score, r_score, b_score, config["weights"])
    feedback = _build_feedback(u_score, e_score, r_score, blackout_count, net_cost)

    return GradeResult(
        score=final,
        uptime_score=round(u_score, 4),
        economic_score=round(e_score, 4),
        reserve_score=round(r_score, 4),
        blackout_score=round(b_score, 4),
        passed=final >= config["pass_threshold"],
        feedback=feedback,
    )


# ── Task 1 — EASY ─────────────────────────────────────────────────────────────

EASY_CONFIG = {
    "task_id":        "task_easy",
    "description": (
        "24-hour operation with high initial battery (80% SoC), mostly clear weather, "
        "and moderate load. The agent simply needs to avoid blackouts and keep costs "
        "below the generous $8 budget. Basic buy/sell timing is sufficient to pass."
    ),
    "difficulty":     "easy",
    "pass_threshold": 0.70,
    "budget_usd":     8.00,
    "weights": {
        "uptime":   0.40,
        "economic": 0.30,
        "reserve":  0.20,
        "blackout": 0.10,
    },
}


class EasyGrader:
    task_id = "task_easy"

    def grade(self, episode_state: Dict[str, Any]) -> GradeResult:
        return grade_episode(episode_state, EASY_CONFIG)

    def describe(self) -> Dict[str, Any]:
        return {
            "task_id":        self.task_id,
            "description":    EASY_CONFIG["description"],
            "difficulty":     EASY_CONFIG["difficulty"],
            "pass_threshold": EASY_CONFIG["pass_threshold"],
            "budget_usd":     EASY_CONFIG["budget_usd"],
            "weights":        EASY_CONFIG["weights"],
        }


# ── Task 2 — MEDIUM ───────────────────────────────────────────────────────────

MEDIUM_CONFIG = {
    "task_id":        "task_medium",
    "description": (
        "Cloudy day scenario. Initial SoC is 50%. Solar generation is intermittent. "
        "The agent must use price forecasts to buy energy during cheap off-peak hours "
        "and avoid buying during expensive peak periods. Budget is tighter at $5."
    ),
    "difficulty":     "medium",
    "pass_threshold": 0.60,
    "budget_usd":     5.00,
    "weights": {
        "uptime":   0.35,
        "economic": 0.40,
        "reserve":  0.15,
        "blackout": 0.10,
    },
}


class MediumGrader:
    task_id = "task_medium"

    def grade(self, episode_state: Dict[str, Any]) -> GradeResult:
        return grade_episode(episode_state, MEDIUM_CONFIG)

    def describe(self) -> Dict[str, Any]:
        return {
            "task_id":        self.task_id,
            "description":    MEDIUM_CONFIG["description"],
            "difficulty":     MEDIUM_CONFIG["difficulty"],
            "pass_threshold": MEDIUM_CONFIG["pass_threshold"],
            "budget_usd":     MEDIUM_CONFIG["budget_usd"],
            "weights":        MEDIUM_CONFIG["weights"],
        }


# ── Task 3 — HARD ─────────────────────────────────────────────────────────────

HARD_CONFIG = {
    "task_id":        "task_hard",
    "description": (
        "Storm and peak-demand crisis. Initial SoC is critically low at 25%. "
        "Solar generation is near-zero. Grid prices spike during the storm. "
        "The agent must use load-shedding strategically to survive 24 hours "
        "without a blackout, within a very tight $3 budget. "
        "Shedding deferrable loads is acceptable; essential loads cost heavily."
    ),
    "difficulty":     "hard",
    "pass_threshold": 0.50,
    "budget_usd":     3.00,
    "weights": {
        "uptime":   0.30,
        "economic": 0.30,
        "reserve":  0.15,
        "blackout": 0.25,   # higher weight — safety is paramount in storms
    },
}


class HardGrader:
    task_id = "task_hard"

    def grade(self, episode_state: Dict[str, Any]) -> GradeResult:
        return grade_episode(episode_state, HARD_CONFIG)

    def describe(self) -> Dict[str, Any]:
        return {
            "task_id":        self.task_id,
            "description":    HARD_CONFIG["description"],
            "difficulty":     HARD_CONFIG["difficulty"],
            "pass_threshold": HARD_CONFIG["pass_threshold"],
            "budget_usd":     HARD_CONFIG["budget_usd"],
            "weights":        HARD_CONFIG["weights"],
        }


# ── Registry ──────────────────────────────────────────────────────────────────

GRADER_REGISTRY: Dict[str, Any] = {
    "task_easy":   EasyGrader(),
    "task_medium": MediumGrader(),
    "task_hard":   HardGrader(),
}


def get_grader(task_id: str):
    if task_id not in GRADER_REGISTRY:
        raise ValueError(
            f"No grader for task_id={task_id!r}. "
            f"Available: {list(GRADER_REGISTRY)}"
        )
    return GRADER_REGISTRY[task_id]
