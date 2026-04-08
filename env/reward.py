"""
Reward function for the Micro-Grid Energy Arbitrator.

Team RauResh — IIT Mandi
-----------------------------------------------------------
Design rationale (documented for Phase 3 human review):

  1. Economic gain is the primary signal — agents that lose money
     consistently cannot be deployed in real micro-grids.

  2. Blackout penalty (-50.0) is intentionally large relative to
     hourly gains (~0.5–2.0 $/hr) so the agent learns that uptime
     is non-negotiable.  This mirrors real DISCOM penalty clauses.

  3. Reserve bonus (+0.15/hr above 20% SoC) provides dense partial-
     progress signal every step — prevents sparse-reward problems.

  4. Load-shed penalty is tiered: shedding deferrable loads costs
     less than shedding essential loads.  Critical loads can never
     be shed — the environment hard-blocks those actions.

  5. Efficiency bonus rewards buying during off-peak hours and
     selling during peak — the core arbitrage objective.

  6. A tiny fixed step cost (-0.01) prevents the agent from running
     idle indefinitely without economic consequence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env.models import ActionType, GridReward, LoadTier


# ── Reward constants ──────────────────────────────────────────────────────────
BLACKOUT_PENALTY        = -50.0   # grid failure — catastrophic
RESERVE_BONUS_PER_STEP  =  0.15   # battery SoC above 20% reserve
LOW_SOC_WARNING_PENALTY = -0.50   # SoC between 10-20% — caution zone
SELL_PEAK_BONUS         =  0.30   # extra reward for selling during peak hours
BUY_OFFPEAK_BONUS       =  0.20   # extra reward for buying during cheap hours
SHED_DEFERRABLE_PENALTY = -1.00   # load shed tier: deferrable
SHED_ESSENTIAL_PENALTY  = -5.00   # load shed tier: essential
SHED_CRITICAL_PENALTY   = -20.0   # load shed tier: critical (should never happen)
STEP_FIXED_COST         = -0.01   # per-step overhead
PEAK_HOURS              = {7, 8, 9, 10, 18, 19, 20, 21}


def compute_reward(
    action_type:        ActionType,
    quantity_kw:        float,
    buy_price:          float,
    sell_price:         float,
    hour:               int,
    soc_before:         float,   # SoC % before action
    soc_after:          float,   # SoC % after action
    blackout_occurred:  bool,
    shed_tier:          Optional[LoadTier],
    solar_output_kw:    float,
) -> GridReward:
    """
    Compute shaped reward for one timestep.

    All monetary values are in USD to keep the scale intuitive.
    The agent's goal: maximise cumulative reward over 24 hours.
    """
    economic_gain    = 0.0
    blackout_pen     = 0.0
    reserve_bonus    = 0.0
    shed_pen         = 0.0
    efficiency_bonus = 0.0
    info: Dict[str, Any] = {}

    # ── Blackout (immediate, non-recoverable this episode) ────────────────
    if blackout_occurred:
        blackout_pen = BLACKOUT_PENALTY
        info["blackout"] = True

    else:
        # ── Economic gain ─────────────────────────────────────────────────
        if action_type == ActionType.BUY_ENERGY:
            economic_gain = -(buy_price * quantity_kw)   # cost
            info["cost_usd"] = round(-economic_gain, 4)

        elif action_type == ActionType.SELL_ENERGY:
            economic_gain = sell_price * quantity_kw     # revenue
            info["revenue_usd"] = round(economic_gain, 4)

        elif action_type == ActionType.STORE_ENERGY:
            # No direct economic gain but preserves battery for later
            economic_gain = 0.0

        # ── Arbitrage efficiency bonus ────────────────────────────────────
        if action_type == ActionType.BUY_ENERGY and hour not in PEAK_HOURS:
            efficiency_bonus = BUY_OFFPEAK_BONUS * min(1.0, quantity_kw / 10.0)
            info["offpeak_buy_bonus"] = True

        if action_type == ActionType.SELL_ENERGY and hour in PEAK_HOURS:
            efficiency_bonus = SELL_PEAK_BONUS * min(1.0, quantity_kw / 10.0)
            info["peak_sell_bonus"] = True

        # ── Battery reserve bonus / warning ──────────────────────────────
        if soc_after >= 20.0:
            reserve_bonus = RESERVE_BONUS_PER_STEP
            info["reserve_healthy"] = True
        elif soc_after >= 10.0:
            reserve_bonus = LOW_SOC_WARNING_PENALTY
            info["low_soc_warning"] = True

        # ── Load-shedding penalty ─────────────────────────────────────────
        if shed_tier is not None:
            if shed_tier == LoadTier.DEFERRABLE:
                shed_pen = SHED_DEFERRABLE_PENALTY
            elif shed_tier == LoadTier.ESSENTIAL:
                shed_pen = SHED_ESSENTIAL_PENALTY
            elif shed_tier == LoadTier.CRITICAL:
                shed_pen = SHED_CRITICAL_PENALTY
            info["shed_tier"] = shed_tier.value

    total = (
        economic_gain
        + blackout_pen
        + reserve_bonus
        + shed_pen
        + efficiency_bonus
        + STEP_FIXED_COST
    )

    return GridReward(
        total=round(total, 4),
        economic_gain=round(economic_gain, 4),
        blackout_penalty=round(blackout_pen, 4),
        reserve_bonus=round(reserve_bonus, 4),
        shed_penalty=round(shed_pen, 4),
        efficiency_bonus=round(efficiency_bonus, 4),
        step_cost=STEP_FIXED_COST,
        info=info,
    )
