"""
Pydantic models for the Micro-Grid Energy Arbitrator OpenEnv.

Team RauResh — IIT Mandi
-----------------------------------------------------------
Design philosophy:
  GridObservation captures every physical signal the agent
  can actually observe on a real micro-grid (voltage, SoC,
  spot price, irradiance forecast).  GridAction is kept to
  three atomic decisions so the action space stays discrete
  but meaningful.  GridReward decomposes into named economic
  and safety sub-signals so researchers can study agent
  behaviour component-by-component.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    BUY_ENERGY   = "buy_energy"    # purchase from the main grid
    SELL_ENERGY  = "sell_energy"   # export surplus to main grid
    STORE_ENERGY = "store_energy"  # route solar → battery only
    IDLE         = "idle"          # hold current state, no trade


class WeatherCondition(str, Enum):
    CLEAR   = "clear"
    PARTIAL = "partial_cloud"
    OVERCAST= "overcast"
    STORM   = "storm"


class LoadTier(str, Enum):
    """Priority tiers for load-shedding (Hard task mechanic)."""
    CRITICAL      = "critical"       # hospital / server room — never shed
    ESSENTIAL     = "essential"      # lighting / refrigeration
    DEFERRABLE    = "deferrable"     # HVAC / EV charging


# ── Sub-models ────────────────────────────────────────────────────────────────

class BatteryState(BaseModel):
    """
    Physics-aware battery representation.
    Depth-of-discharge (DoD) tracking prevents over-cycling damage.
    Efficiency loss formula derived from standard Li-ion datasheets.
    """
    state_of_charge_pct: float = Field(..., ge=0.0, le=100.0,
        description="Current SoC as percentage of rated capacity")
    capacity_kwh:        float = Field(..., gt=0.0,
        description="Rated capacity in kWh")
    max_charge_rate_kw:  float = Field(..., gt=0.0,
        description="Maximum charge power in kW")
    max_discharge_rate_kw: float = Field(..., gt=0.0,
        description="Maximum discharge power in kW")
    cycle_count:         int   = Field(default=0,
        description="Charge/discharge cycle counter — affects degradation")
    temperature_c:       float = Field(default=25.0,
        description="Cell temperature affects usable capacity")

    @property
    def available_kwh(self) -> float:
        """Energy available for discharge, accounting for minimum SoC floor."""
        usable_floor = 0.10  # never discharge below 10% SoC
        return max(0.0, (self.state_of_charge_pct - usable_floor * 100) / 100
                   * self.capacity_kwh)

    @property
    def headroom_kwh(self) -> float:
        """Free space for charging."""
        return (1.0 - self.state_of_charge_pct / 100) * self.capacity_kwh


class SolarPanel(BaseModel):
    """Simulated PV array with irradiance-based output."""
    rated_kw:         float = Field(..., gt=0.0, description="Peak nameplate power")
    panel_efficiency: float = Field(default=0.19, ge=0.05, le=0.30,
        description="Panel efficiency (default: 19% — typical mono-Si)")
    tilt_deg:         float = Field(default=25.0,
        description="Panel tilt angle in degrees")
    azimuth_deg:      float = Field(default=180.0,
        description="Panel azimuth (180 = south-facing)")
    degradation_pct:  float = Field(default=0.5,
        description="Annual degradation rate percent")

    def output_kw(self, irradiance_wm2: float, temperature_c: float) -> float:
        """
        Calculate actual output using irradiance and temperature derating.
        Temperature coefficient: -0.4%/°C above STC (25°C).
        Sensor noise is injected at the environment level, not here.
        """
        temp_derate = 1.0 - 0.004 * max(0.0, temperature_c - 25.0)
        raw = self.rated_kw * (irradiance_wm2 / 1000.0) * temp_derate
        return round(max(0.0, raw), 4)


class GridPricing(BaseModel):
    """Time-of-use electricity pricing model."""
    buy_price_per_kwh:  float = Field(..., ge=0.0,
        description="Current spot price to purchase from main grid ($/kWh)")
    sell_price_per_kwh: float = Field(..., ge=0.0,
        description="Feed-in tariff for exporting to main grid ($/kWh)")
    peak_hours:         List[int] = Field(default_factory=list,
        description="Hours (0-23) classified as peak pricing")
    is_peak_hour:       bool  = Field(default=False)


class LoadProfile(BaseModel):
    """Building energy demand broken down by priority tier."""
    total_demand_kw:     float = Field(..., ge=0.0)
    critical_kw:         float = Field(..., ge=0.0,
        description="Non-shedable load — always served")
    essential_kw:        float = Field(..., ge=0.0)
    deferrable_kw:       float = Field(..., ge=0.0)
    shed_active:         bool  = Field(default=False,
        description="Whether load-shedding is currently engaged")
    shed_tier:           Optional[LoadTier] = Field(default=None)


class ForecastWindow(BaseModel):
    """
    Short-horizon forecasts visible to the agent.
    Intentionally noisy — the agent must reason under uncertainty.
    """
    irradiance_forecast_wm2: List[float] = Field(...,
        description="Irradiance forecast for next N hours")
    price_forecast:          List[float] = Field(...,
        description="Expected spot price for next N hours")
    load_forecast_kw:        List[float] = Field(...,
        description="Expected building load for next N hours")
    weather_sequence:        List[str]   = Field(...,
        description="Predicted weather condition per hour")
    forecast_horizon_hours:  int         = Field(default=6)


# ── Primary Models ────────────────────────────────────────────────────────────

class GridObservation(BaseModel):
    """
    Complete observable state of the micro-grid at one timestep.
    Mirrors what a real SCADA / EMS dashboard would show an operator.
    """
    # Identifiers
    episode_id:      str = Field(..., description="Unique episode identifier")
    task_id:         str = Field(..., description="task_easy | task_medium | task_hard")
    hour:            int = Field(..., ge=0, le=23, description="Current hour (0-23)")
    day:             int = Field(..., ge=1, description="Simulation day number")
    step_number:     int = Field(default=0)
    max_steps:       int = Field(default=24)

    # Physical state
    battery:         BatteryState
    solar:           SolarPanel
    current_solar_output_kw: float = Field(..., ge=0.0,
        description="Actual solar generation this hour (with sensor noise)")
    irradiance_wm2:  float = Field(..., ge=0.0,
        description="Observed solar irradiance W/m²")
    ambient_temp_c:  float = Field(...,
        description="Ambient air temperature °C")
    weather:         WeatherCondition

    # Economic signals
    pricing:         GridPricing
    load:            LoadProfile

    # Forecasts (uncertain — agent must plan ahead)
    forecast:        ForecastWindow

    # Episode accounting
    total_cost_usd:       float = Field(default=0.0)
    total_revenue_usd:    float = Field(default=0.0)
    blackout_count:       int   = Field(default=0)
    hours_above_reserve:  int   = Field(default=0,
        description="Hours battery stayed above 20% SoC — partial progress signal")
    cumulative_reward:    float = Field(default=0.0)
    info:                 Dict[str, Any] = Field(default_factory=dict)


class GridAction(BaseModel):
    """
    One agent decision per timestep.

    action_type : BUY_ENERGY | SELL_ENERGY | STORE_ENERGY | IDLE
    quantity_kw : Power magnitude in kW (agent chooses how aggressively to act)
    shed_tier   : Optional load tier to shed (Hard task only)
    """
    action_type: ActionType = Field(..., description="Primary grid action")
    quantity_kw: float      = Field(default=0.0, ge=0.0, le=100.0,
        description="Power in kW to buy/sell/store")
    shed_tier:   Optional[LoadTier] = Field(default=None,
        description="Activate load-shedding at this priority tier (Hard task)")

    @classmethod
    def from_string(cls, action_str: str) -> "GridAction":
        """
        Parse strings like:
          'buy_energy:5.0'
          'sell_energy:3.5'
          'store_energy:0'
          'idle'
          'buy_energy:10.0:shed_deferrable'
        """
        parts = action_str.strip().lower().split(":")
        atype = ActionType(parts[0])
        qty   = float(parts[1]) if len(parts) > 1 else 0.0
        shed  = None
        if len(parts) > 2 and parts[2].startswith("shed_"):
            tier_str = parts[2].replace("shed_", "")
            shed = LoadTier(tier_str)
        return cls(action_type=atype, quantity_kw=qty, shed_tier=shed)

    def to_string(self) -> str:
        base = f"{self.action_type.value}:{self.quantity_kw:.2f}"
        if self.shed_tier:
            base += f":shed_{self.shed_tier.value}"
        return base


class GridReward(BaseModel):
    """
    Decomposed reward — every component is named and interpretable.
    Researchers can ablate individual components to study agent behaviour.

    Team RauResh reward philosophy:
      - Economic efficiency drives the bulk of the signal
      - Safety (blackout avoidance) is non-negotiable — hard penalty
      - Partial progress via battery reserve bonus every step
      - Load-shed penalty discourages premature power cuts
    """
    total:              float = Field(..., description="Scalar reward for this step")
    economic_gain:      float = Field(default=0.0,
        description="Revenue from selling minus cost of buying ($/step)")
    blackout_penalty:   float = Field(default=0.0,
        description="Large negative if SoC hits 0% — grid failure")
    reserve_bonus:      float = Field(default=0.0,
        description="Small positive for every step above 20% SoC reserve")
    shed_penalty:       float = Field(default=0.0,
        description="Penalty for load-shedding — proportional to tier criticality")
    efficiency_bonus:   float = Field(default=0.0,
        description="Bonus for buying during low-price hours (smart arbitrage)")
    step_cost:          float = Field(default=-0.01,
        description="Tiny fixed cost per step — encourages decisive action")
    info:               Dict[str, Any] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    """Return type for env.step()."""
    observation:      GridObservation
    reward:           float
    done:             bool
    info:             Dict[str, Any] = Field(default_factory=dict)
    reward_breakdown: GridReward
