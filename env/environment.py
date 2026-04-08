"""
MicroGridEnv — OpenEnv-compliant Micro-Grid Energy Arbitrator.

Team RauResh — IIT Mandi
-----------------------------------------------------------
Implements the mandatory OpenEnv interface:
  reset()  → GridObservation
  step()   → EpisodeResult (observation, reward, done, info)
  state()  → Dict (full internal state for graders and logging)

One timestep = one hour of grid operation.
One episode  = 24 hours (one full day).

The agent controls energy flows between:
  [Solar PV array] ──► [Battery Bank] ──► [Building Load]
                   ◄── [Main Grid]  ──►
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional

from env.models import (
    ActionType,
    BatteryState,
    EpisodeResult,
    ForecastWindow,
    GridAction,
    GridObservation,
    GridPricing,
    GridReward,
    LoadProfile,
    LoadTier,
    SolarPanel,
    WeatherCondition,
)
from env.physics import (
    BlackoutError,
    InsufficientFundsError,
    apply_charge,
    apply_discharge,
    attenuate_for_weather,
    build_forecast,
    build_load_sequence,
    build_price_sequence,
    build_weather_sequence,
    clear_sky_irradiance,
)
from env.reward import compute_reward

TASK_IDS     = ["task_easy", "task_medium", "task_hard"]
HOURS_PER_DAY = 24


# ── Default hardware configuration ───────────────────────────────────────────
# These parameters model a realistic campus micro-grid
# (similar to what IIT Mandi's solar installation uses)

DEFAULT_BATTERY = {
    "state_of_charge_pct":    80.0,   # start at 80% SoC
    "capacity_kwh":           50.0,   # 50 kWh LFP bank
    "max_charge_rate_kw":     15.0,
    "max_discharge_rate_kw":  20.0,
    "cycle_count":             0,
    "temperature_c":          28.0,
}

DEFAULT_SOLAR = {
    "rated_kw":          20.0,   # 20 kW rooftop array
    "panel_efficiency":   0.19,
    "tilt_deg":          25.0,
    "azimuth_deg":      180.0,
    "degradation_pct":    0.5,
}

INITIAL_SoC_BY_TASK = {
    "task_easy":   80.0,   # comfortable start
    "task_medium": 50.0,   # mid-range — needs planning
    "task_hard":   25.0,   # critically low — immediate action needed
}


class MicroGridEnv:
    """
    Micro-Grid Energy Arbitrator environment.

    The agent manages a solar + battery micro-grid over 24 hours,
    deciding each hour whether to:
      - BUY energy from the main grid
      - SELL surplus solar to the main grid
      - STORE solar energy in the battery
      - IDLE (let physics run passively)
      - Optionally activate LOAD-SHEDDING (Hard task)

    Episode terminates after 24 steps or on a blackout event.
    """

    metadata = {
        "name":    "microgrid-energy-arbitrator-v1",
        "version": "1.0.0",
        "team":    "RauResh",
        "task_ids": TASK_IDS,
    }

    def __init__(
        self,
        task_id:   str = "task_easy",
        max_steps: int = HOURS_PER_DAY,
        seed:      Optional[int] = 42,
    ):
        if task_id not in TASK_IDS:
            raise ValueError(f"task_id must be one of {TASK_IDS}, got {task_id!r}")
        self.task_id   = task_id
        self.max_steps = max_steps
        self.seed      = seed
        self._rng      = random.Random(seed)

        # Episode state — populated by reset()
        self._episode_id:       str = ""
        self._hour:             int = 0
        self._done:             bool = False
        self._battery:          Optional[BatteryState] = None
        self._solar:            Optional[SolarPanel] = None
        self._weather_sequence: List[WeatherCondition] = []
        self._buy_prices:       List[float] = []
        self._sell_prices:      List[float] = []
        self._load_sequence:    List[Dict[str, float]] = []
        self._blackout_count:   int = 0
        self._total_cost:       float = 0.0
        self._total_revenue:    float = 0.0
        self._hours_above_reserve: int = 0
        self._cumulative_reward:   float = 0.0
        self._episode_log:         List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> GridObservation:
        """
        Initialise a new 24-hour episode.
        Generates a fresh weather sequence, pricing curve, and load profile.
        Returns the Hour 0 observation (no action taken yet).
        """
        self._episode_id   = str(uuid.uuid4())[:8]
        self._hour         = 0
        self._done         = False
        self._blackout_count       = 0
        self._total_cost           = 0.0
        self._total_revenue        = 0.0
        self._hours_above_reserve  = 0
        self._cumulative_reward    = 0.0
        self._episode_log          = []

        # Initialise hardware
        init_soc = INITIAL_SoC_BY_TASK[self.task_id]
        battery_cfg = dict(DEFAULT_BATTERY)
        battery_cfg["state_of_charge_pct"] = init_soc
        self._battery = BatteryState(**battery_cfg)
        self._solar   = SolarPanel(**DEFAULT_SOLAR)

        # Generate deterministic scenario
        self._weather_sequence = build_weather_sequence(self.task_id, self._rng)
        self._buy_prices, self._sell_prices = build_price_sequence(self.task_id, self._rng)
        self._load_sequence = build_load_sequence(self.task_id, self._rng)

        return self._build_observation()

    def step(self, action: GridAction) -> EpisodeResult:
        """
        Advance the simulation by one hour.

        Physics order:
          1. Solar generation (passive, always happens)
          2. Agent action (buy / sell / store / idle)
          3. Load served from available sources
          4. Blackout check
          5. Reward computation
          6. Hour advance
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset() to start a new episode.")
        if self._battery is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        h   = self._hour
        info: Dict[str, Any] = {
            "hour":    h,
            "action":  action.to_string(),
            "task_id": self.task_id,
        }

        # ── 1. Passive solar generation ───────────────────────────────────
        ghi_clear = clear_sky_irradiance(h)
        irradiance = attenuate_for_weather(ghi_clear, self._weather_sequence[h], self._rng)
        ambient_t  = self._ambient_temperature(h)
        solar_output = self._solar.output_kw(irradiance, ambient_t)
        info["solar_output_kw"] = solar_output

        # Always route available solar to battery first (free energy)
        try:
            self._battery = apply_charge(self._battery, solar_output)
        except Exception:
            pass   # battery full — surplus solar wasted

        # ── 2. Agent action ───────────────────────────────────────────────
        soc_before  = self._battery.state_of_charge_pct
        blackout    = False
        shed_tier   = action.shed_tier
        qty         = action.quantity_kw

        if action.action_type == ActionType.BUY_ENERGY:
            # Buy from grid → charge battery
            try:
                self._battery  = apply_charge(self._battery, qty)
                self._total_cost += self._buy_prices[h] * qty
                info["bought_kw"] = qty
            except Exception as e:
                info["action_error"] = str(e)

        elif action.action_type == ActionType.SELL_ENERGY:
            # Discharge battery → sell to grid
            try:
                self._battery   = apply_discharge(self._battery, qty)
                self._total_revenue += self._sell_prices[h] * qty
                info["sold_kw"] = qty
            except BlackoutError as e:
                blackout = True
                info["blackout_reason"] = str(e)

        elif action.action_type == ActionType.STORE_ENERGY:
            # Already handled via passive solar routing above
            info["store_action"] = "solar already routed to battery"

        # ── 3. Serve building load from battery ───────────────────────────
        load = self._load_sequence[h]
        if shed_tier == LoadTier.DEFERRABLE:
            demand_kw = load["critical"] + load["essential"]
        elif shed_tier == LoadTier.ESSENTIAL:
            demand_kw = load["critical"]
        elif shed_tier == LoadTier.CRITICAL:
            demand_kw = 0.0   # blocked — penalty applied in reward
        else:
            demand_kw = load["critical"] + load["essential"] + load["deferrable"]

        if not blackout:
            try:
                self._battery = apply_discharge(self._battery, demand_kw)
            except BlackoutError as e:
                blackout = True
                info["blackout_reason"] = str(e)

        # ── 4. Blackout accounting ────────────────────────────────────────
        if blackout:
            self._blackout_count += 1
            self._done = True   # episode ends on blackout

        # Track reserve health
        soc_after = self._battery.state_of_charge_pct
        if soc_after >= 20.0:
            self._hours_above_reserve += 1

        # ── 5. Reward ─────────────────────────────────────────────────────
        reward_obj = compute_reward(
            action_type=action.action_type,
            quantity_kw=qty,
            buy_price=self._buy_prices[h],
            sell_price=self._sell_prices[h],
            hour=h,
            soc_before=soc_before,
            soc_after=soc_after,
            blackout_occurred=blackout,
            shed_tier=shed_tier,
            solar_output_kw=solar_output,
        )
        self._cumulative_reward += reward_obj.total
        info["reward_breakdown"]  = reward_obj.model_dump()
        info["cumulative_reward"] = round(self._cumulative_reward, 4)
        info["soc_after"]         = round(soc_after, 2)

        self._episode_log.append({
            "hour":   h,
            "action": action.to_string(),
            "reward": reward_obj.total,
            "soc":    round(soc_after, 2),
        })

        # ── 6. Advance hour ───────────────────────────────────────────────
        self._hour += 1
        if self._hour >= self.max_steps:
            self._done = True

        obs = self._build_observation()
        return EpisodeResult(
            observation=obs,
            reward=reward_obj.total,
            done=self._done,
            info=info,
            reward_breakdown=reward_obj,
        )

    def state(self) -> Dict[str, Any]:
        """Full internal state — used by graders and logging."""
        if self._battery is None:
            return {"initialised": False}
        return {
            "episode_id":          self._episode_id,
            "task_id":             self.task_id,
            "hour":                self._hour,
            "max_steps":           self.max_steps,
            "done":                self._done,
            "battery_soc_pct":     round(self._battery.state_of_charge_pct, 2),
            "battery_capacity_kwh":self._battery.capacity_kwh,
            "blackout_count":      self._blackout_count,
            "total_cost_usd":      round(self._total_cost, 4),
            "total_revenue_usd":   round(self._total_revenue, 4),
            "net_cost_usd":        round(self._total_cost - self._total_revenue, 4),
            "hours_above_reserve": self._hours_above_reserve,
            "cumulative_reward":   round(self._cumulative_reward, 4),
            "weather_sequence":    [w.value for w in self._weather_sequence],
            "buy_prices":          self._buy_prices,
            "sell_prices":         self._sell_prices,
            "episode_log":         copy.deepcopy(self._episode_log),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ambient_temperature(self, hour: int) -> float:
        """
        Diurnal temperature model for IIT Mandi (Himachal Pradesh).
        Min ~18°C at 5am, max ~32°C at 2pm.
        """
        base = 25.0
        amplitude = 7.0
        peak_hour = 14
        temp = base + amplitude * math.sin(math.pi * (hour - 5) / (peak_hour - 5))
        return round(temp + self._rng.gauss(0.0, 0.5), 2)

    def _build_observation(self) -> GridObservation:
        h   = min(self._hour, self.max_steps - 1)
        load = self._load_sequence[h]
        ghi  = clear_sky_irradiance(h)
        irr  = attenuate_for_weather(ghi, self._weather_sequence[h], self._rng)
        amb  = self._ambient_temperature(h)
        solar_out = self._solar.output_kw(irr, amb)

        forecast = build_forecast(
            hour=h,
            weather_sequence=self._weather_sequence,
            buy_prices=self._buy_prices,
            load_sequence=self._load_sequence,
            horizon=6,
            rng=self._rng,
        )

        return GridObservation(
            episode_id=self._episode_id,
            task_id=self.task_id,
            hour=h,
            day=1,
            step_number=self._hour,
            max_steps=self.max_steps,
            battery=self._battery.model_copy(),
            solar=self._solar.model_copy(),
            current_solar_output_kw=solar_out,
            irradiance_wm2=irr,
            ambient_temp_c=amb,
            weather=self._weather_sequence[h],
            pricing=GridPricing(
                buy_price_per_kwh=self._buy_prices[h],
                sell_price_per_kwh=self._sell_prices[h],
                peak_hours=list(range(7, 11)) + list(range(18, 22)),
                is_peak_hour=(h in {7,8,9,10,18,19,20,21}),
            ),
            load=LoadProfile(
                total_demand_kw=load["critical"] + load["essential"] + load["deferrable"],
                critical_kw=load["critical"],
                essential_kw=load["essential"],
                deferrable_kw=load["deferrable"],
            ),
            forecast=forecast,
            total_cost_usd=round(self._total_cost, 4),
            total_revenue_usd=round(self._total_revenue, 4),
            blackout_count=self._blackout_count,
            hours_above_reserve=self._hours_above_reserve,
            cumulative_reward=round(self._cumulative_reward, 4),
            info={"done": self._done},
        )


# Fix missing math import
import math
