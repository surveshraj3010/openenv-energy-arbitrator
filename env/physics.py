"""
Physics and market simulation for the Micro-Grid Energy Arbitrator.

Team RauResh — IIT Mandi
-----------------------------------------------------------
All physical constants and formulas are grounded in real-world
energy engineering:

  Solar irradiance model   : clear-sky curve + cloud attenuation
  Battery degradation      : Peukert-inspired depth-of-discharge
  Load profiles            : ISO 50001 building energy archetypes
  Spot pricing             : Day-ahead market with peak/off-peak TOU
  Sensor noise             : Gaussian, calibrated to ±3% RMSE
    (realistic for low-cost irradiance sensors on ESP32/ADS1115)

Why Gaussian noise?
  In hardware deployments (ESP32 + HACS current clamps) the ADC
  quantisation noise dominates at short timescales and is well
  approximated as additive Gaussian.  A seed-controlled RNG makes
  noise reproducible across runs for fair evaluation.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

from env.models import (
    BatteryState,
    ForecastWindow,
    GridPricing,
    LoadProfile,
    LoadTier,
    SolarPanel,
    WeatherCondition,
)


# ── Weather attenuation factors ───────────────────────────────────────────────
CLOUD_ATTENUATION: Dict[WeatherCondition, Tuple[float, float]] = {
    WeatherCondition.CLEAR:    (0.90, 0.97),   # (min, max) fraction of clear-sky GHI
    WeatherCondition.PARTIAL:  (0.45, 0.75),
    WeatherCondition.OVERCAST: (0.10, 0.30),
    WeatherCondition.STORM:    (0.00, 0.08),
}

# ── Peak pricing hours (typical Indian industrial TOU tariff) ─────────────────
PEAK_HOURS = [7, 8, 9, 10, 18, 19, 20, 21]   # morning + evening peak

# ── Base load archetypes (kW) — scaled by task scenario ──────────────────────
LOAD_ARCHETYPES: Dict[str, Dict[str, float]] = {
    # hour 0-5 / 6-11 / 12-17 / 18-23
    "office": {
        "night":   {"critical": 1.0, "essential": 0.5, "deferrable": 0.2},
        "morning": {"critical": 1.0, "essential": 4.0, "deferrable": 6.0},
        "afternoon":{"critical":1.0, "essential": 3.5, "deferrable": 7.5},
        "evening": {"critical": 1.0, "essential": 2.0, "deferrable": 3.0},
    }
}


def hour_to_slot(hour: int) -> str:
    if   0  <= hour <= 5:  return "night"
    elif 6  <= hour <= 11: return "morning"
    elif 12 <= hour <= 17: return "afternoon"
    else:                  return "evening"


# ── Solar irradiance model ────────────────────────────────────────────────────

def clear_sky_irradiance(hour: int, latitude_deg: float = 31.7) -> float:
    """
    Simplified clear-sky Global Horizontal Irradiance (GHI) in W/m².
    Uses a sine-curve approximation for the solar elevation angle.
    Calibrated for IIT Mandi latitude (~31.7°N).
    Returns 0 outside daylight hours.
    """
    # Solar noon at hour 12; daylight roughly 6h to 18h at mid-latitudes
    if hour < 6 or hour > 18:
        return 0.0
    angle = math.pi * (hour - 6) / 12.0          # 0 at sunrise, π at sunset
    elevation = math.sin(angle)
    # Peak GHI ~950 W/m² for clear sky at this latitude in spring
    ghi = 950.0 * elevation * math.cos(math.radians(latitude_deg - 23.5))
    return max(0.0, round(ghi, 2))


def attenuate_for_weather(
    ghi: float,
    weather: WeatherCondition,
    rng: random.Random,
) -> float:
    """Apply cloud attenuation and add sensor noise."""
    lo, hi = CLOUD_ATTENUATION[weather]
    factor = rng.uniform(lo, hi)
    attenuated = ghi * factor
    # Gaussian sensor noise: ±3% RMSE (ESP32 ADC + irradiance sensor spec)
    noise = rng.gauss(0.0, attenuated * 0.03)
    return max(0.0, round(attenuated + noise, 2))


def build_weather_sequence(
    task_id: str,
    rng: random.Random,
) -> List[WeatherCondition]:
    """
    Generate a 24-hour weather sequence appropriate for each task.

    Easy   : mostly clear, guaranteed solar generation
    Medium : mixed cloud — agent must forecast and time purchases
    Hard   : storm / overcast dominant — near-zero solar, peak prices
    """
    if task_id == "task_easy":
        weights = {
            WeatherCondition.CLEAR:    0.80,
            WeatherCondition.PARTIAL:  0.15,
            WeatherCondition.OVERCAST: 0.05,
            WeatherCondition.STORM:    0.00,
        }
    elif task_id == "task_medium":
        weights = {
            WeatherCondition.CLEAR:    0.35,
            WeatherCondition.PARTIAL:  0.40,
            WeatherCondition.OVERCAST: 0.20,
            WeatherCondition.STORM:    0.05,
        }
    else:  # task_hard
        weights = {
            WeatherCondition.CLEAR:    0.05,
            WeatherCondition.PARTIAL:  0.15,
            WeatherCondition.OVERCAST: 0.35,
            WeatherCondition.STORM:    0.45,
        }

    population = list(weights.keys())
    wts        = list(weights.values())
    sequence: List[WeatherCondition] = []

    # Markov-style: weather persists with 70% probability
    current = rng.choices(population, weights=wts, k=1)[0]
    for _ in range(24):
        if rng.random() < 0.70:
            sequence.append(current)
        else:
            current = rng.choices(population, weights=wts, k=1)[0]
            sequence.append(current)
    return sequence


# ── Electricity pricing model ─────────────────────────────────────────────────

def build_price_sequence(
    task_id: str,
    rng: random.Random,
) -> Tuple[List[float], List[float]]:
    """
    Return (buy_prices, sell_prices) for 24 hours.

    Base rate reflects Indian C&I tariff (~₹8/kWh ≈ $0.096).
    Peak multiplier: 1.6× during PEAK_HOURS.
    Feed-in tariff: 60% of buy price (standard net-metering).
    Hard task adds a volatility spike during storm hours.
    """
    base_buy = 0.096   # $/kWh
    buy_prices, sell_prices = [], []

    for h in range(24):
        multiplier = 1.6 if h in PEAK_HOURS else 1.0
        if task_id == "task_hard":
            # Storm premium — grid stress drives prices up
            spike = rng.uniform(1.0, 1.8)
            multiplier *= spike
        noise = rng.gauss(0.0, base_buy * 0.05)
        buy_p  = round(max(0.01, base_buy * multiplier + noise), 4)
        sell_p = round(buy_p * 0.60, 4)
        buy_prices.append(buy_p)
        sell_prices.append(sell_p)

    return buy_prices, sell_prices


# ── Load profile builder ──────────────────────────────────────────────────────

def build_load_sequence(
    task_id: str,
    rng: random.Random,
) -> List[Dict[str, float]]:
    """
    Generate 24-hour building load profile (kW per tier).
    Hard task scales up total demand by 40% — stress scenario.
    """
    scale = {"task_easy": 1.0, "task_medium": 1.2, "task_hard": 1.6}[task_id]
    archetype = LOAD_ARCHETYPES["office"]
    loads = []
    for h in range(24):
        slot = hour_to_slot(h)
        tier = archetype[slot]
        noise_frac = rng.gauss(0.0, 0.05)   # ±5% load variation
        loads.append({
            "critical":   round(tier["critical"] * scale, 3),
            "essential":  round(tier["essential"] * scale * (1 + noise_frac), 3),
            "deferrable": round(tier["deferrable"] * scale * (1 + noise_frac), 3),
        })
    return loads


# ── Battery physics ───────────────────────────────────────────────────────────

class BlackoutError(Exception):
    """Raised when battery SoC hits zero — grid failure event."""
    pass


class InsufficientFundsError(Exception):
    """Raised when agent tries to buy more energy than budget allows."""
    pass


def apply_charge(battery: BatteryState, energy_kw: float, dt_hours: float = 1.0) -> BatteryState:
    """
    Charge the battery by energy_kw for dt_hours.
    Round-trip efficiency: 92% (LFP chemistry typical).
    Temperature derating: -0.5% per °C above 35°C.
    """
    efficiency = 0.92
    if battery.temperature_c > 35.0:
        efficiency *= 1.0 - 0.005 * (battery.temperature_c - 35.0)

    actual_kw   = min(energy_kw, battery.max_charge_rate_kw, battery.headroom_kwh / dt_hours)
    energy_in   = actual_kw * dt_hours * efficiency
    new_soc     = min(100.0, battery.state_of_charge_pct + (energy_in / battery.capacity_kwh) * 100)

    return battery.model_copy(update={
        "state_of_charge_pct": round(new_soc, 3),
        "cycle_count": battery.cycle_count,
    })


def apply_discharge(battery: BatteryState, energy_kw: float, dt_hours: float = 1.0) -> BatteryState:
    """
    Discharge battery by energy_kw for dt_hours.
    Raises BlackoutError if available energy is insufficient.
    Peukert correction: high discharge rates slightly reduce usable capacity.
    """
    peukert_factor = 1.0 + 0.02 * max(0.0, energy_kw / battery.max_discharge_rate_kw - 0.5)
    actual_draw    = min(energy_kw * peukert_factor, battery.max_discharge_rate_kw)
    energy_out     = actual_draw * dt_hours

    if energy_out > battery.available_kwh + 0.001:
        raise BlackoutError(
            f"Battery depleted: needed {energy_out:.2f} kWh, "
            f"available {battery.available_kwh:.2f} kWh"
        )

    new_soc = max(0.0, battery.state_of_charge_pct - (energy_out / battery.capacity_kwh) * 100)
    new_cycles = battery.cycle_count + round(energy_out / battery.capacity_kwh, 3)

    return battery.model_copy(update={
        "state_of_charge_pct": round(new_soc, 3),
        "cycle_count": round(new_cycles, 3),
    })


# ── Forecast builder ──────────────────────────────────────────────────────────

def build_forecast(
    hour: int,
    weather_sequence: List[WeatherCondition],
    buy_prices: List[float],
    load_sequence: List[Dict[str, float]],
    horizon: int = 6,
    rng: random.Random = None,
) -> ForecastWindow:
    """
    Build a noisy short-horizon forecast the agent can see.
    Irradiance and price forecasts have ±10% forecast error injected.
    """
    if rng is None:
        rng = random.Random()

    future_hours = list(range(hour + 1, min(hour + 1 + horizon, 24)))
    irr_forecast, price_forecast, load_forecast, wx_seq = [], [], [], []

    for fh in future_hours:
        ghi  = clear_sky_irradiance(fh)
        wx   = weather_sequence[fh]
        lo, hi = CLOUD_ATTENUATION[wx]
        irr  = ghi * rng.uniform(lo, hi) * rng.gauss(1.0, 0.10)
        irr_forecast.append(round(max(0.0, irr), 2))

        price_err = rng.gauss(1.0, 0.10)
        price_forecast.append(round(buy_prices[fh] * price_err, 4))

        ld = load_sequence[fh]
        load_forecast.append(round((ld["critical"] + ld["essential"] + ld["deferrable"]) * rng.gauss(1.0, 0.05), 3))
        wx_seq.append(wx.value)

    # Pad to horizon if near end of day
    while len(irr_forecast) < horizon:
        irr_forecast.append(0.0)
        price_forecast.append(price_forecast[-1] if price_forecast else 0.096)
        load_forecast.append(load_forecast[-1] if load_forecast else 1.0)
        wx_seq.append(WeatherCondition.CLEAR.value)

    return ForecastWindow(
        irradiance_forecast_wm2=irr_forecast[:horizon],
        price_forecast=price_forecast[:horizon],
        load_forecast_kw=load_forecast[:horizon],
        weather_sequence=wx_seq[:horizon],
        forecast_horizon_hours=horizon,
    )
