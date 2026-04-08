---
title: Micro-Grid Energy Arbitrator OpenEnv
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - energy
  - micro-grid
  - solar
  - battery
  - reinforcement-learning
  - sustainability
---

# Micro-Grid Energy Arbitrator — OpenEnv

**Team RauResh — IIT Mandi**

An OpenEnv-compliant environment where an AI agent manages energy flows
across a solar PV array, lithium battery bank, and main electricity grid
over a 24-hour horizon. Grounded in real hardware constraints (ESP32 sensor
noise, LFP battery chemistry, Indian C&I tariff structures) and modelled
after IIT Mandi's campus micro-grid.

---

## Why This Domain

Grid-scale energy management is one of the most consequential optimisation
problems of the next decade. A poorly-timed energy purchase during peak
hours costs money. A depleted battery during a storm causes a blackout.
This environment forces an agent to reason about:

- **Uncertainty** — solar forecasts are noisy; weather changes hour-to-hour
- **Arbitrage** — buy cheap at midnight, sell expensive at peak
- **Safety** — never let battery SoC hit zero (catastrophic penalty)
- **Load priority** — shed deferrable loads before essential ones in a crisis

---

## Environment Description

One episode = 24 hours of micro-grid operation.
One timestep = 1 hour.

The agent observes the full grid state each hour and chooses one action:

| Action | Description |
|--------|-------------|
| `buy_energy:<kw>` | Purchase kW from main grid → charges battery |
| `sell_energy:<kw>` | Export kW from battery → revenue |
| `store_energy:0` | Route solar to battery, no grid trade |
| `idle:0` | No transaction this hour |
| `buy_energy:<kw>:shed_deferrable` | Buy + shed deferrable loads |

Hardware configuration (modelled from real specs):
- **Solar array**: 20 kW rooftop mono-Si PV
- **Battery bank**: 50 kWh LFP lithium iron phosphate
- **Location**: IIT Mandi, Himachal Pradesh (31.7°N, 76.9°E)
- **Sensor noise**: Gaussian ±3% RMSE (ESP32 + ADS1115 ADC)

---

## Observation Space

```json
{
  "episode_id": "a3f9bc12",
  "task_id": "task_easy",
  "hour": 9,
  "battery": {
    "state_of_charge_pct": 62.4,
    "capacity_kwh": 50.0,
    "max_charge_rate_kw": 15.0,
    "max_discharge_rate_kw": 20.0,
    "temperature_c": 28.5
  },
  "current_solar_output_kw": 14.2,
  "irradiance_wm2": 748.3,
  "weather": "clear",
  "pricing": {
    "buy_price_per_kwh": 0.1536,
    "sell_price_per_kwh": 0.0922,
    "is_peak_hour": true
  },
  "load": {
    "total_demand_kw": 11.6,
    "critical_kw": 1.0,
    "essential_kw": 4.8,
    "deferrable_kw": 5.8
  },
  "forecast": {
    "irradiance_forecast_wm2": [812, 780, 690, 540, 340, 120],
    "price_forecast": [0.154, 0.155, 0.096, 0.096, 0.154, 0.153],
    "weather_sequence": ["clear","clear","partial_cloud","overcast","clear","clear"]
  },
  "total_cost_usd": 0.42,
  "total_revenue_usd": 0.18,
  "blackout_count": 0,
  "hours_above_reserve": 9,
  "cumulative_reward": 1.24
}
```

---

## Reward Function

Shaped signal every timestep — never sparse:

| Component | Range | Description |
|-----------|-------|-------------|
| Economic gain | -5.0 → +3.0 | Revenue minus cost per hour |
| Blackout penalty | -50.0 | Immediate on SoC = 0% |
| Reserve bonus | +0.15/hr | Battery SoC above 20% floor |
| Low SoC warning | -0.50/hr | SoC between 10–20% |
| Efficiency bonus | 0 → +0.30 | Buy off-peak / sell peak |
| Shed penalty | -1.0 to -20.0 | Load-shedding by tier |
| Step cost | -0.01 | Fixed operational overhead |

---

## Tasks

### Task 1 — Easy (`task_easy`) | Pass threshold: 0.70

- Initial SoC: **80%**
- Weather: mostly clear (80% clear probability)
- Budget ceiling: **$8.00**

24-hour operation with comfortable battery and good solar. The agent
needs to avoid blackouts and keep costs below budget. Basic buy/sell
timing on the price curve is sufficient.

### Task 2 — Medium (`task_medium`) | Pass threshold: 0.60

- Initial SoC: **50%**
- Weather: mixed cloud (40% partial, 35% clear)
- Budget ceiling: **$5.00**

Intermittent solar means the agent cannot rely on free generation.
It must use the 6-hour price forecast to buy during cheap off-peak
windows and avoid over-spending during peak hours.

### Task 3 — Hard (`task_hard`) | Pass threshold: 0.50

- Initial SoC: **25% (critically low)**
- Weather: storm-dominant (45% storm, 35% overcast)
- Budget ceiling: **$3.00**

Near-zero solar. Price spikes during the storm. The agent must
make strategic load-shedding decisions (deferrable loads first)
while buying only when absolutely necessary to prevent blackout.

---

## Grader Weights

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Uptime (no blackout) | 40% | 35% | 30% |
| Economic efficiency | 30% | 40% | 30% |
| Battery reserve | 20% | 15% | 15% |
| Blackout safety gate | 10% | 10% | 25% |

---

## Baseline Scores

Tested with `meta-llama/Llama-3.1-8B-Instruct` (seed=42):

| Task | Avg Score | Pass Rate | Avg Blackouts |
|------|-----------|-----------|---------------|
| task_easy   | ~0.60 | ~67% | ~0.3 |
| task_medium | ~0.45 | ~33% | ~0.7 |
| task_hard   | ~0.30 |  ~0% | ~1.2 |

---

## Setup & Usage

### Local

```bash
git clone <repo>
cd openenv-microgrid
pip install -r requirements.txt

# Start API server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
docker build -t microgrid-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  microgrid-env
```

### Python SDK

```python
from env import MicroGridEnv, GridAction, ActionType
from graders import get_grader

env = MicroGridEnv(task_id="task_easy", seed=42)
obs = env.reset()

# Buy 5 kW off-peak
action = GridAction(action_type=ActionType.BUY_ENERGY, quantity_kw=5.0)
result = env.step(action)
print(result.reward)          # shaped reward this hour
print(result.observation.battery.state_of_charge_pct)  # SoC after

# Run 24 hours then grade
for _ in range(23):
    env.step(GridAction(action_type=ActionType.IDLE, quantity_kw=0.0))

grade = get_grader("task_easy").grade(env.state())
print(grade.score)   # 0.0 – 1.0
```

### API

```python
import requests
BASE = "http://localhost:7860"

# Start episode
r = requests.post(f"{BASE}/reset", json={"task_id": "task_medium", "seed": 42})
session_id = r.json()["session_id"]

# Buy 5 kW
r = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action_type": "buy_energy",
    "quantity_kw": 5.0
})
print(r.json()["reward"])

# Grade
r = requests.post(f"{BASE}/grade", json={"session_id": session_id})
print(r.json()["score"])
```

---

## Project Structure

```
openenv-microgrid/
├── app.py               # FastAPI server — HF Spaces entrypoint
├── inference.py         # Baseline inference script (mandatory)
├── openenv.yaml         # OpenEnv spec file
├── validate.py          # Pre-submission validator
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py        # GridObservation, GridAction, GridReward (Pydantic)
│   ├── physics.py       # Solar irradiance, battery chemistry, weather, pricing
│   ├── environment.py   # MicroGridEnv — reset() / step() / state()
│   └── reward.py        # 6-component shaped reward function
├── graders/
│   ├── __init__.py
│   └── graders.py       # EasyGrader, MediumGrader, HardGrader
└── tests/
    └── test_env.py      # 46 unit tests
```

---

## Physics Notes

All physical constants are grounded in real hardware:

- **Battery efficiency**: 92% round-trip (LFP chemistry datasheet)
- **Temperature derating**: -0.4%/°C panel output above 25°C (STC)
- **Peukert correction**: High discharge rates reduce usable capacity
- **Sensor noise**: Gaussian ±3% RMSE — calibrated to ESP32 + ADS1115 ADC
- **Solar model**: Clear-sky GHI sine curve at 31.7°N, 950 W/m² peak
- **Tariff**: Indian C&I TOU — ₹8/kWh base (≈$0.096), 1.6× peak multiplier
