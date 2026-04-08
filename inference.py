"""
Inference Script — Micro-Grid Energy Arbitrator OpenEnv
========================================================
Team RauResh — IIT Mandi

Mandatory submission file. Runs an LLM agent against all 3 tasks
and reports reproducible baseline scores.

Environment variables required:
  API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier
  HF_TOKEN       API key

Usage:
  python inference.py
  python inference.py --task task_easy
  python inference.py --seed 42
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.environment import MicroGridEnv
from env.models import ActionType, GridAction, LoadTier
from graders.graders import get_grader

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

MAX_STEPS   = 24
TEMPERATURE = 0.1
MAX_TOKENS  = 200
SEED        = 42
TASK_IDS    = ["task_easy", "task_medium", "task_hard"]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an AI energy manager controlling a solar + battery micro-grid for 24 hours.
Each hour you observe the grid state and choose one action.

AVAILABLE ACTIONS — output exactly one per turn:
  buy_energy:<kw>          Buy kW from main grid (charges battery)
  sell_energy:<kw>         Sell kW surplus to main grid (discharges battery)
  store_energy:0           Route solar to battery, no grid trade
  idle:0                   Do nothing this hour
  buy_energy:<kw>:shed_deferrable    Buy AND shed deferrable loads

RULES:
  - Never let battery hit 0% — that is a BLACKOUT (episode ends, huge penalty)
  - Buy during cheap off-peak hours (midnight to 6am, midday)
  - Sell during expensive peak hours (7-10am, 6-9pm)
  - Use load-shedding only when battery is critically low (<15%)
  - Replace <kw> with a number between 0.0 and 20.0

Output ONLY the action string. No explanation.
Examples:
  buy_energy:5.0
  sell_energy:3.0
  store_energy:0
  idle:0
""").strip()


def build_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    battery  = obs.get("battery", {})
    pricing  = obs.get("pricing", {})
    load     = obs.get("load", {})
    forecast = obs.get("forecast", {})
    solar    = obs.get("current_solar_output_kw", 0)

    history_str = "\n".join(f"  {h}" for h in history[-4:]) if history else "  (first hour)"

    fc_prices = forecast.get("price_forecast", [])
    fc_irr    = forecast.get("irradiance_forecast_wm2", [])
    fc_wx     = forecast.get("weather_sequence", [])

    return textwrap.dedent(f"""
    HOUR {obs.get('hour', step)} / 23  |  Task: {obs.get('task_id')}

    BATTERY:
      SoC: {battery.get('state_of_charge_pct', 0):.1f}%
      Capacity: {battery.get('capacity_kwh', 50)} kWh
      Available: {round((battery.get('state_of_charge_pct',0)-10)/100 * battery.get('capacity_kwh',50), 2)} kWh

    SOLAR:
      Output now: {solar:.2f} kW
      Irradiance: {obs.get('irradiance_wm2', 0):.0f} W/m²
      Weather: {obs.get('weather', 'unknown')}

    PRICING:
      Buy price:  ${pricing.get('buy_price_per_kwh', 0):.4f}/kWh
      Sell price: ${pricing.get('sell_price_per_kwh', 0):.4f}/kWh
      Peak hour?: {pricing.get('is_peak_hour', False)}

    LOAD:
      Total demand: {load.get('total_demand_kw', 0):.2f} kW
      Critical: {load.get('critical_kw', 0):.2f} kW  |  Essential: {load.get('essential_kw', 0):.2f} kW  |  Deferrable: {load.get('deferrable_kw', 0):.2f} kW

    FORECAST (next {len(fc_prices)} hours):
      Prices:     {[round(p,4) for p in fc_prices]}
      Irradiance: {[round(i,1) for i in fc_irr]}
      Weather:    {fc_wx}

    EPISODE SO FAR:
      Cost: ${obs.get('total_cost_usd', 0):.3f}  |  Revenue: ${obs.get('total_revenue_usd', 0):.3f}
      Blackouts: {obs.get('blackout_count', 0)}  |  Reserve hours: {obs.get('hours_above_reserve', 0)}
      Cumulative reward: {obs.get('cumulative_reward', 0):.3f}

    RECENT ACTIONS:
    {history_str}

    What is your action for this hour?
    """).strip()


VALID_ACTIONS = {
    "buy_energy", "sell_energy", "store_energy", "idle"
}
FALLBACK = "idle:0"


def parse_action(text: str) -> GridAction:
    text = (text or "").strip().lower().splitlines()[0].strip()
    for prefix in ["action:", "output:", "next:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    parts = text.split(":")
    atype_str = parts[0].strip()
    if atype_str not in VALID_ACTIONS:
        return GridAction.from_string(FALLBACK)
    try:
        atype = ActionType(atype_str)
        qty   = float(parts[1]) if len(parts) > 1 else 0.0
        shed  = None
        if len(parts) > 2 and "shed_" in parts[2]:
            tier_str = parts[2].replace("shed_", "")
            try:
                shed = LoadTier(tier_str)
            except ValueError:
                pass
        return GridAction(action_type=atype, quantity_kw=max(0.0, min(qty, 20.0)), shed_tier=shed)
    except Exception:
        return GridAction.from_string(FALLBACK)


def run_episode(
    client: OpenAI,
    task_id: str,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    print(f"\n[START] task={task_id} seed={seed}")

    env = MicroGridEnv(task_id=task_id, max_steps=MAX_STEPS, seed=seed)
    obs = env.reset()
    obs_dict = obs.model_dump()

    history:      List[str] = []
    total_reward: float     = 0.0
    steps:        int       = 0

    if verbose:
        print(f"{'─'*55}")
        print(f"Episode: {obs.episode_id} | Task: {task_id}")
        print(f"Initial SoC: {obs.battery.state_of_charge_pct}%")
        print(f"{'─'*55}")

    for step in range(MAX_STEPS):
        user_prompt = build_prompt(obs_dict, step, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [API error hour {step}]: {exc}")
            response_text = FALLBACK

        action = parse_action(response_text)
        result = env.step(action)
        obs_dict = result.observation.model_dump()
        total_reward += result.reward
        steps += 1

        soc = result.observation.battery.state_of_charge_pct
        print(f"[STEP] hour={step:02d} action={action.to_string():25s} reward={result.reward:+.3f} soc={soc:.1f}%")
        history.append(f"Hour {step:02d}: {action.to_string()} → reward {result.reward:+.3f} soc={soc:.1f}%")

        if result.done:
            break

    state  = env.state()
    grader = get_grader(task_id)
    grade  = grader.grade(state)

    print(f"\n[END] task={task_id}")
    print(f"  Grade score    : {grade.score:.3f} ({'PASS' if grade.passed else 'FAIL'})")
    print(f"  Uptime score   : {grade.uptime_score:.3f}")
    print(f"  Economic score : {grade.economic_score:.3f}")
    print(f"  Reserve score  : {grade.reserve_score:.3f}")
    print(f"  Blackouts      : {state['blackout_count']}")
    print(f"  Net cost       : ${state['net_cost_usd']:.3f}")
    if grade.feedback != "Good grid management":
        print(f"  Feedback       : {grade.feedback}")

    return {
        "task_id":        task_id,
        "episode_id":     state["episode_id"],
        "steps_taken":    steps,
        "total_reward":   round(total_reward, 4),
        "grade_score":    grade.score,
        "uptime_score":   grade.uptime_score,
        "economic_score": grade.economic_score,
        "reserve_score":  grade.reserve_score,
        "blackout_count": state["blackout_count"],
        "net_cost_usd":   state["net_cost_usd"],
        "passed":         grade.passed,
        "feedback":       grade.feedback,
    }


def main(task_filter: Optional[str] = None, seed: int = SEED, verbose: bool = True) -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks  = [task_filter] if task_filter else TASK_IDS

    print(f"\n{'='*55}")
    print("Micro-Grid Energy Arbitrator — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"Seed  : {seed}")
    print(f"{'='*55}")

    all_results:    List[Dict[str, Any]] = []
    task_summaries: List[Dict[str, Any]] = []

    for task_id in tasks:
        # Run 3 episodes per task with different seeds for variance check
        task_results = []
        for i in range(3):
            result = run_episode(client, task_id, seed=seed + i, verbose=verbose)
            task_results.append(result)
            all_results.append(result)

        avg_score   = sum(r["grade_score"]  for r in task_results) / len(task_results)
        pass_rate   = sum(1 for r in task_results if r["passed"]) / len(task_results)
        avg_reward  = sum(r["total_reward"] for r in task_results) / len(task_results)
        avg_blackout= sum(r["blackout_count"] for r in task_results) / len(task_results)

        task_summaries.append({
            "task_id":      task_id,
            "n_episodes":   len(task_results),
            "avg_score":    round(avg_score, 4),
            "pass_rate":    round(pass_rate, 4),
            "avg_reward":   round(avg_reward, 4),
            "avg_blackouts":round(avg_blackout, 2),
        })

    print(f"\n{'='*65}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"{'Task':<15} {'Episodes':>9} {'Avg Score':>10} {'Pass Rate':>10} {'Blackouts':>10}")
    print(f"{'─'*65}")
    for s in task_summaries:
        print(
            f"{s['task_id']:<15} {s['n_episodes']:>9} "
            f"{s['avg_score']:>10.3f} {s['pass_rate']:>10.1%} {s['avg_blackouts']:>10.1f}"
        )
    overall = sum(s["avg_score"] for s in task_summaries) / len(task_summaries)
    print(f"{'─'*65}")
    print(f"{'OVERALL':<15} {'':>9} {overall:>10.3f}")
    print(f"{'='*65}\n")

    output = {
        "model":           MODEL_NAME,
        "seed":            seed,
        "task_summaries":  task_summaries,
        "episode_results": all_results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results written to baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default=None)
    parser.add_argument("--seed",  type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(task_filter=args.task, seed=args.seed, verbose=not args.quiet)
