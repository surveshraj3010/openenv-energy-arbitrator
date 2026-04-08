#!/usr/bin/env python3
"""
Pre-submission validation script — Micro-Grid Energy Arbitrator
Team RauResh — IIT Mandi

Run before submitting:
    python validate.py
    python validate.py --strict
"""

import argparse
import subprocess
import sys
import traceback
from typing import List, Tuple

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   return f"{GREEN}  ✓ PASS{RESET}  {msg}"
def fail(msg): return f"{RED}  ✗ FAIL{RESET}  {msg}"
def warn(msg): return f"{YELLOW}  ⚠ WARN{RESET}  {msg}"

Results = List[Tuple[str, str]]


def check_files(results: Results) -> None:
    import os
    required = [
        ("inference.py",       "Mandatory inference script"),
        ("openenv.yaml",       "OpenEnv spec"),
        ("Dockerfile",         "Container definition"),
        ("requirements.txt",   "Python dependencies"),
        ("README.md",          "Documentation"),
        ("app.py",             "FastAPI server"),
        ("env/models.py",      "Typed models"),
        ("env/physics.py",     "Physics simulator"),
        ("env/environment.py", "Core environment"),
        ("env/reward.py",      "Reward function"),
        ("graders/graders.py", "Task graders"),
        ("tests/test_env.py",  "Unit tests"),
    ]
    for path, desc in required:
        if os.path.exists(path):
            results.append(("pass", ok(f"file: {path} ({desc})")))
        else:
            results.append(("fail", fail(f"file: {path} MISSING ({desc})")))


def check_yaml(results: Results) -> None:
    import yaml
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        for field in ["name", "version", "description", "observation_space",
                      "action_space", "reward", "tasks"]:
            if field in spec:
                results.append(("pass", ok(f"yaml: field '{field}' present")))
            else:
                results.append(("fail", fail(f"yaml: missing field '{field}'")))

        tasks = spec.get("tasks", [])
        if len(tasks) >= 3:
            results.append(("pass", ok(f"yaml: {len(tasks)} tasks defined (≥3 required)")))
        else:
            results.append(("fail", fail(f"yaml: only {len(tasks)} task(s)")))

        difficulties = [t.get("difficulty") for t in tasks]
        if set(difficulties) >= {"easy", "medium", "hard"}:
            results.append(("pass", ok("yaml: easy / medium / hard difficulties present")))
        else:
            results.append(("fail", fail(f"yaml: missing difficulties — found {difficulties}")))

        for task in tasks:
            tid = task.get("id", "?")
            pt  = task.get("pass_threshold")
            if pt is not None and 0 < pt < 1:
                results.append(("pass", ok(f"yaml: task '{tid}' pass_threshold={pt}")))
            else:
                results.append(("fail", fail(f"yaml: task '{tid}' missing valid pass_threshold")))

    except Exception as e:
        results.append(("fail", fail(f"yaml: parse error — {e}")))


def check_imports(results: Results) -> None:
    mods = [
        ("env.environment", "MicroGridEnv"),
        ("env.models",      "GridAction, GridObservation, GridReward, EpisodeResult, ActionType"),
        ("env.physics",     "BlackoutError, InsufficientFundsError, apply_charge, apply_discharge"),
        ("env.reward",      "compute_reward"),
        ("graders.graders", "get_grader, GRADER_REGISTRY, EasyGrader, MediumGrader, HardGrader"),
    ]
    for mod, symbols in mods:
        try:
            m = __import__(mod, fromlist=symbols.split(", "))
            for sym in [s.strip() for s in symbols.split(",")]:
                getattr(m, sym)
            results.append(("pass", ok(f"import: {mod}")))
        except Exception as e:
            results.append(("fail", fail(f"import: {mod} — {e}")))


def check_interface(results: Results) -> None:
    from env.environment import MicroGridEnv
    from env.models import ActionType, EpisodeResult, GridAction, GridObservation

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            env = MicroGridEnv(task_id=task_id, seed=42)

            obs = env.reset()
            assert isinstance(obs, GridObservation)
            assert obs.step_number == 0
            assert obs.blackout_count == 0
            assert obs.total_cost_usd == 0.0
            results.append(("pass", ok(f"reset()  [{task_id}]: clean GridObservation")))

            s = env.state()
            assert isinstance(s, dict)
            for key in ["episode_id", "task_id", "hour", "done",
                        "battery_soc_pct", "blackout_count", "cumulative_reward"]:
                assert key in s, f"Missing key: {key}"
            results.append(("pass", ok(f"state()  [{task_id}]: dict with required keys")))

            action = GridAction(action_type=ActionType.BUY_ENERGY, quantity_kw=3.0)
            result = env.step(action)
            assert isinstance(result, EpisodeResult)
            assert isinstance(result.reward, float)
            assert isinstance(result.done, bool)
            results.append(("pass", ok(f"step()   [{task_id}]: EpisodeResult(reward=float, done=bool)")))

            # Step after done raises
            for _ in range(30):
                try:
                    r = env.step(action)
                    if r.done:
                        break
                except RuntimeError:
                    break
            try:
                env.step(action)
                results.append(("fail", fail(f"step()   [{task_id}]: should raise after done")))
            except RuntimeError:
                results.append(("pass", ok(f"step()   [{task_id}]: raises RuntimeError when done")))

        except Exception as e:
            results.append(("fail", fail(f"interface [{task_id}]: {e}\n{traceback.format_exc()}")))


def check_graders(results: Results) -> None:
    from env.environment import MicroGridEnv
    from env.models import ActionType, GridAction
    from graders.graders import get_grader, GRADER_REGISTRY

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        assert task_id in GRADER_REGISTRY
        results.append(("pass", ok(f"grader [{task_id}]: registered")))

    # Scores in [0, 1]
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        state = {
            "blackout_count": 0, "net_cost_usd": 2.0,
            "total_revenue_usd": 0.5, "hours_above_reserve": 20, "hour": 24,
        }
        score = get_grader(task_id).grade(state).score
        if 0.0 <= score <= 1.0:
            results.append(("pass", ok(f"grader [{task_id}]: score={score:.3f} ∈ [0,1]")))
        else:
            results.append(("fail", fail(f"grader [{task_id}]: score={score} out of [0,1]")))

    # Deterministic
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        state = {
            "blackout_count": 0, "net_cost_usd": 3.0,
            "total_revenue_usd": 0.0, "hours_above_reserve": 18, "hour": 24,
        }
        g = get_grader(task_id)
        s1 = g.grade(state).score
        s2 = g.grade(state).score
        if s1 == s2:
            results.append(("pass", ok(f"grader [{task_id}]: deterministic")))
        else:
            results.append(("fail", fail(f"grader [{task_id}]: non-deterministic {s1} ≠ {s2}")))

    # Varies across scenarios
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        g = get_grader(task_id)
        scenarios = [
            {"blackout_count": 0, "net_cost_usd": 0.5,  "total_revenue_usd": 1.0, "hours_above_reserve": 24, "hour": 24},
            {"blackout_count": 0, "net_cost_usd": 6.0,  "total_revenue_usd": 0.0, "hours_above_reserve": 10, "hour": 24},
            {"blackout_count": 1, "net_cost_usd": 10.0, "total_revenue_usd": 0.0, "hours_above_reserve": 5,  "hour": 12},
        ]
        scores = {round(g.grade(s).score, 3) for s in scenarios}
        if len(scores) > 1:
            results.append(("pass", ok(f"grader [{task_id}]: varied scores {sorted(scores)}")))
        else:
            results.append(("fail", fail(f"grader [{task_id}]: constant score {scores}")))

    # Difficulty ordering: easy random > hard random
    easy_state = {"blackout_count": 0, "net_cost_usd": 4.0, "total_revenue_usd": 0.0, "hours_above_reserve": 20, "hour": 24}
    hard_state = {"blackout_count": 1, "net_cost_usd": 8.0, "total_revenue_usd": 0.0, "hours_above_reserve": 5,  "hour": 12}
    easy_s = get_grader("task_easy").grade(easy_state).score
    hard_s = get_grader("task_hard").grade(hard_state).score
    if easy_s >= hard_s:
        results.append(("pass", ok(f"difficulty: easy={easy_s:.3f} ≥ hard={hard_s:.3f}")))
    else:
        results.append(("warn", warn(f"difficulty ordering: easy={easy_s:.3f} < hard={hard_s:.3f}")))


def check_reward(results: Results) -> None:
    from env.environment import MicroGridEnv
    from env.models import ActionType, GridAction

    env = MicroGridEnv(task_id="task_easy", seed=42)
    env.reset()

    rewards = []
    actions = [
        GridAction(action_type=ActionType.BUY_ENERGY,   quantity_kw=5.0),
        GridAction(action_type=ActionType.IDLE,          quantity_kw=0.0),
        GridAction(action_type=ActionType.SELL_ENERGY,   quantity_kw=2.0),
        GridAction(action_type=ActionType.STORE_ENERGY,  quantity_kw=0.0),
        GridAction(action_type=ActionType.BUY_ENERGY,   quantity_kw=10.0),
    ]
    for a in actions:
        try:
            r = env.step(a)
            rewards.append(r.reward)
            rb = r.reward_breakdown
            computed = (rb.economic_gain + rb.blackout_penalty + rb.reserve_bonus
                        + rb.shed_penalty + rb.efficiency_bonus + rb.step_cost)
            assert abs(computed - rb.total) < 0.001, "Breakdown doesn't sum to total"
        except Exception:
            break

    unique = len(set(round(r, 4) for r in rewards))
    if unique >= 3:
        results.append(("pass", ok(f"reward: {unique} distinct values — shaped signal confirmed")))
    else:
        results.append(("warn", warn(f"reward: only {unique} distinct values")))

    results.append(("pass", ok("reward: breakdown components sum to total")))


def check_inference_script(results: Results) -> None:
    try:
        with open("inference.py") as f:
            src = f.read()
        checks = [
            ("from openai import OpenAI" in src or "import openai" in src, "imports OpenAI client"),
            ("os.getenv" in src,          "reads env vars via os.getenv"),
            ("API_BASE_URL" in src,        "references API_BASE_URL"),
            ("MODEL_NAME" in src,          "references MODEL_NAME"),
            ("HF_TOKEN" in src,            "references HF_TOKEN"),
            ("def main" in src,            "has main() function"),
            ("if __name__" in src,         "has __main__ guard"),
            ("[START]" in src,             "logs [START] marker"),
            ("[STEP]" in src,              "logs [STEP] marker"),
            ("[END]" in src,               "logs [END] marker"),
            ("baseline_results" in src,    "writes results JSON"),
            ("task_easy" in src and "task_medium" in src and "task_hard" in src,
             "runs all 3 tasks"),
        ]
        for passed, msg in checks:
            results.append(("pass" if passed else "fail",
                             ok(f"inference.py: {msg}") if passed else fail(f"inference.py: {msg}")))
    except Exception as e:
        results.append(("fail", fail(f"inference.py: {e}")))


def check_dockerfile(results: Results) -> None:
    try:
        with open("Dockerfile") as f:
            df = f.read()
        checks = [
            ("FROM python"    in df, "FROM python base image"),
            ("COPY requirements" in df, "copies requirements.txt"),
            ("pip install"    in df, "runs pip install"),
            ("7860"           in df, "exposes HF Spaces port 7860"),
            ("uvicorn"        in df, "starts uvicorn server"),
            ("HEALTHCHECK"    in df, "has HEALTHCHECK"),
        ]
        for passed, msg in checks:
            results.append(("pass" if passed else "warn",
                             ok(f"Dockerfile: {msg}") if passed else warn(f"Dockerfile: {msg}")))
    except FileNotFoundError:
        results.append(("fail", fail("Dockerfile: not found")))


def check_tests(results: Results) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True, text=True,
    )
    summary = (r.stdout.strip().split("\n") or [""])[-1]
    if r.returncode == 0:
        results.append(("pass", ok(f"pytest: {summary}")))
    else:
        results.append(("fail", fail(f"pytest: {summary}\n{r.stdout[-600:]}")))


def main(strict: bool = False) -> None:
    sys.path.insert(0, ".")

    print(f"\n{BOLD}{'═'*65}")
    print("  Micro-Grid Energy Arbitrator — Pre-Submission Validation")
    print(f"  Team RauResh — IIT Mandi")
    print(f"{'═'*65}{RESET}\n")

    results: Results = []
    sections = [
        ("Required files",           check_files),
        ("openenv.yaml spec",        check_yaml),
        ("Python imports",           check_imports),
        ("Interface (reset/step/state)", check_interface),
        ("Graders",                  check_graders),
        ("Reward function",          check_reward),
        ("inference.py",             check_inference_script),
        ("Dockerfile",               check_dockerfile),
        ("Unit tests (pytest)",      check_tests),
    ]

    for section_name, fn in sections:
        print(f"{BOLD}── {section_name}{RESET}")
        before = len(results)
        try:
            fn(results)
        except Exception as e:
            results.append(("fail", fail(f"{section_name}: unexpected error — {e}")))
        for _, msg in results[before:]:
            print(msg)
        print()

    passes   = sum(1 for s, _ in results if s == "pass")
    warnings = sum(1 for s, _ in results if s == "warn")
    failures = sum(1 for s, _ in results if s == "fail")

    print(f"{BOLD}{'═'*65}")
    print(f"  SUMMARY:  {passes} passed  |  {warnings} warnings  |  {failures} failed  |  {len(results)} total")
    print(f"{'═'*65}{RESET}\n")

    if failures == 0 and (not strict or warnings == 0):
        print(f"{GREEN}{BOLD}  ✓ READY TO SUBMIT{RESET}\n")
        sys.exit(0)
    else:
        msg = f"fix {failures} failure(s)" if failures else f"fix {warnings} warning(s) (strict mode)"
        print(f"{RED}{BOLD}  ✗ NOT READY — {msg}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    main(strict=args.strict)
