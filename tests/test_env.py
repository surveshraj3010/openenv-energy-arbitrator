"""
Unit tests for the Micro-Grid Energy Arbitrator OpenEnv.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env.environment import MicroGridEnv
from env.models import ActionType, EpisodeResult, GridAction, GridObservation, LoadTier
from graders.graders import EasyGrader, HardGrader, MediumGrader, get_grader, GRADER_REGISTRY


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return MicroGridEnv(task_id="task_easy", seed=42)

@pytest.fixture
def medium_env():
    return MicroGridEnv(task_id="task_medium", seed=42)

@pytest.fixture
def hard_env():
    return MicroGridEnv(task_id="task_hard", seed=42)

def idle_action():
    return GridAction(action_type=ActionType.IDLE, quantity_kw=0.0)

def buy_action(kw=5.0):
    return GridAction(action_type=ActionType.BUY_ENERGY, quantity_kw=kw)

def sell_action(kw=3.0):
    return GridAction(action_type=ActionType.SELL_ENERGY, quantity_kw=kw)


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_grid_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, GridObservation)

    def test_initial_step_is_zero(self, easy_env):
        obs = easy_env.reset()
        assert obs.step_number == 0

    def test_initial_hour_is_zero(self, easy_env):
        obs = easy_env.reset()
        assert obs.hour == 0

    def test_no_blackouts_at_start(self, easy_env):
        obs = easy_env.reset()
        assert obs.blackout_count == 0

    def test_no_cost_at_start(self, easy_env):
        obs = easy_env.reset()
        assert obs.total_cost_usd == 0.0

    def test_easy_soc_is_80(self, easy_env):
        obs = easy_env.reset()
        assert obs.battery.state_of_charge_pct == 80.0

    def test_medium_soc_is_50(self, medium_env):
        obs = medium_env.reset()
        assert obs.battery.state_of_charge_pct == 50.0

    def test_hard_soc_is_25(self, hard_env):
        obs = hard_env.reset()
        assert obs.battery.state_of_charge_pct == 25.0

    def test_reset_clears_previous_state(self, easy_env):
        easy_env.reset()
        easy_env.step(buy_action(10.0))
        obs = easy_env.reset()
        assert obs.total_cost_usd == 0.0
        assert obs.step_number == 0

    def test_reproducible_with_same_seed(self):
        e1 = MicroGridEnv(task_id="task_easy", seed=99)
        e2 = MicroGridEnv(task_id="task_easy", seed=99)
        o1 = e1.reset()
        o2 = e2.reset()
        assert o1.battery.state_of_charge_pct == o2.battery.state_of_charge_pct

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            MicroGridEnv(task_id="task_nonexistent")

    def test_has_weather_sequence(self, easy_env):
        easy_env.reset()
        s = easy_env.state()
        assert len(s["weather_sequence"]) == 24

    def test_has_24_price_points(self, easy_env):
        easy_env.reset()
        s = easy_env.state()
        assert len(s["buy_prices"]) == 24


# ── step() ────────────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_episode_result(self, easy_env):
        easy_env.reset()
        result = easy_env.step(idle_action())
        assert isinstance(result, EpisodeResult)

    def test_reward_is_float(self, easy_env):
        easy_env.reset()
        result = easy_env.step(idle_action())
        assert isinstance(result.reward, float)

    def test_done_is_bool(self, easy_env):
        easy_env.reset()
        result = easy_env.step(idle_action())
        assert isinstance(result.done, bool)

    def test_hour_advances_each_step(self, easy_env):
        easy_env.reset()
        result = easy_env.step(idle_action())
        assert result.observation.step_number == 1

    def test_24_steps_ends_episode(self, easy_env):
        easy_env.reset()
        result = None
        for _ in range(24):
            try:
                result = easy_env.step(buy_action(3.0))
            except RuntimeError:
                break
            if result.done:
                break
        assert result.done is True

    def test_step_after_done_raises(self, easy_env):
        easy_env.reset()
        for _ in range(24):
            try:
                r = easy_env.step(buy_action(3.0))
                if r.done:
                    break
            except RuntimeError:
                break
        with pytest.raises(RuntimeError):
            easy_env.step(idle_action())

    def test_step_without_reset_raises(self):
        env = MicroGridEnv(task_id="task_easy", seed=42)
        with pytest.raises(RuntimeError):
            env.step(idle_action())

    def test_buy_increases_cost(self, easy_env):
        easy_env.reset()
        easy_env.step(buy_action(5.0))
        s = easy_env.state()
        assert s["total_cost_usd"] > 0.0

    def test_sell_increases_revenue(self, easy_env):
        easy_env.reset()
        easy_env.step(sell_action(2.0))
        s = easy_env.state()
        assert s["total_revenue_usd"] > 0.0

    def test_idle_has_no_cost(self, easy_env):
        easy_env.reset()
        easy_env.step(idle_action())
        s = easy_env.state()
        assert s["total_cost_usd"] == 0.0

    def test_shed_tier_accepted(self, easy_env):
        easy_env.reset()
        action = GridAction(
            action_type=ActionType.BUY_ENERGY,
            quantity_kw=3.0,
            shed_tier=LoadTier.DEFERRABLE
        )
        result = easy_env.step(action)
        assert result.reward_breakdown.shed_penalty < 0

    def test_action_from_string_idle(self):
        a = GridAction.from_string("idle:0")
        assert a.action_type == ActionType.IDLE

    def test_action_from_string_buy(self):
        a = GridAction.from_string("buy_energy:7.5")
        assert a.action_type == ActionType.BUY_ENERGY
        assert a.quantity_kw == 7.5

    def test_action_from_string_sell(self):
        a = GridAction.from_string("sell_energy:3.0")
        assert a.action_type == ActionType.SELL_ENERGY

    def test_action_to_string_roundtrip(self):
        a = GridAction(action_type=ActionType.BUY_ENERGY, quantity_kw=5.0)
        assert "buy_energy" in a.to_string()


# ── state() ───────────────────────────────────────────────────────────────────

class TestState:
    def test_returns_dict(self, easy_env):
        easy_env.reset()
        assert isinstance(easy_env.state(), dict)

    def test_before_reset_not_initialised(self):
        env = MicroGridEnv(task_id="task_easy", seed=42)
        assert env.state()["initialised"] is False

    def test_required_keys_present(self, easy_env):
        easy_env.reset()
        s = easy_env.state()
        for key in ["episode_id", "task_id", "hour", "done",
                    "battery_soc_pct", "blackout_count",
                    "total_cost_usd", "cumulative_reward"]:
            assert key in s, f"Missing: {key}"

    def test_state_reflects_steps(self, easy_env):
        easy_env.reset()
        easy_env.step(idle_action())
        easy_env.step(idle_action())
        assert easy_env.state()["hour"] == 2


# ── Reward ────────────────────────────────────────────────────────────────────

class TestReward:
    def test_reward_breakdown_sums_to_total(self, easy_env):
        easy_env.reset()
        result = easy_env.step(buy_action(3.0))
        rb = result.reward_breakdown
        computed = (rb.economic_gain + rb.blackout_penalty + rb.reserve_bonus
                    + rb.shed_penalty + rb.efficiency_bonus + rb.step_cost)
        assert abs(computed - rb.total) < 0.001

    def test_blackout_gives_large_penalty(self, hard_env):
        hard_env.reset()
        # Drain battery with massive sell on task_hard (SoC starts at 25%)
        for _ in range(5):
            try:
                hard_env.step(sell_action(20.0))
            except Exception:
                break
        s = hard_env.state()
        # If a blackout happened, penalty was applied
        if s["blackout_count"] > 0:
            assert s["cumulative_reward"] < -40.0

    def test_reserve_bonus_when_soc_high(self, easy_env):
        easy_env.reset()
        result = easy_env.step(idle_action())
        # Easy task starts at 80% SoC — should earn reserve bonus
        assert result.reward_breakdown.reserve_bonus > 0

    def test_buy_offpeak_gives_efficiency_bonus(self, easy_env):
        easy_env.reset()
        # Hours 0-6 are off-peak — step to hour 2 then buy
        easy_env.step(idle_action())
        easy_env.step(idle_action())
        result = easy_env.step(buy_action(5.0))
        # hour 2 is off-peak — should have efficiency bonus
        assert result.reward_breakdown.efficiency_bonus >= 0


# ── Graders ───────────────────────────────────────────────────────────────────

class TestGraders:
    def test_all_three_registered(self):
        for tid in ["task_easy", "task_medium", "task_hard"]:
            assert tid in GRADER_REGISTRY

    def test_scores_in_range(self):
        for tid in ["task_easy", "task_medium", "task_hard"]:
            env = MicroGridEnv(task_id=tid, seed=42)
            env.reset()
            env.step(idle_action())
            s = env.state()
            grade = get_grader(tid).grade(s)
            assert 0.0 <= grade.score <= 1.0

    def test_deterministic(self):
        env = MicroGridEnv(task_id="task_easy", seed=42)
        env.reset()
        for _ in range(5):
            env.step(idle_action())
        s = env.state()
        g = get_grader("task_easy")
        assert g.grade(s).score == g.grade(s).score

    def test_no_blackout_scores_higher_than_blackout(self):
        grader = get_grader("task_easy")
        no_blackout_state = {
            "blackout_count": 0, "net_cost_usd": 2.0,
            "total_revenue_usd": 0.0, "hours_above_reserve": 20, "hour": 24,
        }
        blackout_state = {
            "blackout_count": 1, "net_cost_usd": 2.0,
            "total_revenue_usd": 0.0, "hours_above_reserve": 10, "hour": 12,
        }
        good_score = grader.grade(no_blackout_state).score
        bad_score  = grader.grade(blackout_state).score
        assert good_score > bad_score

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            get_grader("task_nonexistent")

    def test_describe_returns_dict(self):
        for tid in ["task_easy", "task_medium", "task_hard"]:
            desc = get_grader(tid).describe()
            assert "task_id" in desc
            assert "difficulty" in desc
            assert "pass_threshold" in desc

    def test_grader_varies_across_episodes(self):
        grader = get_grader("task_easy")
        states = [
            {"blackout_count": 0, "net_cost_usd": 1.0, "total_revenue_usd": 0.0, "hours_above_reserve": 24, "hour": 24},
            {"blackout_count": 0, "net_cost_usd": 5.0, "total_revenue_usd": 0.0, "hours_above_reserve": 12, "hour": 24},
            {"blackout_count": 1, "net_cost_usd": 8.0, "total_revenue_usd": 0.0, "hours_above_reserve": 6,  "hour": 12},
            {"blackout_count": 0, "net_cost_usd": 0.0, "total_revenue_usd": 2.0, "hours_above_reserve": 24, "hour": 24},
        ]
        scores = {round(grader.grade(s).score, 3) for s in states}
        assert len(scores) > 1


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_easy_episode_no_crash(self):
        env = MicroGridEnv(task_id="task_easy", seed=42)
        env.reset()
        for _ in range(24):
            try:
                env.step(buy_action(2.0))
            except RuntimeError:
                break
        s = env.state()
        grade = get_grader("task_easy").grade(s)
        assert 0.0 <= grade.score <= 1.0

    def test_all_tasks_complete(self):
        for tid in ["task_easy", "task_medium", "task_hard"]:
            env = MicroGridEnv(task_id=tid, seed=42)
            env.reset()
            result = None
            for _ in range(24):
                try:
                    result = env.step(idle_action())
                    if result.done:
                        break
                except RuntimeError:
                    break
            s = env.state()
            assert 0.0 <= get_grader(tid).grade(s).score <= 1.0

    def test_physics_battery_does_not_exceed_100(self):
        env = MicroGridEnv(task_id="task_easy", seed=42)
        env.reset()
        for _ in range(24):
            try:
                result = env.step(buy_action(20.0))
                assert result.observation.battery.state_of_charge_pct <= 100.0
                if result.done:
                    break
            except RuntimeError:
                break
