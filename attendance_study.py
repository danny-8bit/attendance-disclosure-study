"""Exact and tabular learning experiments for the attendance inspection game."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import json
from math import comb, lcm, sqrt
from pathlib import Path
import platform
from time import perf_counter
from typing import NamedTuple

import numpy as np


LESSONS = 16
PHASE_LENGTH = 8
CHECKS = 10
FIRST_QUOTAS = (4, 5, 6)
TOTAL_ONLY, PHASE_QUOTAS = range(2)
DISCLOSURE_NAMES = ("total_only", "phase_quotas")
ATTEND, ABSENT = range(2)
LINEAR, WARNING = range(2)
PENALTY_NAMES = ("linear", "first_warning")
UNIT = 1000
DP_COSTS = tuple(range(50, 951, 25))
LEARNING_COSTS = (250, 450, 575, 625, 650, 750, 850)
TRAIN_SEEDS = tuple(range(20260, 20276))
TRAIN_SEMESTERS = 32768
VALIDATION_SEED, VALIDATION_SIZE = 37700, 512
TEST_SEED, TEST_SIZE = 38400, 4096
CHECKPOINTS = (0, 128, 512, 2048, 8192, 32768)
T_CRITICAL_95 = {1: 12.706204736, 15: 2.131449546}
TIE_TOLERANCE = 1e-12


class State(NamedTuple):
    lesson: int
    checked_first: int
    checked_second: int
    caught: int
    known_first_quota: int = 0


def check_patterns(quota: int) -> np.ndarray:
    patterns = np.zeros((comb(PHASE_LENGTH, quota), PHASE_LENGTH), dtype=np.uint8)
    for row, positions in enumerate(combinations(range(PHASE_LENGTH), quota)):
        patterns[row, list(positions)] = 1
    return patterns


@dataclass
class Schedules:
    checks: np.ndarray
    first_quotas: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray

    @classmethod
    def enumerate(cls) -> Schedules:
        groups = []
        for first_quota in FIRST_QUOTAS:
            first = check_patterns(first_quota)
            second = check_patterns(CHECKS - first_quota)
            groups.append(np.column_stack((np.repeat(first, len(second), axis=0),
                                           np.tile(second, (len(first), 1)))))
        sizes = np.array([len(group) for group in groups])
        common_denominator = lcm(*(int(size) for size in sizes))
        return cls(
            np.concatenate(groups),
            np.repeat(FIRST_QUOTAS, sizes),
            np.repeat(common_denominator // sizes, sizes),
            np.cumsum(sizes) - sizes,
            sizes,
        )

    def draw(self, rng: np.random.Generator, count: int) -> np.ndarray:
        groups = rng.integers(len(FIRST_QUOTAS), size=count)
        return self.offsets[groups] + rng.integers(self.sizes[groups])


def choose_or_zero(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


@lru_cache(maxsize=None)
def check_probability(lesson: int, checked_first: int, checked_second: int,
                      known_first_quota: int) -> Fraction:
    """Condition on the ordered prefix via its sufficient phase counts."""
    taken_first = min(lesson, PHASE_LENGTH)
    taken_second = lesson - taken_first
    remaining_first = PHASE_LENGTH - taken_first
    remaining_second = PHASE_LENGTH - taken_second
    candidates = (known_first_quota,) if known_first_quota else FIRST_QUOTAS
    total_likelihood = Fraction(0)
    weighted_probability = Fraction(0)
    for first_quota in candidates:
        second_quota = CHECKS - first_quota
        left_first = first_quota - checked_first
        left_second = second_quota - checked_second
        likelihood = (Fraction(choose_or_zero(remaining_first, left_first),
                               comb(PHASE_LENGTH, first_quota))
                      * Fraction(choose_or_zero(remaining_second, left_second),
                                 comb(PHASE_LENGTH, second_quota)))
        if likelihood:
            probability = (Fraction(left_first, remaining_first) if remaining_first
                           else Fraction(left_second, remaining_second))
            total_likelihood += likelihood
            weighted_probability += likelihood * probability
    return weighted_probability / total_likelihood


def advance(state: State, action: int, checked: int) -> State:
    checked_first = checked if state.lesson < PHASE_LENGTH else 0
    return State(state.lesson + 1, state.checked_first + checked_first,
                 state.checked_second + checked - checked_first,
                 state.caught + checked * (action == ABSENT), state.known_first_quota)


def penalty_for_absence(caught: int | np.ndarray, rule: int) -> int | np.ndarray:
    if rule == LINEAR:
        return 1
    return caught > 0


@dataclass
class Game:
    states: list[State]
    probabilities: list[Fraction]
    transitions: np.ndarray
    roots: np.ndarray
    previously_caught: np.ndarray

    @property
    def terminal(self) -> int:
        return len(self.states)

    @classmethod
    def build(cls) -> Game:
        roots = [State(0, 0, 0, 0, quota) for quota in (0, *FIRST_QUOTAS)]
        next_layer = set(roots)
        states, probabilities = [], []
        for _ in range(LESSONS):
            layer = sorted(next_layer)
            states.extend(layer)
            next_layer = set()
            for state in layer:
                probability = check_probability(state.lesson, state.checked_first,
                                                state.checked_second, state.known_first_quota)
                probabilities.append(probability)
                for checked, chance in enumerate((1 - probability, probability)):
                    if chance:
                        next_layer.update(advance(state, action, checked) for action in (ATTEND, ABSENT))
        ids = {state: number for number, state in enumerate(states)}
        transitions = np.full((len(states), 2, 2), len(states), dtype=np.intp)
        for number, (state, probability) in enumerate(zip(states, probabilities)):
            if state.lesson + 1 < LESSONS:
                for checked, chance in enumerate((1 - probability, probability)):
                    if chance:
                        for action in (ATTEND, ABSENT):
                            transitions[number, action, checked] = ids[advance(state, action, checked)]
        root_ids = np.full((2, max(FIRST_QUOTAS) + 1), -1, dtype=np.intp)
        for quota in FIRST_QUOTAS:
            root_ids[TOTAL_ONLY, quota] = ids[roots[0]]
            root_ids[PHASE_QUOTAS, quota] = ids[State(0, 0, 0, 0, quota)]
        return cls(states, probabilities, transitions, root_ids,
                   np.array([state.caught for state in states], dtype=np.uint8))


def solve(game: Game, cost: int, rule: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find an optimal policy and bound attendance across utility-maximizing policies."""
    utility_cost = Fraction(cost, UNIT)
    values = [Fraction(0)] * (game.terminal + 1)
    fewest = values.copy()
    most = values.copy()
    policy = np.zeros(game.terminal, dtype=np.uint8)
    for state in reversed(range(game.terminal)):
        p = game.probabilities[state]
        stay = 1 - p
        attend = (-utility_cost + stay * values[game.transitions[state, ATTEND, 0]]
                  + p * values[game.transitions[state, ATTEND, 1]])
        absent = (stay * values[game.transitions[state, ABSENT, 0]]
                  + p * (values[game.transitions[state, ABSENT, 1]]
                         - int(penalty_for_absence(game.states[state].caught, rule))))
        policy[state] = absent > attend
        values[state] = max(attend, absent)
        optimal_actions = ((ATTEND, ABSENT) if attend == absent else (int(policy[state]),))
        least_counts, most_counts = [], []
        for action in optimal_actions:
            if_unchecked, if_checked = game.transitions[state, action]
            immediate = int(action == ATTEND)
            least_counts.append(immediate + stay * fewest[if_unchecked] + p * fewest[if_checked])
            most_counts.append(immediate + stay * most[if_unchecked] + p * most[if_checked])
        fewest[state], most[state] = min(least_counts), max(most_counts)

    def start_mean(numbers: list[Fraction], mode: int) -> Fraction:
        return sum(numbers[game.roots[mode, quota]] for quota in FIRST_QUOTAS) / len(FIRST_QUOTAS)

    modes = (TOTAL_ONLY, PHASE_QUOTAS)
    start_values = np.array([start_mean(values, mode) for mode in modes], dtype=np.float64)
    attendance_bounds = np.array([[start_mean(edge, mode) / LESSONS for edge in (fewest, most)]
                                  for mode in modes], dtype=np.float64)
    return policy, start_values, attendance_bounds


@dataclass
class Rollout:
    attendance_mask: np.ndarray
    caught: np.ndarray
    by_lesson: np.ndarray
    attendance: np.ndarray
    utility: np.ndarray


def evaluate(game: Game, schedules: Schedules, schedule_ids: np.ndarray, policies: np.ndarray,
             modes: np.ndarray, costs: np.ndarray, rule: int,
             weights: np.ndarray | None = None) -> Rollout:
    """Apply fixed policies to schedules and reveal each check only after the action."""
    states = game.roots[modes[:, None], schedules.first_quotas[schedule_ids]]
    policy_ids = np.arange(len(policies))[:, None]
    attendance_mask = np.zeros(states.shape, dtype=np.uint16)
    attendance_count = np.zeros(states.shape, dtype=np.uint8)
    caught = np.zeros(states.shape, dtype=np.uint8)
    by_lesson = np.empty((len(policies), LESSONS))
    for lesson, checked in enumerate(schedules.checks[schedule_ids].T):
        actions = policies[policy_ids, states]
        attended = actions == ATTEND
        attendance_mask |= attended.astype(np.uint16) << lesson
        attendance_count += attended
        caught += (actions == ABSENT) & checked.astype(bool)
        by_lesson[:, lesson] = np.average(attended, weights=weights, axis=1)
        states = game.transitions[states, actions, checked]
    penalties = caught if rule == LINEAR else np.maximum(caught, 1) - 1
    utility = -costs[:, None] * attendance_count - penalties
    return Rollout(attendance_mask, caught, by_lesson,
                   np.average(attendance_count, weights=weights, axis=1) / LESSONS,
                   np.average(utility, weights=weights, axis=1))


def greedy(q: np.ndarray) -> np.ndarray:
    return (q[..., ABSENT] > q[..., ATTEND] + TIE_TOLERANCE).astype(np.uint8)


def exact_study(game: Game, schedules: Schedules) -> list[dict]:
    all_ids = np.arange(len(schedules.checks))
    modes = np.array([TOTAL_ONLY, PHASE_QUOTAS])
    rows = []
    for rule in (LINEAR, WARNING):
        for cost in DP_COSTS:
            policy, values, attendance_bounds = solve(game, cost, rule)
            roll = evaluate(game, schedules, all_ids, np.stack((policy, policy)), modes,
                            np.full(2, cost / UNIT), rule, schedules.weights)
            np.testing.assert_allclose(roll.utility, values, atol=1e-11, rtol=0)
            if values[PHASE_QUOTAS] + TIE_TOLERANCE < values[TOTAL_ONLY]:
                raise AssertionError("An optimal student can ignore the additional quota information")
            if (np.any(roll.attendance < attendance_bounds[:, 0] - TIE_TOLERANCE)
                    or np.any(roll.attendance > attendance_bounds[:, 1] + TIE_TOLERANCE)):
                raise AssertionError("The optimal policy lies outside optimal-attendance bounds")
            rows.append({"rule": PENALTY_NAMES[rule], "cost": cost / UNIT,
                         "attendance": roll.attendance.tolist(), "utility": values.tolist(),
                         "by_lesson": roll.by_lesson.tolist(),
                         "optimal_attendance_bounds": attendance_bounds.tolist()})
    return rows


def train(game: Game, schedules: Schedules, seeds: tuple[int, ...], episodes: int,
          validation_ids: np.ndarray, checkpoints: tuple[int, ...]) -> dict:
    """Vectorize over independent runs, preserving a separate Q table per run."""
    n_seeds = len(seeds)
    schedule_rows = np.empty((episodes, n_seeds), dtype=np.intp)
    explore_roll = np.empty((episodes, LESSONS, n_seeds), dtype=np.float32)
    explore_action = np.empty((episodes, LESSONS, n_seeds), dtype=np.uint8)
    for column, seed in enumerate(seeds):
        schedule_seed, explore_seed = np.random.SeedSequence(seed).spawn(2)
        schedule_rows[:, column] = schedules.draw(np.random.default_rng(schedule_seed), episodes)
        rng = np.random.default_rng(explore_seed)
        explore_roll[:, :, column] = rng.random((episodes, LESSONS), dtype=np.float32)
        explore_action[:, :, column] = rng.integers(2, size=(episodes, LESSONS), dtype=np.uint8)

    seed_indices = np.tile(np.arange(n_seeds), 2 * len(LEARNING_COSTS))
    modes = np.repeat(np.tile((TOTAL_ONLY, PHASE_QUOTAS), len(LEARNING_COSTS)), n_seeds)
    costs = np.repeat(np.array(LEARNING_COSTS) / UNIT, 2 * n_seeds)
    run = np.arange(len(costs))
    q = np.zeros((len(run), game.terminal + 1, 2))
    visits = np.zeros(q.shape, dtype=np.uint32)
    rates = np.arange(1, episodes + 1, dtype=np.float64) ** -0.6
    decay_steps = max(1, (4 * episodes + 4) // 5 - 1)
    curve = []

    def record(episode: int) -> None:
        validation = evaluate(game, schedules, validation_ids, greedy(q), modes, costs, WARNING)
        curve.append((episode, validation.attendance, validation.utility))
        print(f"Checkpoint {episode:>5}/{episodes}: validation mean utility "
              f"{validation.utility.mean():.4f}", flush=True)

    record(0)
    for episode in range(episodes):
        schedule_ids = schedule_rows[episode, seed_indices]
        checks = schedules.checks[schedule_ids].T
        states = game.roots[modes, schedules.first_quotas[schedule_ids]]
        epsilon = 1.0 - 0.95 * min(episode / decay_steps, 1.0)
        for lesson, checked in enumerate(checks):
            actions = np.where(explore_roll[episode, lesson, seed_indices] < epsilon,
                               explore_action[episode, lesson, seed_indices],
                               greedy(q[run, states]))
            successors = game.transitions[states, actions, checked]
            penalty = (actions == ABSENT) & (checked == 1) & (game.previously_caught[states] > 0)
            reward = np.where(actions == ATTEND, -costs, -penalty.astype(np.float64))
            keys = (run, states, actions)
            visits[keys] += 1
            target = reward + np.maximum(q[run, successors, ATTEND], q[run, successors, ABSENT])
            q[keys] += rates[visits[keys] - 1] * (target - q[keys])
            states = successors
        if episode + 1 in checkpoints:
            record(episode + 1)
    return {"q": q, "visits": visits, "modes": modes, "costs": costs,
            "seeds": np.array(seeds)[seed_indices], "curve": curve}


def mean_and_ci(samples: np.ndarray) -> dict:
    mean = float(samples.mean())
    half_width = T_CRITICAL_95[len(samples) - 1] * samples.std(ddof=1) / sqrt(len(samples))
    return {"mean": mean, "ci95": [mean - half_width, mean + half_width]}


def summarize_learning(test: Rollout, exact_test: Rollout,
                       exact_rows: list[dict], n_seeds: int) -> list[dict]:
    results = []
    by_cost = {row["cost"]: row for row in exact_rows if row["rule"] == PENALTY_NAMES[WARNING]}
    for index, cost in enumerate(LEARNING_COSTS):
        sample = slice(index * 2 * n_seeds, (index + 1) * 2 * n_seeds)
        attendance = test.attendance[sample].reshape(2, n_seeds)
        utility = test.utility[sample].reshape(2, n_seeds)
        by_lesson = test.by_lesson[sample].reshape(2, n_seeds, LESSONS)
        exact_utility = exact_test.utility[sample].reshape(2, n_seeds)
        dp_utility = np.array(by_cost[cost / UNIT]["utility"])
        regret = dp_utility[:, None] - exact_utility
        if np.min(regret) < -TIE_TOLERANCE:
            raise AssertionError("A policy exceeded the exact optimum under the same information")
        results.append({"cost": cost / UNIT, "attendance": attendance.mean(axis=1).tolist(),
                        "utility": utility.mean(axis=1).tolist(), "by_lesson": by_lesson.mean(axis=1).tolist(),
                        "attendance_difference_pp": mean_and_ci(100 * (attendance[1] - attendance[0])),
                        "utility_difference": mean_and_ci(utility[1] - utility[0]),
                        "exact_utility_regret": [mean_and_ci(group) for group in regret]})
    return results


def write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(output: Path, quick: bool) -> None:
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    schedules = Schedules.enumerate()
    game = Game.build()
    exact = exact_study(game, schedules)
    print(f"Exact benchmark verified for {len(schedules.checks):,} schedules "
          f"and {len(game.states):,} observable states", flush=True)
    seeds = TRAIN_SEEDS[:2] if quick else TRAIN_SEEDS
    episodes = 1024 if quick else TRAIN_SEMESTERS
    checkpoints = (0, 128, 512, 1024) if quick else CHECKPOINTS
    validation_ids = schedules.draw(np.random.default_rng(VALIDATION_SEED), 128 if quick else VALIDATION_SIZE)
    test_ids = schedules.draw(np.random.default_rng(TEST_SEED), 256 if quick else TEST_SIZE)
    training = train(game, schedules, seeds, episodes, validation_ids, checkpoints)
    training_policies = greedy(training["q"])
    modes = training["modes"]
    costs = training["costs"]
    test = evaluate(game, schedules, test_ids, training_policies, modes, costs, WARNING)
    exact_test = evaluate(game, schedules, np.arange(len(schedules.checks)), training_policies,
                          modes, costs, WARNING, schedules.weights)
    learning = summarize_learning(test, exact_test, exact, len(seeds))
    mode_names = [DISCLOSURE_NAMES[mode] for mode in modes]
    write_csv(output / "training_curve.csv", ("semester", "cost", "disclosure", "seed", "attendance", "utility"),
              ((epoch, cost, mode, seed, attendance, utility)
               for epoch, att, util in training["curve"]
               for cost, mode, seed, attendance, utility in zip(costs, mode_names, training["seeds"], att, util)))
    write_csv(output / "test_by_seed.csv", ("cost", "disclosure", "seed", "attendance", "utility", "exact_utility"),
              zip(costs, mode_names, training["seeds"], test.attendance, test.utility, exact_test.utility))
    np.savez_compressed(output / "raw_results.npz", state_table=np.array(game.states),
                        q_values=training["q"], visits=training["visits"], modes=modes, costs=costs,
                        seeds=training["seeds"], all_checks=schedules.checks, all_first_quotas=schedules.first_quotas,
                        all_schedule_weights=schedules.weights, test_schedule_ids=test_ids,
                        validation_schedule_ids=validation_ids, test_attendance_mask=test.attendance_mask,
                        test_caught=test.caught)
    parameters = {"lessons": LESSONS, "checks": CHECKS, "first_quotas": list(FIRST_QUOTAS),
                  "schedule_count": len(schedules.checks), "phase_group_sizes": schedules.sizes.tolist(),
                  "observable_states": len(game.states), "disclosure_order": list(DISCLOSURE_NAMES),
                  "train_seeds": list(seeds), "train_semesters": episodes, "checkpoints": list(checkpoints),
                  "test_seed": TEST_SEED, "test_size": len(test_ids),
                  "validation_seed": VALIDATION_SEED, "validation_size": len(validation_ids),
                  "alpha_exponent": -0.6, "gamma": 1.0, "epsilon_initial": 1.0, "epsilon_final": 0.05,
                  "epsilon_decay_fraction": 0.8, "python": platform.python_version(), "numpy": np.__version__,
                  "elapsed_seconds": round(perf_counter() - started, 2), "quick": quick}
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"parameters": parameters, "exact": exact, "learning": learning}, stream, indent=2)
        stream.write("\n")
    print(f"Wrote results to {output} in {parameters['elapsed_seconds']:.1f} seconds", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a two-seed smoke experiment")
    parser.add_argument("--output-dir", type=Path, help="Override the result directory")
    args = parser.parse_args()
    default = "quick_results" if args.quick else "results"
    run(args.output_dir or Path(__file__).parent / default, args.quick)


if __name__ == "__main__":
    main()
