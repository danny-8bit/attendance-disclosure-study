from collections import defaultdict
from fractions import Fraction
import unittest

import numpy as np

from attendance_study import (
    ABSENT, ATTEND, CHECKS, FIRST_QUOTAS, Game, LESSONS, LINEAR, PHASE_QUOTAS,
    Schedules, State, TOTAL_ONLY, WARNING, advance, check_probability, evaluate,
    exact_study, penalty_for_absence, solve,
)


class AttendanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedules = Schedules.enumerate()
        cls.game = Game.build()

    def test_schedule_support_and_prior(self):
        schedules = self.schedules
        self.assertEqual(schedules.checks.shape, (7056, LESSONS))
        self.assertEqual(len(np.unique(schedules.checks, axis=0)), 7056)
        np.testing.assert_array_equal(schedules.sizes, (1960, 3136, 1960))
        np.testing.assert_array_equal(schedules.checks.sum(axis=1), CHECKS)
        np.testing.assert_array_equal(schedules.checks[:, :8].sum(axis=1), schedules.first_quotas)
        self.assertEqual(len({schedules.weights[schedules.first_quotas == q].sum()
                              for q in FIRST_QUOTAS}), 1)

    def test_visible_roots_and_phase_transition(self):
        states = self.game.states
        np.testing.assert_array_equal(self.game.roots[TOTAL_ONLY, 4:], (0, 0, 0))
        self.assertEqual([states[i].known_first_quota for i in self.game.roots[PHASE_QUOTAS, 4:]],
                         [4, 5, 6])
        self.assertEqual(check_probability(0, 0, 0, 0), Fraction(5, 8))
        for q in FIRST_QUOTAS:
            self.assertEqual(check_probability(0, 0, 0, q), Fraction(q, 8))
            self.assertEqual(check_probability(8, q, 0, 0), Fraction(10 - q, 8))

    def test_bayes_transition_against_weighted_schedules(self):
        schedules = self.schedules
        totals = defaultdict(lambda: [0, 0])
        for checks, quota, weight in zip(schedules.checks, schedules.first_quotas, schedules.weights):
            first = second = 0
            for lesson, checked in enumerate(checks.tolist()):
                for visible in (0, int(quota)):
                    counts = totals[lesson, first, second, visible]
                    counts[0] += int(weight)
                    counts[1] += checked * int(weight)
                if lesson < 8:
                    first += checked
                else:
                    second += checked
        for args, (mass, checked_mass) in totals.items():
            self.assertEqual(check_probability(*args), Fraction(checked_mass, mass))

    def test_only_a_checked_absence_increments_caught_and_penalty(self):
        start = State(7, 3, 0, 0, 4)
        self.assertEqual(advance(start, ATTEND, 1), State(8, 4, 0, 0, 4))
        self.assertEqual(advance(start, ABSENT, 0), State(8, 3, 0, 0, 4))
        self.assertEqual(advance(start, ABSENT, 1), State(8, 4, 0, 1, 4))
        self.assertEqual(advance(State(8, 4, 0, 1, 4), ABSENT, 1), State(9, 4, 1, 2, 4))
        self.assertEqual(penalty_for_absence(0, WARNING), 0)
        self.assertEqual(penalty_for_absence(1, WARNING), 1)
        self.assertEqual(penalty_for_absence(0, LINEAR), 1)

    def test_paired_rollout_and_semester_utility(self):
        masks = np.stack((np.full(self.game.terminal, ABSENT), np.full(self.game.terminal, ATTEND)))
        schedules = self.schedules.draw(np.random.default_rng(4), 32)
        roll = evaluate(self.game, self.schedules, schedules, masks, np.array([0, 1]),
                        np.array([0.625, 0.625]), WARNING)
        np.testing.assert_array_equal(roll.attendance, (0, 1))
        np.testing.assert_array_equal(roll.caught[0], 10)
        np.testing.assert_array_equal(roll.caught[1], 0)
        np.testing.assert_array_equal(roll.attendance_mask[1], 65535)
        np.testing.assert_array_equal(roll.utility, (-9, -10))

    def test_linear_policy_and_two_reference_results(self):
        for cost in (575, 650):
            policy, _, _ = solve(self.game, cost, LINEAR)
            expected = [p < Fraction(cost, 1000) for p in self.game.probabilities]
            np.testing.assert_array_equal(policy, expected)
        rows = {row['cost']: row for row in exact_study(self.game, self.schedules)
                if row['rule'] == 'linear'}
        np.testing.assert_allclose(rows[0.575]['attendance'], [0.666294642857, 0.590476190476], atol=1e-11)
        np.testing.assert_allclose(rows[0.65]['attendance'], [0.430803571429, 0.49375], atol=1e-11)


if __name__ == '__main__':
    unittest.main()
