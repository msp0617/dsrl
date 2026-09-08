"""Small regression tests; no simulator, torch, or GPU required."""
import unittest

import numpy as np
import pandas as pd

from analyze_completed_runs import (actual_eval, area, clean_history,
                                    deficit_area, interpolate, sem)


class CompletedAnalysisTests(unittest.TestCase):
    def test_append_order_rollback_removes_stale_future(self):
        raw = pd.DataFrame({"env_steps": [0, 10, 20, 30, 15, 25],
                            "success_rate": [.1, .2, .3, .4, .5, .6]})
        self.assertEqual(clean_history(raw).env_steps.tolist(), [0, 10, 15, 25])

    def test_restart_from_zero_is_not_a_second_seed(self):
        raw = pd.DataFrame({"env_steps": [0, 10, 0, 10, 20],
                            "success_rate": [.9, .8, .1, .2, .3]})
        self.assertEqual(clean_history(raw).success_rate.tolist(), [.1, .2, .3])

    def test_numeric_interpolation_and_no_extrapolation(self):
        vals = interpolate([0, 10, 100], [0, .5, 1], [-1, 5, 55, 101])
        np.testing.assert_allclose(vals[1:3], [.25, .75])
        self.assertTrue(np.isnan(vals[[0, 3]]).all())

    def test_legacy_hold_and_linear_boundary_are_distinct(self):
        self.assertAlmostEqual(area([0, 5, 10], [0, 1, 0], 8), 5.5/8)
        self.assertAlmostEqual(area([0, 5, 10], [0, 1, 0], 8, "linear"), 4.6/8)
        self.assertTrue(np.isnan(area([1, 5], [0, 1], 8)))

    def test_endpoint_is_observed_task_specific_and_bounded(self):
        ev = pd.DataFrame({"env_steps": [102128, 127136], "success_rate": [.4, .5]})
        self.assertEqual(actual_eval(ev, 127136), (.5, 127136))
        self.assertTrue(np.isnan(actual_eval(ev, 129152)[0]))
        self.assertEqual(actual_eval(ev, 127152), (.5, 127136))

    def test_single_seed_is_not_zero_uncertainty(self):
        self.assertTrue(np.isnan(sem([.5])))
        self.assertAlmostEqual(sem([.4, .6]), .1)

    def test_regret_clips_at_exact_crossing(self):
        self.assertAlmostEqual(deficit_area([0, 10], [0, 1], 10, .5), .125)


if __name__ == "__main__":
    unittest.main()
