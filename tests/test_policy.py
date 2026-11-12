import unittest

from ai_self_evaluation.config import PRESETS, load_yaml
from ai_self_evaluation.errors import EvaluationError
from ai_self_evaluation.policy import advance
from ai_self_evaluation.statistics import box_distance, describe


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_yaml(PRESETS / "advanced.yaml")["early_stopping"]
        self.settings.update(min_epochs=1, plateau_patience_epochs=3, post_target_epochs=2,
                             target_confirmations=2, min_delta=.05)

    def test_accumulated_small_improvements_and_absolute_best(self):
        state = None
        for epoch, value in enumerate([.4, .37, .34, .36, .32, .33], 1):
            state = advance(state, epoch, value, self.settings)
        self.assertTrue(state["should_stop"])
        self.assertEqual(state["reason"], "plateau")
        self.assertEqual(state["best_epoch"], 5)
        self.assertAlmostEqual(state["best_distance"], .32)
        self.assertAlmostEqual(state["progress_reference"], .34)

    def test_target_requires_window_and_confirmation(self):
        state = advance(None, 1, .02, self.settings)
        self.assertFalse(state["should_stop"])
        state = advance(state, 2, .019, self.settings)
        self.assertFalse(state["should_stop"])
        state = advance(state, 3, .018, self.settings)
        self.assertEqual(state["reason"], "target_reached_and_confirmed")

    def test_target_resets_after_deterioration(self):
        self.settings["plateau_patience_epochs"] = 20
        state = advance(None, 1, .02, self.settings)
        state = advance(state, 2, .1, self.settings)
        self.assertIsNone(state["target_start_epoch"])
        state = advance(state, 3, .02, self.settings)
        self.assertFalse(state["should_stop"])

    def test_target_window_waits_even_with_short_plateau_patience(self):
        self.settings.update(plateau_patience_epochs=1, post_target_epochs=3)
        state = advance(None, 1, .02, self.settings)
        state = advance(state, 2, .02, self.settings)
        self.assertFalse(state["should_stop"])
        state = advance(state, 3, .02, self.settings)
        self.assertFalse(state["should_stop"])
        state = advance(state, 4, .02, self.settings)
        self.assertEqual(state["reason"], "target_reached_and_confirmed")

    def test_missing_epochs_do_not_consume_patience(self):
        state = advance(None, 1, .4, self.settings)
        state = advance(state, 20, .4, self.settings)
        self.assertEqual(state["stalled_epochs"], 1)
        self.assertFalse(state["should_stop"])

    def test_equal_best_keeps_first_checkpoint(self):
        state = advance(None, 1, .4, self.settings)
        state = advance(state, 2, .4, self.settings)
        self.assertEqual(state["best_epoch"], 1)

    def test_monotonic_epochs(self):
        state = advance(None, 3, .4, self.settings)
        with self.assertRaises(EvaluationError):
            advance(state, 2, .3, self.settings)

    def test_distance_and_whiskers(self):
        real = describe([1, 2, 2, 3, 4])
        self.assertEqual(box_distance(real, real), 0)
        self.assertAlmostEqual(box_distance(describe([0, 0]), describe([4, 4])), 1)
        self.assertEqual(real["n"], 5)


if __name__ == "__main__":
    unittest.main()
