import json
import math
import unittest

from scripts.lunar_v7_angle_grid_extension import (
    ALL_ANGLES,
    BASE_GRID,
    NEW_ANGLES,
    build_grid,
    contrast,
    split_scenario_id,
)


class AngleGridExtensionTests(unittest.TestCase):
    def test_incremental_grid_adds_exactly_the_six_missing_angles(self):
        base = json.loads(BASE_GRID.read_text())
        grid = build_grid(base)
        self.assertEqual(NEW_ANGLES, (-3, -2, -1, 1, 2, 3))
        self.assertEqual(ALL_ANGLES, tuple(range(-4, 5)))
        self.assertEqual(len(grid), 108)
        self.assertEqual(len({row["scenario_id"] for row in grid}), 108)
        for index, original in enumerate(base):
            variants = grid[6 * index : 6 * index + 6]
            self.assertEqual(
                [round(math.degrees(row["angle"])) for row in variants],
                list(NEW_ANGLES),
            )
            for row in variants:
                base_id, degrees = split_scenario_id(row["scenario_id"])
                self.assertEqual(base_id, original["scenario_id"])
                self.assertIn(degrees, NEW_ANGLES)
                for key in original.keys() - {"scenario_id", "angle"}:
                    self.assertEqual(row[key], original[key])

    def test_combined_contrast_clusters_repeated_angles_within_seed(self):
        rows = []
        for seed in range(2001, 2021):
            difference = (seed - 2000) / 100
            for angle in (-1, 1):
                for condition, onset in (("DESC", 0.1), ("ECON", 0.1 + difference)):
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "base_scene": "scene",
                            "angle_degrees": angle,
                            "intervention": "original",
                            "primary_observed": True,
                            "primary_onset_time_s": onset,
                        }
                    )
        result = contrast(rows, "original", (-1, 1))
        self.assertAlmostEqual(result["estimate_s"], 0.105)
        self.assertEqual(result["joint_events_per_seed"], [2] * 20)
        self.assertEqual(result["positive_negative_zero_seeds"], [20, 0, 0])


if __name__ == "__main__":
    unittest.main()
