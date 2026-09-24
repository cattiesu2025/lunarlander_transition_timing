import json
import math
import unittest

from scripts.lunar_tilt_extension import BASE_GRID, build_grid, posture_summary


class TiltExtensionTests(unittest.TestCase):
    def test_matched_grid_changes_only_angle_and_identifier(self):
        base = json.loads(BASE_GRID.read_text())
        grid = build_grid(base)
        self.assertEqual(len({r['scenario_id'] for r in grid}), 54)
        for i, original in enumerate(base):
            variants = grid[3 * i:3 * i + 3]
            self.assertEqual([r['angle'] for r in variants],
                             [math.radians(-4), 0, math.radians(4)])
            for row in variants:
                self.assertLessEqual(abs(row['angle']), .08)
                for key in original.keys() - {'scenario_id', 'angle'}:
                    self.assertEqual(row[key], original[key])

    def test_postcontact_bounces_excluded_from_diagnostics(self):
        rows = [dict(action=1, angle=.05, x_before=10, x=10.1),
                dict(action=2, angle=.02, x_before=10.1, x=10.2),
                dict(action=3, angle=1, x=15, contact=True),
                dict(action=3, angle=2, x=20)]
        result = posture_summary(rows)
        self.assertEqual(result['precontact_steps'], 2)
        self.assertEqual(result['precontact_side_actions'], 1)
        self.assertAlmostEqual(result['precontact_x_range'], .2)
        self.assertAlmostEqual(result['precontact_final_angle_deg'], math.degrees(.02))


if __name__ == '__main__':
    unittest.main()
