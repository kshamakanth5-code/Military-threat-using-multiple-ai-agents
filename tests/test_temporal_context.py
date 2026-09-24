import unittest

from backend.temporal_context import build_temporal_context


class TemporalContextTests(unittest.TestCase):
    def test_multiple_people_with_rapid_motion_flag_possible_altercation(self):
        report = build_temporal_context({
            'motionScore': 0.58,
            'movingPersons': 2,
            'personCount': 2,
            'threatLevel': 'HIGH',
            'detections': [],
        })
        self.assertTrue(report['interpersonalAggression'])
        self.assertEqual(report['movementPattern'], 'possible physical altercation')
        self.assertEqual(report['temporalAnomaly'], 'elevated')

    def test_high_motion_is_flagged_as_sudden_movement(self):
        report = build_temporal_context({
            'motionScore': 0.82,
            'poseActivities': ['crawling', 'running'],
            'threatLevel': 'HIGH',
            'movingPersons': 2,
            'movingObjects': 1,
            'detections': [{'confidence': 91}],
        })
        self.assertEqual(report['movementPattern'], 'sudden movement')
        self.assertIn('eventTimeline', report)
        self.assertGreater(len(report['eventTimeline']), 0)

    def test_low_motion_remains_stationary(self):
        report = build_temporal_context({
            'motionScore': 0.08,
            'poseActivities': ['standing'],
            'threatLevel': 'LOW',
            'movingPersons': 0,
            'movingObjects': 0,
            'detections': [],
        })
        self.assertEqual(report['movementPattern'], 'routine movement')
        self.assertIn('riskEvolution', report)


if __name__ == '__main__':
    unittest.main()
