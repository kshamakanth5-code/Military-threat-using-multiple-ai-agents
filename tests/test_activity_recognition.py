import unittest

from backend.server import build_risk_regions, box_iou, classify_risk_band, deduplicate_person_detections
from backend.server import classify_activity


def pose_points(wrists=((5, 5), (15, 6)), ankles=((45, 48), (55, 48))):
    values = {
        5: (45, 20), 6: (55, 20),
        11: (45, 40), 12: (55, 40),
        13: (45, 45), 14: (55, 45),
        15: ankles[0], 16: ankles[1],
        9: wrists[0], 10: wrists[1],
    }
    return [{'index': index, 'x': point[0], 'y': point[1], 'confidence': 90} for index, point in values.items()]


class ActivityRecognitionTests(unittest.TestCase):
    def test_overlapping_person_boxes_are_counted_once(self):
        detections = [
            {'box': [10, 10, 30, 60], 'confidence': 91},
            {'box': [11, 11, 30, 60], 'confidence': 72},
            {'box': [70, 10, 20, 50], 'confidence': 80},
        ]
        self.assertGreater(box_iou(detections[0]['box'], detections[1]['box']), 0.6)
        self.assertEqual(len(deduplicate_person_detections(detections)), 2)

    def test_risk_bands_follow_heatmap_and_email_thresholds(self):
        self.assertEqual(classify_risk_band(20), 'NORMAL')
        self.assertEqual(classify_risk_band(40), 'LOW_RISK')
        self.assertEqual(classify_risk_band(60), 'HIGH_RISK')
        self.assertEqual(classify_risk_band(74), 'HIGH_RISK')
        self.assertEqual(classify_risk_band(75), 'SERIOUS_RISK')
        self.assertEqual(classify_risk_band(89), 'SERIOUS_RISK')
        self.assertEqual(classify_risk_band(90), 'VERY_HIGH_RISK')
        self.assertEqual(classify_risk_band(94), 'VERY_HIGH_RISK')
        self.assertEqual(classify_risk_band(95), 'CRITICAL')

    def test_person_risk_region_reuses_detection_coordinates(self):
        regions = build_risk_regions(
            [{'box': [10, 20, 30, 40], 'confidence': 90}],
            [{'activity': 'running', 'confidence': 90}],
            0.6,
            [],
        )
        self.assertEqual(regions[0]['personId'], 'person_01')
        self.assertEqual(regions[0]['centerX'], 25)
        self.assertEqual(regions[0]['centerY'], 40)
        self.assertGreaterEqual(regions[0]['riskScore'], 60)
        self.assertTrue(regions[0]['heatmapActive'])

    def test_standing_person_does_not_become_high_risk_from_detection_confidence(self):
        regions = build_risk_regions(
            [{'box': [10, 20, 30, 40], 'confidence': 99}],
            [{'activity': 'standing', 'confidence': 99}],
            0.02,
            [],
        )
        self.assertLess(regions[0]['riskScore'], 60)
        self.assertEqual(regions[0]['riskBand'], 'NORMAL')

    def test_jump_is_detected_from_raised_arms_compact_legs_and_motion(self):
        result = classify_activity([0, 0, 20, 80], pose_points(), 0.6)
        self.assertEqual(result['label'], 'jumping')

    def test_dance_is_detected_from_asymmetric_raised_limbs(self):
        result = classify_activity(
            [0, 0, 20, 80],
            pose_points(wrists=((42, 5), (58, 16)), ankles=((35, 52), (65, 47))),
            0.3,
        )
        self.assertEqual(result['label'], 'dancing')

    def test_running_remains_available_from_high_motion(self):
        result = classify_activity([0, 0, 20, 80], [], 0.8)
        self.assertEqual(result['label'], 'running')

    def test_person_without_meaningful_motion_is_standing(self):
        result = classify_activity([0, 0, 20, 80], [], 0.02)
        self.assertEqual(result['label'], 'standing')


if __name__ == '__main__':
    unittest.main()
