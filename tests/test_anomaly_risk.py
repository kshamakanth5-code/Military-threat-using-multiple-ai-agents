import unittest

from backend.anomaly_risk import build_anomaly_risk_report


class AnomalyRiskTests(unittest.TestCase):
    def test_anomaly_detection_increases_risk_for_conflicting_signals(self):
        report = build_anomaly_risk_report({
            'riskLevel': 'MEDIUM',
            'confidence': 68,
            'sensorReliability': 0.72,
            'temperatureAnomaly': 0.82,
            'motionAnomaly': 0.76,
            'audioEvent': 'breakage',
            'zoneContext': 'restricted perimeter',
            'historicalContext': 'repeated suspicious dwell',
        })
        self.assertEqual(report['anomalyStatus'], 'anomaly_detected')
        self.assertIn(report['riskLevel'], {'MEDIUM', 'HIGH'})
        self.assertGreater(report['confidence'], 50)

    def test_clean_scene_stays_low_risk(self):
        report = build_anomaly_risk_report({
            'riskLevel': 'LOW',
            'confidence': 18,
            'sensorReliability': 0.92,
            'temperatureAnomaly': 0.12,
            'motionAnomaly': 0.08,
            'audioEvent': 'none',
            'zoneContext': 'routine patrol',
            'historicalContext': 'normal activity',
        })
        self.assertEqual(report['anomalyStatus'], 'normal')
        self.assertEqual(report['riskLevel'], 'LOW')


if __name__ == '__main__':
    unittest.main()
