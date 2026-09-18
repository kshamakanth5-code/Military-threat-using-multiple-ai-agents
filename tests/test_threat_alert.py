import unittest

from backend.server import HIGH_RISK_EMAIL_CONFIDENCE


class ThreatAlertPolicyTests(unittest.TestCase):
    def test_email_threshold_is_strictly_above_ninety(self):
        self.assertEqual(HIGH_RISK_EMAIL_CONFIDENCE, 90.0)
        self.assertFalse(90.0 > HIGH_RISK_EMAIL_CONFIDENCE)
        self.assertTrue(90.1 > HIGH_RISK_EMAIL_CONFIDENCE)


if __name__ == '__main__':
    unittest.main()