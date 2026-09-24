import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from backend.agents import process_threat_event


class ThreatAlertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.database = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.database.close()
        process_threat_event._state.clear()
        self.sender = lambda *args: True
        self.user = {'userId': 'user-a', 'email': 'user-a@example.com'}

    def tearDown(self):
        os.unlink(self.database.name)

    def event(self, risk, user=None):
        return {'riskScore': risk, 'threatLevel': 'HIGH', 'reason': 'Test threat', 'timestamp': '2026-09-23T10:00:00+00:00', 'source': 'test'}

    def process(self, risk, user=None, sender=None):
        return process_threat_event(self.event(risk), user or self.user, self.database.name, sender or self.sender)

    def test_risk_below_threshold_does_not_alert(self):
        result = self.process(89)
        self.assertEqual(result['emailStatus'], 'NOT_ELIGIBLE')
        self.assertFalse(result['emailSent'])

    def test_threshold_requires_confirmation(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '5', 'RISK_THRESHOLD': '95'}):
            for _ in range(4):
                result = self.process(95)
            self.assertEqual(result['emailStatus'], 'AWAITING_CONFIRMATION')
            self.assertEqual(self.process(95)['emailStatus'], 'SENT')

    def test_confirmed_high_risk_uses_authenticated_email(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '1'}):
            recipients = []
            result = self.process(97, sender=lambda recipient, *args: recipients.append(recipient) or True)
        self.assertTrue(result['emailSent'])
        self.assertEqual(recipients, ['user-a@example.com'])

    def test_cooldown_allows_only_one_email_for_repeated_frames(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '5', 'ALERT_COOLDOWN_MINUTES': '5'}):
            sent = []
            for _ in range(100):
                result = self.process(99, sender=lambda recipient, *args: sent.append(recipient) or True)
        self.assertEqual(len(sent), 1)
        self.assertEqual(result['emailStatus'], 'COOLDOWN')

    def test_dropping_below_threshold_resets_confirmation(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '3'}):
            self.process(95)
            self.process(95)
            self.assertEqual(self.process(94)['emailStatus'], 'NOT_ELIGIBLE')
            self.assertEqual(self.process(95)['confirmationFrames'], 1)

    def test_email_failure_is_stored_without_raising(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '1'}):
            result = self.process(97, sender=lambda *args: False)
        self.assertFalse(result['emailSent'])
        self.assertEqual(result['emailStatus'], 'FAILED')
        with closing(sqlite3.connect(self.database.name)) as connection:
            self.assertEqual(connection.execute('SELECT email_sent, email_status FROM alerts').fetchone(), (0, 'FAILED'))

    def test_each_authenticated_user_receives_only_their_alert(self):
        with patch.dict(os.environ, {'CONFIRMATION_FRAMES': '1'}):
            recipients = []
            self.process(97, user={'userId': 'user-a', 'email': 'a@example.com'}, sender=lambda recipient, *args: recipients.append(recipient) or True)
            self.process(97, user={'userId': 'user-b', 'email': 'b@example.com'}, sender=lambda recipient, *args: recipients.append(recipient) or True)
        self.assertEqual(recipients, ['a@example.com', 'b@example.com'])


if __name__ == '__main__':
    unittest.main()