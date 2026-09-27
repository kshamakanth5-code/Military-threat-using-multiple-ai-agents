import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from backend import agents
from backend.agents import process_threat_event


class ThreatAlertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.database = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.database.close()
        self.supabase = patch.object(agents, '_supabase_request', return_value=None)
        self.supabase_mock = self.supabase.start()
        self.supabase_mock.side_effect = self._mock_supabase
        self.smtp = patch.object(agents, 'send_threat_notification', return_value=True)
        self.smtp_mock = self.smtp.start()
        self.alert_env = patch.dict(os.environ, {'ALERT_EMAIL': 'alerts@example.com'})
        self.alert_env.start()
        self.user = {'userId': 'user-a', 'email': 'operator@example.com'}

    def _mock_supabase(self, method, path, body=None, *, prefer=None):
        if method == 'PATCH' and 'email_status=eq.pending' in path and prefer == 'return=representation':
            return [{'id': 'claimed-alert'}]
        return None

    def tearDown(self):
        self.supabase.stop()
        self.smtp.stop()
        self.alert_env.stop()
        os.unlink(self.database.name)

    def process(self, confidence, user_id=None):
        event = {
            'riskScore': confidence,
            'threatLevel': 'MEDIUM',
            'reason': 'Test threat',
            'timestamp': '2026-09-23T10:00:00+00:00',
            'source': 'test',
        }
        user = {**self.user, 'userId': user_id or self.user['userId']}
        return process_threat_event(event, user, self.database.name)

    def test_confidence_below_75_percent_does_not_queue(self):
        for confidence in (0.70, 0.7499, 70, 74.99):
            with self.subTest(confidence=confidence):
                self.supabase_mock.reset_mock()
                self.smtp_mock.reset_mock()
                result = self.process(confidence, f'user-{confidence}')
                self.assertEqual(result['emailStatus'], 'NOT_ELIGIBLE')
                self.supabase_mock.assert_not_called()
                self.smtp_mock.assert_not_called()

    def test_at_or_above_75_percent_queues_and_sends_via_smtp(self):
        for confidence in (0.75, 75, 0.751, 75.1, 80, 95):
            with self.subTest(confidence=confidence):
                self.supabase_mock.reset_mock()
                self.smtp_mock.reset_mock()
                result = self.process(confidence, f'user-{confidence}')
                self.assertEqual(result['emailStatus'], 'SENT')
                self.assertTrue(result['emailSent'])
                self.assertEqual(self.supabase_mock.call_count, 3)
                self.smtp_mock.assert_called_once()
                self.assertEqual(self.smtp_mock.call_args.args[0], 'alerts@example.com')
                method, path, body = self.supabase_mock.call_args_list[0].args[:3]
                self.assertEqual((method, path), ('POST', 'threat_alerts'))
                self.assertGreaterEqual(body['confidence'] / 100, 0.75)
                self.assertEqual(body['email_status'], 'pending')
                self.assertIsNone(body['email_recipient'])
                final_update = self.supabase_mock.call_args_list[-1].args[2]
                self.assertEqual(final_update['email_status'], 'sent')
                self.assertEqual(final_update['sent_to'], 'alerts@example.com')

    def test_confidence_gate_does_not_depend_on_threat_level(self):
        event = {'riskScore': 80, 'threatLevel': 'LOW', 'reason': 'Test', 'timestamp': '2026-09-23T10:00:00+00:00'}
        result = process_threat_event(event, self.user, self.database.name)
        self.assertEqual(result['emailStatus'], 'SENT')
        self.assertEqual(self.supabase_mock.call_count, 3)
        self.smtp_mock.assert_called_once()

    def test_cooldown_suppresses_repeated_camera_frames(self):
        self.process(80)
        result = self.process(95)
        self.assertEqual(result['emailStatus'], 'COOLDOWN')
        self.assertEqual(self.supabase_mock.call_count, 3)
        self.assertEqual(self.smtp_mock.call_count, 1)

    def test_smtp_failure_is_recorded_as_failed(self):
        self.smtp_mock.return_value = False
        result = self.process(80)
        self.assertEqual(result['emailStatus'], 'FAILED')
        self.assertFalse(result['emailSent'])
        self.assertGreaterEqual(self.supabase_mock.call_count, 3)
        final_update = self.supabase_mock.call_args_list[-1].args[2]
        self.assertEqual(final_update['email_status'], 'failed')
        self.assertIsNone(final_update['sent_to'])


if __name__ == '__main__':
    unittest.main()
