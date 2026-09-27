import hashlib
import os
import sqlite3
from contextlib import closing
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from backend import server


class PasswordResetTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).resolve().parents[1] / 'backend' / '.password-reset-test.db'
        self.db_path.unlink(missing_ok=True)
        self.db_patch = patch.object(server, 'ACCOUNT_DB', self.db_path)
        self.db_patch.start()
        self.client = TestClient(server.app)
        server.initialize_account_db()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                'INSERT INTO accounts (user_id, email, user_name, password_hash) VALUES (?, ?, ?, ?)',
                ('user-a', 'registered@example.com', 'Operator A', server.password_digest('old-password')),
            )
        self.email_mock = patch('backend.server.send_password_reset_email', return_value=True)
        self.send_email = self.email_mock.start()
        self.env_patch = patch.dict(os.environ, {
            'FRONTEND_URL': 'http://localhost:4173',
            'PASSWORD_RESET_EXPIRY_MINUTES': '15',
            'PASSWORD_RESET_MAX_REQUESTS': '3',
            'PASSWORD_RESET_WINDOW_MINUTES': '15',
        })
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.email_mock.stop()
        self.db_patch.stop()
        self.db_path.unlink(missing_ok=True)

    def request_reset(self, email='registered@example.com', **extra):
        return self.client.post('/api/forgot-password', json={'email': email, **extra})

    def captured_token(self):
        reset_url = self.send_email.call_args.args[1]
        return parse_qs(urlparse(reset_url).query)['token'][0]

    def test_reset_targets_database_email_and_new_password_replaces_old(self):
        response = self.request_reset(email='REGISTERED@example.com', email_to_send_reset_link='attacker@example.com')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['message'], server._password_reset_message()['message'])
        self.send_email.assert_called_once()
        self.assertEqual(self.send_email.call_args.args[0], 'registered@example.com')
        self.assertTrue(self.send_email.call_args.args[1].startswith('http://localhost:4173/?token='))

        token = self.captured_token()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            saved_hash, used_at = connection.execute(
                'SELECT token_hash, used_at FROM password_reset_tokens'
            ).fetchone()
            self.assertEqual(saved_hash, token_hash)
            self.assertIsNone(used_at)
        self.assertNotIn(token, saved_hash)

        reset_response = self.client.post('/api/reset-password', json={'token': token, 'newPassword': 'new-password'})
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(reset_response.json()['message'], 'Password reset successfully.')
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            saved_password = connection.execute('SELECT password_hash FROM accounts WHERE user_id = ?', ('user-a',)).fetchone()[0]
        self.assertNotEqual(saved_password, 'new-password')

        self.assertFalse(self.client.post('/api/login', json={'email': 'registered@example.com', 'password': 'old-password'}).json()['ok'])
        self.assertTrue(self.client.post('/api/login', json={'email': 'registered@example.com', 'password': 'new-password'}).json()['ok'])
        reuse_response = self.client.post('/api/reset-password', json={'token': token, 'newPassword': 'third-password'})
        self.assertEqual(reuse_response.status_code, 400)
        self.assertIn('invalid or has already been used', reuse_response.json()['detail'])

    def test_unknown_email_has_the_same_generic_response_and_sends_nothing(self):
        known = self.request_reset()
        unknown = self.request_reset('not-registered@example.com')
        self.assertEqual(known.status_code, unknown.status_code)
        self.assertEqual(known.json(), unknown.json())
        self.assertEqual(self.send_email.call_count, 1)

    def test_expired_and_weak_password_tokens_are_rejected(self):
        response = self.request_reset()
        token = self.captured_token()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                'UPDATE password_reset_tokens SET expires_at = ? WHERE token_hash = ?',
                ('2000-01-01T00:00:00+00:00', token_hash),
            )
        expired = self.client.post('/api/reset-password', json={'token': token, 'newPassword': 'long-enough'})
        self.assertEqual(expired.status_code, 400)
        self.assertEqual(expired.json()['detail'], 'Password reset link has expired.')

        self.request_reset()
        fresh_token = self.captured_token()
        weak = self.client.post('/api/reset-password', json={'token': fresh_token, 'newPassword': '123'})
        self.assertEqual(weak.status_code, 400)

    def test_reset_requests_are_rate_limited_without_changing_generic_response(self):
        with patch.dict(os.environ, {'PASSWORD_RESET_MAX_REQUESTS': '1'}):
            first = self.request_reset()
            second = self.request_reset()
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.send_email.call_count, 1)

    def test_failed_email_delivery_invalidates_the_undelivered_link(self):
        self.send_email.return_value = False
        response = self.request_reset()
        self.assertEqual(response.status_code, 200)
        token = self.captured_token()
        rejected = self.client.post('/api/reset-password', json={'token': token, 'newPassword': 'new-password'})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn('invalid or has already been used', rejected.json()['detail'])


if __name__ == '__main__':
    unittest.main()
