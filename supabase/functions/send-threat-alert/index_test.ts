import { createHandler, normalizedConfidence } from './index.ts';

const baseEnv: Record<string, string> = {
  ATAS_WEBHOOK_SECRET: 'test-webhook-secret',
  SUPABASE_URL: 'https://example.supabase.co',
  SUPABASE_SERVICE_ROLE_KEY: 'test-db-key',
  RESEND_API_KEY: 'test-resend-key',
  ALERT_EMAIL: 'alerts@example.com',
  RESEND_FROM_EMAIL: 'ATAS Alerts <alerts@verified.example>',
};

function row(confidence: number, status = 'pending') {
  return {
    id: '00000000-0000-4000-8000-000000000001',
    threat_type: 'Knife detection',
    threat_level: 'MEDIUM',
    confidence,
    location: 'Zone A',
    detected_at: '2026-09-24T10:00:00.000Z',
    message: 'Review required',
    email_status: status,
    email_recipient: null,
    sent_to: null,
    provider_email_id: null,
  };
}

function harness(initialRow: ReturnType<typeof row>, options: {
  env?: Record<string, string>;
  providerStatus?: number;
} = {}) {
  const state = { row: initialRow, resendCalls: 0, patchCalls: 0 };
  const env = { ...baseEnv, ...(options.env ?? {}) };
  const fetchMock = async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? 'GET';
    if (url.includes('/rest/v1/threat_alerts') && method === 'GET') return Response.json([state.row]);
    if (url.includes('/rest/v1/threat_alerts') && method === 'PATCH') {
      state.patchCalls += 1;
      const body = JSON.parse(String(init.body ?? '{}'));
      if (url.includes('email_status=in.') && !['pending', 'failed'].includes(state.row.email_status)) return Response.json([]);
      if (url.includes('email_status=eq.processing') && state.row.email_status !== 'processing') return new Response(null, { status: 204 });
      state.row = { ...state.row, ...body };
      return url.includes('select=*') ? Response.json([state.row]) : new Response(null, { status: 204 });
    }
    if (url === 'https://api.resend.com/emails') {
      state.resendCalls += 1;
      return options.providerStatus && options.providerStatus >= 400
        ? Response.json({ message: 'provider rejected recipient' }, { status: options.providerStatus })
        : Response.json({ id: 'email-id-1' });
    }
    throw new Error(`Unexpected fetch: ${method} ${url}`);
  };
  const handler = createHandler({ env: (name) => env[name], fetch: fetchMock, now: () => new Date('2026-09-24T10:01:00.000Z') });
  const request = () => new Request('https://edge.example/functions/v1/send-threat-alert', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-atas-webhook-secret': 'test-webhook-secret' },
    body: JSON.stringify({ type: 'INSERT', table: 'threat_alerts', record: { id: state.row.id } }),
  });
  return { handler, request, state };
}

Deno.test('normalizes both fraction and percentage confidence values', () => {
  assertApprox(normalizedConfidence(0.751), 0.751);
  assertApprox(normalizedConfidence(75.1), 0.751);
});

for (const confidence of [0.70, 0.7499]) {
  Deno.test(`does not email at confidence ${confidence}`, async () => {
    const h = harness(row(confidence));
    const response = await h.handler(h.request());
    assertEquals(response.status, 200);
    assertEquals(h.state.resendCalls, 0);
  });
}

for (const confidence of [0.75, 0.751, 0.80, 0.95]) {
  Deno.test(`emails above threshold at confidence ${confidence}`, async () => {
    const h = harness(row(confidence));
    const response = await h.handler(h.request());
    assertEquals(response.status, 200);
    assertEquals(h.state.resendCalls, 1);
    assertEquals(h.state.row.email_status, 'sent');
    assertEquals(h.state.row.sent_at, '2026-09-24T10:01:00.000Z');
    assertEquals(h.state.row.email_recipient, 'alerts@example.com');
    assertEquals(h.state.row.sent_to, 'alerts@example.com');
    assertEquals(h.state.row.provider_email_id, 'email-id-1');
  });
}

Deno.test('marks provider failure as failed', async () => {
  const h = harness(row(80), { providerStatus: 422 });
  const response = await h.handler(h.request());
  assertEquals(response.status, 502);
  assertEquals(h.state.row.email_status, 'failed');
  assertEquals(h.state.row.sent_at, null);
});

Deno.test('rejects invalid recipient and stores failed status', async () => {
  const h = harness(row(80), { env: { ALERT_EMAIL: 'not-an-email' } });
  const response = await h.handler(h.request());
  assertEquals(response.status, 400);
  assertEquals(h.state.row.email_status, 'failed');
  assertEquals(h.state.resendCalls, 0);
});

Deno.test('repeated webhook after success does not send again', async () => {
  const h = harness(row(80));
  await h.handler(h.request());
  await h.handler(h.request());
  assertEquals(h.state.resendCalls, 1);
  assertEquals(h.state.row.email_status, 'sent');
});

Deno.test('already-sent alert is ignored', async () => {
  const h = harness(row(80, 'sent'));
  const response = await h.handler(h.request());
  assertEquals(response.status, 200);
  assertEquals(h.state.resendCalls, 0);
});

Deno.test('missing Resend key is recorded as failed without sending', async () => {
  const h = harness(row(80), { env: { RESEND_API_KEY: '' } });
  const response = await h.handler(h.request());
  assertEquals(response.status, 500);
  assertEquals(h.state.row.email_status, 'failed');
  assertEquals(h.state.resendCalls, 0);
});

Deno.test('missing ALERT_EMAIL is recorded as failed without sending', async () => {
  const h = harness(row(80), { env: { ALERT_EMAIL: '' } });
  const response = await h.handler(h.request());
  assertEquals(response.status, 500);
  assertEquals(h.state.row.email_status, 'failed');
  assertEquals(h.state.resendCalls, 0);
});

function assertEquals(actual: unknown, expected: unknown) {
  if (actual !== expected) throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

function assertApprox(actual: number, expected: number) {
  if (Math.abs(actual - expected) > 1e-10) throw new Error(`Expected approximately ${expected}, got ${actual}`);
}
