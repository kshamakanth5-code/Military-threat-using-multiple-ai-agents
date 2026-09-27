type AlertRow = {
  id: string;
  threat_type: string;
  threat_level: string;
  confidence: number | string;
  location: string | null;
  detected_at: string;
  message: string;
  email_status: string;
  email_recipient?: string | null;
  sent_to?: string | null;
  provider_email_id?: string | null;
  sent_at?: string | null;
};

type Env = (name: string) => string | undefined;
type Fetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

const json = (body: Record<string, unknown>, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });

export function normalizedConfidence(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric > 1 ? numeric / 100 : numeric));
}

function escapeHtml(value: unknown): string {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]!);
}

function validEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function constantTimeEqual(left: string, right: string): boolean {
  const encoder = new TextEncoder();
  const a = encoder.encode(left);
  const b = encoder.encode(right);
  let difference = a.length ^ b.length;
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    difference |= (a[index % Math.max(a.length, 1)] ?? 0) ^ (b[index % Math.max(b.length, 1)] ?? 0);
  }
  return difference === 0;
}

export function createHandler(options: { env?: Env; fetch?: Fetch; now?: () => Date } = {}) {
  const readEnv = options.env ?? ((name) => Deno.env.get(name));
  const fetcher = options.fetch ?? fetch;
  const now = options.now ?? (() => new Date());

  return async (request: Request): Promise<Response> => {
    if (request.method !== 'POST') return json({ ok: false, error: 'Method not allowed.' }, 405);

    const webhookSecret = readEnv('ATAS_WEBHOOK_SECRET');
    if (!webhookSecret) return json({ ok: false, error: 'ATAS_WEBHOOK_SECRET is not configured.' }, 500);
    const suppliedSecret = request.headers.get('x-atas-webhook-secret') ?? '';
    if (!constantTimeEqual(suppliedSecret, webhookSecret)) return json({ ok: false, error: 'Unauthorized webhook request.' }, 401);

    let payload: Record<string, unknown>;
    try {
      payload = await request.json();
    } catch {
      return json({ ok: false, error: 'Request body must be valid JSON.' }, 400);
    }
    if (payload.type && payload.type !== 'INSERT') return json({ ok: true, skipped: true, reason: 'unsupported_event' });
    if (payload.table && payload.table !== 'threat_alerts') return json({ ok: false, error: 'Unexpected webhook table.' }, 400);
    const id = String((payload.record as Partial<AlertRow> | undefined)?.id ?? payload.id ?? '');
    if (!id) return json({ ok: false, error: 'Webhook payload is missing record.id.' }, 400);

    const supabaseUrl = readEnv('SUPABASE_URL')?.replace(/\/$/, '');
    let supabaseKey = readEnv('SUPABASE_SERVICE_ROLE_KEY');
    if (!supabaseKey) {
      try {
        const secretKeys = JSON.parse(readEnv('SUPABASE_SECRET_KEYS') ?? '{}') as Record<string, string>;
        supabaseKey = secretKeys.default ?? Object.values(secretKeys)[0];
      } catch { /* report the missing server credential below */ }
    }
    if (!supabaseUrl || !supabaseKey) return json({ ok: false, error: 'Supabase server credentials are not configured.' }, 500);

    const dbHeaders = {
      apikey: supabaseKey,
      authorization: `Bearer ${supabaseKey}`,
      'content-type': 'application/json',
    };
    const rowUrl = `${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&select=*`;
    let row: AlertRow;
    try {
      const response = await fetcher(rowUrl, { headers: dbHeaders });
      if (!response.ok) return json({ ok: false, error: `Could not load alert row (HTTP ${response.status}).` }, 502);
      const rows = await response.json() as AlertRow[];
      if (!rows.length) return json({ ok: false, error: 'Alert row was not found.' }, 404);
      row = rows[0];
    } catch {
      return json({ ok: false, error: 'Could not reach Supabase to load the alert.' }, 502);
    }

    const confidence = normalizedConfidence(row.confidence);
    if (confidence < 0.75) return json({ ok: true, skipped: true, reason: 'confidence_below_75_percent' });
    if (row.email_status === 'sent') return json({ ok: true, skipped: true, reason: 'already_sent' });
    const retryInProgress = row.email_status === 'processing';
    if (!retryInProgress && !['pending', 'failed'].includes(row.email_status)) {
      return json({ ok: true, skipped: true, reason: 'alert_not_queued' });
    }

    const recipient = (readEnv('ALERT_EMAIL') ?? '').trim();
    const resendKey = readEnv('RESEND_API_KEY');
    const from = (readEnv('RESEND_FROM_EMAIL') ?? '').trim();
    const missing = [
      !recipient && 'ALERT_EMAIL',
      !resendKey && 'RESEND_API_KEY',
      !from && 'RESEND_FROM_EMAIL',
    ].filter(Boolean);
    if (!recipient) console.error(`[send-threat-alert] ALERT_EMAIL is missing for alert ${id}.`);
    if (!resendKey) console.error(`[send-threat-alert] RESEND_API_KEY is missing for alert ${id}.`);
    if (!from) console.error(`[send-threat-alert] RESEND_FROM_EMAIL is missing for alert ${id}.`);
    const fail = async (reason: string, status = 500) => {
      try {
        const statuses = row.email_status === 'processing' ? 'pending,failed,processing' : 'pending,failed';
        await fetcher(`${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&email_status=in.%28${statuses}%29`, {
          method: 'PATCH', headers: { ...dbHeaders, prefer: 'return=minimal' },
          body: JSON.stringify({ email_status: 'failed', sent_at: null, email_recipient: null, sent_to: null }),
        });
      } catch { /* keep the original, useful error response */ }
      return json({ ok: false, error: reason }, status);
    };
    if (missing.length) return await fail(`Missing Edge Function secret(s): ${missing.join(', ')}.`);
    if (!validEmail(recipient)) return await fail('ALERT_EMAIL is not a valid email address.', 400);
    if (!validEmail(from.match(/<([^>]+)>/)?.[1] ?? from)) {
      return await fail('RESEND_FROM_EMAIL must be a valid email or a display name with a valid email.', 400);
    }

    // Atomically claim pending/failed rows. A retry of a processing row uses
    // the same Resend idempotency key to recover after a function interruption.
    if (!retryInProgress) {
      let claimed: AlertRow[];
      try {
        const claimUrl = `${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&email_status=in.%28pending%2Cfailed%29&select=*`;
        const response = await fetcher(claimUrl, {
          method: 'PATCH',
          headers: { ...dbHeaders, prefer: 'return=representation' },
          body: JSON.stringify({ email_status: 'processing', email_recipient: recipient, sent_at: null }),
        });
        if (!response.ok) return json({ ok: false, error: `Could not claim alert for delivery (HTTP ${response.status}).` }, 502);
        claimed = await response.json() as AlertRow[];
      } catch {
        return json({ ok: false, error: 'Could not claim alert for delivery.' }, 502);
      }
      if (!claimed.length) return json({ ok: true, skipped: true, reason: 'duplicate_or_already_processing' });
      row = claimed[0];
    }

    const percent = (confidence * 100).toFixed(1);
    const threatType = String(row.threat_type || 'Threat detected');
    const threatLevel = String(row.threat_level || 'HIGH');
    const location = String(row.location || 'Not available');
    const detectedAt = String(row.detected_at || 'Not available');
    const message = String(row.message || 'Immediate operator attention is required.');
    const subjectType = threatType.replace(/[\r\n]+/g, ' ').slice(0, 160);
    const subject = `🚨 HIGH THREAT ALERT - ${subjectType}`;
    const text = [
      'HIGH THREAT DETECTED', '',
      `Threat Type: ${threatType}`,
      `Threat Level: ${threatLevel}`,
      `Confidence: ${percent}%`,
      `Location: ${location}`,
      `Detected At: ${detectedAt}`, '',
      'Message:', message, '',
      'Immediate attention is required. Please review the ATAS monitoring dashboard.',
    ].join('\n');
    const html = `<h1>HIGH THREAT DETECTED</h1><ul><li><b>Threat Type:</b> ${escapeHtml(threatType)}</li><li><b>Threat Level:</b> ${escapeHtml(threatLevel)}</li><li><b>Confidence:</b> ${percent}%</li><li><b>Location:</b> ${escapeHtml(location)}</li><li><b>Detected At:</b> ${escapeHtml(detectedAt)}</li></ul><h2>Message</h2><p>${escapeHtml(message).replace(/\n/g, '<br>')}</p><p><strong>Immediate attention is required.</strong> Please review the ATAS monitoring dashboard.</p>`;

    let resendResponse: Response;
    try {
      resendResponse = await fetcher('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          authorization: `Bearer ${resendKey}`,
          'content-type': 'application/json',
          'Idempotency-Key': `atas-threat-alert/${id}`,
        },
        body: JSON.stringify({ from, to: [recipient], subject, text, html }),
      });
    } catch (error) {
      console.error(`[send-threat-alert] Resend request failed for alert ${id}: ${error instanceof Error ? error.message : 'unknown error'}`);
      await fetcher(`${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&email_status=eq.processing`, {
        method: 'PATCH', headers: { ...dbHeaders, prefer: 'return=minimal' },
        body: JSON.stringify({ email_status: 'failed', sent_at: null, email_recipient: recipient }),
      }).catch(() => undefined);
      return json({ ok: false, error: 'Resend request failed before a response was received.' }, 502);
    }
    if (!resendResponse.ok) {
      const detail = (await resendResponse.text()).slice(0, 500);
      console.error(`[send-threat-alert] Resend rejected alert ${id} (HTTP ${resendResponse.status}): ${detail}`);
      if (resendResponse.status === 409) {
        return json({ ok: false, error: 'Resend reports an idempotent request is already in progress; retry this webhook shortly.', detail }, 503);
      }
      await fetcher(`${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&email_status=eq.processing`, {
        method: 'PATCH', headers: { ...dbHeaders, prefer: 'return=minimal' },
        body: JSON.stringify({ email_status: 'failed', sent_at: null, email_recipient: recipient }),
      }).catch(() => undefined);
      return json({ ok: false, error: `Resend rejected the email (HTTP ${resendResponse.status}).`, detail }, 502);
    }

    let providerEmailId: string | null = null;
    try {
      const accepted = await resendResponse.json() as { id?: string };
      providerEmailId = accepted.id ?? null;
    } catch { /* Resend accepted the request even if its response body was empty. */ }
    console.info(`[send-threat-alert] Resend accepted alert ${id}; provider email id ${providerEmailId ?? 'unavailable'}.`);

    const sentAt = now().toISOString();
    try {
      const updateUrl = `${supabaseUrl}/rest/v1/threat_alerts?id=eq.${encodeURIComponent(id)}&email_status=eq.processing`;
      const response = await fetcher(updateUrl, {
        method: 'PATCH', headers: { ...dbHeaders, prefer: 'return=minimal' },
        body: JSON.stringify({ email_status: 'sent', sent_at: sentAt, email_recipient: recipient, sent_to: recipient, provider_email_id: providerEmailId }),
      });
      if (!response.ok) return json({ ok: false, error: `Email sent, but alert status update failed (HTTP ${response.status}); resend the webhook to reconcile.` }, 502);
    } catch {
      return json({ ok: false, error: 'Email sent, but alert status update failed; resend the webhook to reconcile.' }, 502);
    }
    return json({ ok: true, alertId: id, emailStatus: 'sent', sentAt, recipient, providerEmailId });
  };
}

if (import.meta.main) Deno.serve(createHandler());
