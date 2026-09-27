-- Delivery metadata for existing threat alert rows; safe to run on a populated table.
alter table public.threat_alerts add column if not exists sent_to text;
alter table public.threat_alerts add column if not exists provider_email_id text;
