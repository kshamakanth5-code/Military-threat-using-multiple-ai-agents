-- ATAS email-alert queue for Supabase.
-- The application/backend must insert HIGH and CRITICAL events here.
-- PostgreSQL never sends email; an application worker or Edge Function processes pending rows.

create table if not exists public.threat_alerts (
  id uuid primary key default gen_random_uuid(),
  threat_type text not null,
  threat_level text not null default 'LOW',
  confidence numeric(5, 2) not null default 0,
  location text,
  detected_at timestamptz not null default timezone('utc', now()),
  message text not null,
  email_recipient text,
  email_status text not null default 'not_required',
  sent_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  constraint threat_alerts_threat_level_check
    check (threat_level in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
  constraint threat_alerts_confidence_check
    check (confidence >= 0 and confidence <= 100),
  constraint threat_alerts_email_status_check
    check (email_status in ('not_required', 'pending', 'processing', 'sent', 'failed'))
);

-- If the table existed already, add only missing columns. Existing data is preserved.
alter table public.threat_alerts add column if not exists id uuid default gen_random_uuid();
alter table public.threat_alerts add column if not exists threat_type text;
alter table public.threat_alerts add column if not exists threat_level text default 'LOW';
alter table public.threat_alerts add column if not exists confidence numeric(5, 2) default 0;
alter table public.threat_alerts add column if not exists location text;
alter table public.threat_alerts add column if not exists detected_at timestamptz default timezone('utc', now());
alter table public.threat_alerts add column if not exists message text;
alter table public.threat_alerts add column if not exists email_recipient text;
alter table public.threat_alerts add column if not exists email_status text default 'not_required';
alter table public.threat_alerts add column if not exists sent_at timestamptz;
alter table public.threat_alerts add column if not exists created_at timestamptz default timezone('utc', now());

-- Add missing defaults without overwriting existing values.
alter table public.threat_alerts alter column threat_level set default 'LOW';
alter table public.threat_alerts alter column confidence set default 0;
alter table public.threat_alerts alter column detected_at set default timezone('utc', now());
alter table public.threat_alerts alter column email_status set default 'not_required';
alter table public.threat_alerts alter column created_at set default timezone('utc', now());

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.threat_alerts'::regclass
      and contype = 'p'
  ) then
    alter table public.threat_alerts add constraint threat_alerts_pkey primary key (id);
  end if;
end;
$$;

-- Add the requested constraints only when they are not already present.
do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.threat_alerts'::regclass
      and conname = 'threat_alerts_threat_level_check'
  ) then
    alter table public.threat_alerts
      add constraint threat_alerts_threat_level_check
      check (threat_level in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'));
  end if;

  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.threat_alerts'::regclass
      and conname = 'threat_alerts_confidence_check'
  ) then
    alter table public.threat_alerts
      add constraint threat_alerts_confidence_check
      check (confidence >= 0 and confidence <= 100);
  end if;

  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.threat_alerts'::regclass
      and conname = 'threat_alerts_email_status_check'
  ) then
    alter table public.threat_alerts
      add constraint threat_alerts_email_status_check
      check (email_status in ('not_required', 'pending', 'processing', 'sent', 'failed'));
  end if;
end;
$$;

create index if not exists threat_alerts_threat_level_idx
  on public.threat_alerts (threat_level);

create index if not exists threat_alerts_detected_at_idx
  on public.threat_alerts (detected_at desc);

create index if not exists threat_alerts_email_status_idx
  on public.threat_alerts (email_status);

-- Enforce the queue contract for critical events. This does not send email.
create or replace function public.queue_high_threat_email_alert()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
  if new.threat_level in ('HIGH', 'CRITICAL') then
    new.email_status := 'pending';
    new.sent_at := null;
  elsif new.email_status is null then
    new.email_status := 'not_required';
  end if;
  return new;
end;
$$;

drop trigger if exists threat_alerts_queue_email on public.threat_alerts;
create trigger threat_alerts_queue_email
before insert on public.threat_alerts
for each row
execute function public.queue_high_threat_email_alert();

-- Existing ATAS authentication is local SQLite, not Supabase Auth. Keep the
-- table inaccessible to browser clients; the backend/worker should use the
-- Supabase service role key server-side. Service-role access bypasses RLS.
alter table public.threat_alerts enable row level security;

-- Do not create public INSERT/SELECT policies. With RLS enabled and no policies,
-- anon/authenticated browser roles cannot read or write alert records.
