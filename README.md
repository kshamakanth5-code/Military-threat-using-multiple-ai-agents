# ATAS: AI Threat Anticipation System

ATAS is a multi-agent, real-time computer-vision threat anticipation system. It receives camera frames, detects people and objects, extracts human pose, recognizes activity, builds contextual risk evidence, visualizes risk on the live dashboard, and routes confirmed critical events to the Alert Agent.

The system is designed as a decision-support and early-warning platform. Its outputs are risk estimates that require human verification; they are not proof of criminal intent.

## Core Pipeline

```text
Live camera or uploaded video
        |
        v
Detection Agent
        |
        v
Pose Extraction Agent
        |
        v
Activity Recognition Agent
        |
        v
Intent Prediction Agent
        |
        v
Threat Analysis Agent
        |
        v
Structured threat event and risk score
        |
        +----------------------+
        |                      |
        v                      v
Heatmap visualization       Alert & Learning Agent
                               |
                               v
                         Email Notification Service
                               |
                               v
                    Local SQLite alert persistence
```

The frontend agent definitions are composed in `agents/index.js`. The live camera inference path is implemented by the FastAPI backend and called by `frontend/src/App.jsx`.

## Project Structure

```text
.
â”œâ”€â”€ agents/                         JavaScript agent definitions
â”‚   â”œâ”€â”€ detectionAgent.js
â”‚   â”œâ”€â”€ poseExtractionAgent.js
â”‚   â”œâ”€â”€ activityRecognitionAgent.js
â”‚   â”œâ”€â”€ intentPredictionAgent.js
â”‚   â”œâ”€â”€ threatAnalysisAgent.js
â”‚   â”œâ”€â”€ explainabilityAgent.js
â”‚   â”œâ”€â”€ alertLearningAgent.js
â”‚   â””â”€â”€ index.js
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ server.py                   FastAPI entrypoint and live frame route
â”‚   â”œâ”€â”€ agents.py                   Agent mesh and Alert Agent policy
â”‚   â”œâ”€â”€ email_service.py            SMTP threat notification service
â”‚   â”œâ”€â”€ temporal_context.py         Temporal movement and aggression context
â”‚   â”œâ”€â”€ multimodal_risk.py          Sensor and evidence aggregation
â”‚   â”œâ”€â”€ anomaly_risk.py             Anomaly risk report
â”‚   â”œâ”€â”€ facial_expression.py        Facial-expression and next-action helpers
â”‚   â”œâ”€â”€ train_pipeline.py           Normal/wall-crossing image model pipeline
â”‚   â”œâ”€â”€ activity_recognition.py     Activity labels, pose features, GRU and temporal smoother
â”‚   â”œâ”€â”€ train_activity.py           Subject-separated pose-sequence trainer and evaluator
â”‚   â”œâ”€â”€ model_meta.json             Threat model metadata
â”‚   â””â”€â”€ accounts.db                 Local account/session/alert database at runtime
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ src/App.jsx                 React dashboard, login, camera, heatmap
â”‚   â”œâ”€â”€ src/lib/supabase.js         Optional Supabase client
â”‚   â”œâ”€â”€ package.json                Vite frontend dependencies and scripts
â”‚   â””â”€â”€ .env.local                  Local Supabase frontend configuration
â”œâ”€â”€ dataset_frames/                 Normal and wall-crossing image data
â”œâ”€â”€ tests/                          Standard-library regression tests
â”œâ”€â”€ supabase/migrations/            Optional Supabase SQL migration
â”œâ”€â”€ requirements.txt                Python dependencies
â”œâ”€â”€ .env.example                    SMTP and risk-policy template
â””â”€â”€ .gitignore                      Local secrets, databases, models, and builds
```

## Technology Stack

### Backend

- Python 3
- FastAPI
- Uvicorn
- SQLite for the current local account and alert runtime
- PyTorch and Torchvision
- Ultralytics YOLO for object and pose detection
- Pillow and OpenCV-related dependencies for image handling
- Python `smtplib` for SMTP email delivery

### Frontend

- React 19
- Vite
- Framer Motion
- Lucide React
- Browser camera APIs using `getUserMedia`
- Canvas frame capture and periodic backend analysis

### Optional Database Integration

The frontend includes `@supabase/supabase-js` and an optional client in `frontend/src/lib/supabase.js`. The current runtime still uses SQLite. Supabase is prepared for a server-side or Edge Function alert queue and does not replace the existing SQLite authentication flow automatically.

## Agents

### 1. Detection Agent

Uses the available YOLO models to identify people and detected objects. Person detections include normalized bounding boxes and confidence values. Overlapping person boxes are deduplicated using intersection-over-union logic so one person is not counted twice.

Dangerous-object detections are restricted to knife, gun, launcher, bomb, and other supported weapon labels. Other objects detected by the general COCO model are returned as harmless objects and do not raise the threat score. `backend/train_weapon.py` prepares a filtered knife/gun/launcher dataset; when the Simuletic CCTV dataset is downloaded to `activity_dataset/cctv-weapon-dataset`, it also uses its generic weapon boxes as a separate class and its person-only frames as negative examples. The Simuletic dataset is synthetic and CC BY 4.0; it has no bomb-specific labels, so bomb detection remains open-vocabulary and needs a separately labeled evaluation set.

Download the Simuletic source files while preserving their YOLO annotations, then prepare and train the detector:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Simuletic/cctv-weapon-dataset', repo_type='dataset', local_dir='activity_dataset/cctv-weapon-dataset')"
python -m backend.train_weapon
```

The filtered dataset and trained checkpoints are local generated artifacts covered by `.gitignore`. The HF sample has six scene groups; the trainer holds out Scenes 5 and 6 for validation. Its generic weapon boxes are not relabeled as guns or knives. The latest fine-tune scored very poorly on that generic weapon class, so the backend keeps the better validated local detector first and only uses the HF checkpoint as a fallback when the local checkpoint is absent.

For sharp objects, the live path combines `runs/detect/sharp_object_detector/weights/best.pt` with YOLO-World prompts for blades, shiny blades, and sharp metal objects. On the 32-image `dataset_frames/sharp_object_detection/val` split, this ensemble matched 37 of 41 labeled knife/sword boxes at IoU 0.50, with one unmatched detection (90.2% recall, 97.4% precision). That validation set has very few small objects, so it does not establish reliable performance for tiny blades.

### 2. Pose Extraction Agent

Uses `yolo11n-pose.pt` to extract human keypoints. Keypoints include normalized coordinates and confidence values. Pose boxes support activity classification and dashboard skeleton overlays.

### 3. Activity Recognition Agent

The live recognizer reuses YOLO11's COCO-17 pose keypoints and keeps a short pose history for each tracked person. It applies temporal voting and can load a trained GRU sequence model from `backend/activity_model.pth`. Its extensible labels include:

- standing
- sitting
- squatting
- walking
- running
- jumping
- dancing
- waving
- hands up
- clapping
- kicking
- bending
- falling
- lying
- crawling
- sleeping
- jogging
- kneeling
- raising hands
- turning
- stopping
- starting to walk
- starting to run
- climbing
- crouching

When a full activity GRU is not available, the live path reports sustained per-person movement as `MOVING` from the tracked box trajectory and uses pose geometry for coarse postures. Specific gait labels such as walking versus running still need a reliable, real-camera temporal model. Training data layout and limitations are documented in `activity_dataset/README.md`.

The API response contains per-person `activityPrediction` and `activityPredictions` records with activity, normalized confidence, timestamp, pose availability, and a stable per-user person ID. Set `ACTIVITY_DEBUG=true` to include keypoints and the corresponding risk score in the API's `activityDebug` field without changing the UI.

### 4. Intent Prediction Agent

Provides an intent-oriented interpretation of movement and trajectory context, such as routine patrol, suspicious intent, restricted approach, concealment, or possible intrusion.

### 5. Threat Analysis Agent

Combines detection, pose, motion, activity, object evidence, temporal context, anomaly reports, and sensor availability into a structured threat result. The live response includes fields such as:

```json
{
  "threatLevel": "HIGH",
  "riskScore": 82,
  "confidence": 82,
  "activity": "running",
  "timestamp": "2026-09-24T10:00:00+00:00",
  "threatEvent": {
    "userId": "authenticated-user-id",
    "riskScore": 82,
    "threatLevel": "HIGH",
    "reason": "...",
    "source": "live_camera"
  }
}
```

### 6. Explainability Agent

Generates operator-facing rationale and contextual explanations from the detected evidence. It does not make email decisions.

### 7. Alert Agent

Implemented in `backend/agents.py` as `process_threat_event`. It:

1. Receives the structured threat event.
2. Reads the authenticated user ID and email from the server session.
3. Normalizes confidence to the range 0â€“1 and queues an alert at `0.75` or higher.
4. Applies the existing per-user cooldown so camera frames from one continuing incident do not create an email every two seconds.
5. Inserts a `pending` row in Supabase and records the audit result in SQLite.
6. The trusted FastAPI backend atomically claims the queued row, sends through configured SMTP, and updates the Supabase row only after the SMTP server accepts the message.

### 8. Learning Agent

The existing Alert & Learning Agent mesh records feedback and exposes an agent-feedback endpoint. Operator feedback is stored for later model refinement; automatic retraining is not claimed by the current implementation.

## Authentication and Security

Registration and login are implemented by the FastAPI backend using local SQLite accounts. Passwords are stored as PBKDF2-SHA256 digests with per-account salts.

Successful login returns:

- `userId`
- authenticated email
- user name
- session token

Protected analysis and alert routes require:

```text
Authorization: Bearer <sessionToken>
```

The backend resolves the recipient email from the session. The frontend cannot select an arbitrary alert recipient.

Security measures include:

- SMTP credentials remain server-side.
- `.env` and `*.env.local` are ignored by Git.
- SMTP passwords, API keys, and tokens are not logged.
- Alert requests without a valid session return `401`.
- Email content sanitizes dynamic values.
- Email failures do not stop frame processing.
- Alert delivery failures are persisted as failed statuses.

## Risk and Heatmap Policy

The current heatmap uses per-person and per-object risk regions returned by `/api/analyze-frame`.

Each region contains:

```json
{
  "personId": "person_01",
  "entityType": "person",
  "x": 10,
  "y": 20,
  "width": 30,
  "height": 40,
  "centerX": 25,
  "centerY": 40,
  "riskScore": 72,
  "riskBand": "SERIOUS_RISK",
  "heatmapActive": true,
  "activity": "running"
}
```

Coordinates are percentages of the camera frame, so the frontend can overlay them directly over the existing video.

### Risk Colors

| Risk | Meaning | Color |
|---:|---|---|
| 0-29% | Normal | Blue |
| 30-59% | Low risk | Green |
| 60-74% | High-risk zone begins | Yellow |
| 75-89% | Serious risk | Orange |
| 90-94% | Very high risk | Red |
| 95-100% | Critical threat | Deep purple |

The dashboard provides two camera modes:

- **Normal camera:** live feed without heatmap tint.
- **Heatmap camera:** full-screen risk tint plus localized heat regions and detections.

## Alert Policy

Default policy:

```env
RISK_THRESHOLD_HEATMAP=60
ALERT_COOLDOWN_MINUTES=5
```

- Risk below 60% does not activate a risk heatmap.
- Risk from 60% activates the heatmap.
- Email eligibility uses normalized confidence `confidence >= 0.75` (75.0% qualifies).
- An alert attempt suppresses repeat emails for five minutes per user, including provider failures.
- Edge Function email failures are recorded as `email_status='failed'` and do not crash camera analysis.

The Alert Agent does not send one email per frame.

## Email Notifications

Threat alert and operator confirmation email use server-side SMTP configured in the root `.env`. For Gmail, use a Google App Password rather than the normal account password. The alert recipient is the server-side `ALERT_EMAIL` value:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=sender@example.com
SMTP_USERNAME=sender@example.com
SMTP_PASSWORD=google-app-password
ALERT_EMAIL=operator@example.com
```

The threat email gate is based on normalized confidence: `confidence >= 0.75`. The Supabase column stores the existing percentage scale. A successful SMTP send sets `email_status='sent'`, `sent_to`, and `sent_at`; failures remain failed and leave `sent_to` null. Do not enable the Resend Database Webhook while using SMTP, or both providers may send the same alert. The Edge Function remains available for a future Resend setup with a verified sender domain.

The dashboard also supports a browser notification after a confirmed alert if notification permission is granted. A true background mobile push notification would require a push provider such as Firebase Cloud Messaging and a service worker.

## API Routes

Important backend routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Backend/model/email configuration status |
| POST | `/api/register` | Create a local operator account |
| POST | `/api/login` | Authenticate and issue a session token |
| POST | `/api/forgot-password` | Request a reset link (always returns a generic response) |
| POST | `/api/reset-password` | Validate a one-time token and replace the password |
| POST | `/api/resend-confirmation` | Resend confirmation through the authenticated account email |
| POST | `/api/analyze-frame` | Analyze a live camera frame and return agents, risk, regions, and alert state |
| POST | `/api/threat-alert` | Process an authenticated manual threat event |
| GET | `/api/agents` | Return the current agent mesh |
| POST | `/api/agent-feedback` | Store operator activity feedback |
| GET | `/api/dataset` | Return dataset and model overview |
| POST | `/api/multimodal-risk` | Build a multimodal risk report |
| POST | `/api/temporal-context` | Build a temporal movement report |
| POST | `/api/anomaly-risk` | Build an anomaly risk report |
| POST | `/api/predict` | Predict a stored image using the threat model |
| POST | `/api/train` | Train the existing image threat model |

## Running the Project

### Backend

From the repository root:

```powershell
c:/Users/ksham/OneDrive/Desktop/major/.venv-1/Scripts/python.exe -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

Or run the VS Code task named `Start backend`.

### Frontend

```powershell
npm --prefix frontend run dev
```

Open:

```text
http://localhost:4173
```

The backend API and interactive API documentation are available at:

```text
http://localhost:8000
http://localhost:8000/docs
```

## Supabase Integration

The optional Supabase frontend client is configured with:

```env
VITE_SUPABASE_URL=https://qneyazbyhpscccdqluda.supabase.co
VITE_SUPABASE_ANON_KEY=your-publishable-key
```

These values belong in `frontend/.env.local`, which is ignored by Git.

The existing migration at `supabase/migrations/20260924_create_threat_alerts.sql` describes `public.threat_alerts`, including:

- threat type and level
- confidence
- location
- detection and creation timestamps
- message and recipient
- pending/processing/sent/failed email state
- indexes
- a HIGH/CRITICAL queue trigger
- Row Level Security

The FastAPI backend inserts eligible rows with `email_status='pending'` using server-only `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. It atomically claims each row, sends through Gmail SMTP configured in the root `.env`, and updates the Supabase status only after SMTP accepts the message. Resend Edge Function files remain as a separate option for deployments with a verified sender domain; do not enable their database webhook when SMTP delivery is active.

### Configure Gmail SMTP alerts

Set these server-side values in the root `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=kshamakanth5@gmail.com
SMTP_USERNAME=kshamakanth5@gmail.com
SMTP_PASSWORD=your-google-app-password
ALERT_EMAIL=kshamakanth5@gmail.com
```

Password reset uses the same SMTP settings. Set `FRONTEND_URL` to the browser-visible app origin so reset links open the right deployment. Optional settings are `PASSWORD_RESET_EXPIRY_MINUTES` (default 15), `PASSWORD_RESET_MAX_REQUESTS` (default 3), and `PASSWORD_RESET_WINDOW_MINUTES` (default 15). Reset tokens are stored as hashes in a dedicated SQLite table; the existing accounts table is unchanged. A successful reset revokes the account's existing login sessions. No separate SQL migration is required; the backend creates the two reset tables when it initializes the existing SQLite database.

Use a Google App Password, not the normal Gmail password. Keep the app password out of Git and browser environment files. Restart the backend after changing `.env`. The dashboard's **Send test email** action exercises the same queue and SMTP delivery path. If a Resend Database Webhook was previously created, disable it to prevent duplicate sends.

Apply `supabase/migrations/20260925_add_email_delivery_metadata.sql` to the existing table if `sent_to` and `provider_email_id` are not already present. Gmail SMTP does not provide a Resend provider email ID, so `provider_email_id` stays null for SMTP sends.
## Testing

The repository uses Python's standard-library `unittest` runner:

```powershell
c:/Users/ksham/OneDrive/Desktop/major/.venv-1/Scripts/python.exe -m unittest discover -s tests -v
```

The Python tests cover:

- activity classification
- jumping, dancing, and running heuristics
- stationary-person risk behavior
- overlapping-person deduplication
- heatmap risk bands
- physical-altercation temporal classification
- strict confidence threshold behavior (including exactly 75%)
- cooldown protection

The legacy Resend Edge Function tests cover threshold boundaries and provider failure:

```powershell
deno test -A supabase/functions/send-threat-alert/index_test.ts
```

Build the frontend with:

```powershell
npm --prefix frontend run build
```

## End-to-End Verification

1. Start the backend and frontend.
2. Register an operator account with a real email address.
3. Configure Supabase server credentials, Gmail SMTP with a Google App Password, and `ALERT_EMAIL` in the root `.env`. Disable any old Resend Database Webhook.
4. Log in and allow browser camera permission.
5. Confirm the backend session token is issued by login.
6. Open the normal camera view to inspect detections without tint.
7. Switch to heatmap camera view.
8. Observe blue/green colors for normal and low-risk activity.
9. Observe yellow/orange/red/purple heat as risk increases.
10. Generate an event below 75% and verify no email is sent.
11. Generate an event at or above 75% and verify a pending Supabase row is inserted and claimed.
12. Verify Gmail SMTP accepts one message, `email_status` becomes `sent`, `sent_to` is populated, and `sent_at` is set.
13. Replay the same event and verify cooldown/claiming prevents duplicate email.
14. Simulate an SMTP failure and verify `email_status='failed'` while the camera pipeline continues.

## Current Limitations and Research Considerations

- The available activity data is synthetic still imagery plus a small set of broad motion clips; it does not support reliable 28-class activity training. The live fallback reports generic movement and coarse posture.
- The camera submits an analyzed frame every two seconds (0.5 fps), which is too sparse for reliable gait-cycle distinctions such as walking versus jogging versus running.
- `train_activity.py` requires genuine action labels and subject-separated clips; it does not treat Normal/Wall crossing or weapon labels as activity ground truth.
- The existing threat model metadata contains two image classes: `Normal Class` and `Wall crossing`, with 120 recorded training samples in the current metadata.
- Heatmap risk is a decision-support estimate and should be calibrated against a labeled validation set.
- Physical-altercation detection is a conservative multi-person rapid-motion heuristic, not proof that a fight occurred.
- Camera framing, lighting, occlusion, pose confidence, detector confidence, and missing modalities affect output quality.
- Threat email requires valid Gmail SMTP credentials, `ALERT_EMAIL`, and Supabase server credentials in the backend environment. SMTP failures are persisted as failed status.
- The backend still uses SQLite for operator authentication and its local audit log; Supabase stores the email alert queue.

## Responsible Deployment

ATAS should be evaluated with representative, consented data and measured using precision, recall, false-positive rate, false-negative rate, calibration, and latency. Operators should be able to review the source frame and evidence before taking action. The system should not be used as the sole basis for law-enforcement, employment, housing, education, or other high-impact decisions.
