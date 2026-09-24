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
├── agents/                         JavaScript agent definitions
│   ├── detectionAgent.js
│   ├── poseExtractionAgent.js
│   ├── activityRecognitionAgent.js
│   ├── intentPredictionAgent.js
│   ├── threatAnalysisAgent.js
│   ├── explainabilityAgent.js
│   ├── alertLearningAgent.js
│   └── index.js
├── backend/
│   ├── server.py                   FastAPI entrypoint and live frame route
│   ├── agents.py                   Agent mesh and Alert Agent policy
│   ├── email_service.py            SMTP threat notification service
│   ├── temporal_context.py         Temporal movement and aggression context
│   ├── multimodal_risk.py          Sensor and evidence aggregation
│   ├── anomaly_risk.py             Anomaly risk report
│   ├── facial_expression.py        Facial-expression and next-action helpers
│   ├── train_pipeline.py           Normal/wall-crossing image model pipeline
│   ├── train_activity.py           Activity-training placeholder and data contract
│   ├── model_meta.json             Threat model metadata
│   └── accounts.db                 Local account/session/alert database at runtime
├── frontend/
│   ├── src/App.jsx                 React dashboard, login, camera, heatmap
│   ├── src/lib/supabase.js         Optional Supabase client
│   ├── package.json                Vite frontend dependencies and scripts
│   └── .env.local                  Local Supabase frontend configuration
├── dataset_frames/                 Normal and wall-crossing image data
├── tests/                          Standard-library regression tests
├── supabase/migrations/            Optional Supabase SQL migration
├── requirements.txt                Python dependencies
├── .env.example                    SMTP and risk-policy template
└── .gitignore                      Local secrets, databases, models, and builds
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

### 2. Pose Extraction Agent

Uses `yolo11n-pose.pt` to extract human keypoints. Keypoints include normalized coordinates and confidence values. Pose boxes support activity classification and dashboard skeleton overlays.

### 3. Activity Recognition Agent

The current live activity recognizer uses pose geometry, body proportions, motion score, and temporal stabilization. It supports labels including:

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

The dashboard waits for repeated consistent frames before changing its stable activity label. The repository does not currently contain a trained action-recognition sequence model; therefore these activity labels are pose-geometry heuristics rather than a validated deep action-recognition model.

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
3. Applies the email risk threshold.
4. Requires consecutive qualifying frames.
5. Applies per-user cooldown protection.
6. Generates an alert ID.
7. Calls the email notification service.
8. Persists the alert result in SQLite.
9. Returns email status to the frontend.

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
RISK_THRESHOLD_EMAIL=95
CONFIRMATION_FRAMES=5
ALERT_COOLDOWN_MINUTES=5
```

- Risk below 60% does not activate a risk heatmap.
- Risk from 60% activates the heatmap.
- Risk from 95% is email-eligible after confirmation.
- Five consecutive qualifying frames are required by default.
- A successful alert suppresses repeat emails for five minutes per user.
- Email failures are recorded and do not crash the pipeline.

The Alert Agent does not send one email per frame.

## Email Notifications

Email delivery is implemented in `backend/email_service.py` using SMTP. The sender is configured in the root `.env` file:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=sender@example.com
SMTP_USERNAME=sender@example.com
SMTP_PASSWORD=google-app-password
```

For Gmail, use a Google App Password rather than the normal account password. The recipient is the authenticated logged-in user's account email.

The dashboard also supports a browser notification after a confirmed alert if notification permission is granted. A true background mobile push notification would require a push provider such as Firebase Cloud Messaging and a service worker.

## API Routes

Important backend routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Backend/model/email configuration status |
| POST | `/api/register` | Create a local operator account |
| POST | `/api/login` | Authenticate and issue a session token |
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

The migration at `supabase/migrations/20260924_create_threat_alerts.sql` creates an optional `public.threat_alerts` queue. It includes:

- threat type and level
- confidence
- location
- detection and creation timestamps
- message and recipient
- pending/processing/sent/failed email state
- indexes
- a HIGH/CRITICAL queue trigger
- Row Level Security

The migration does not send email. A backend worker or Edge Function must process `email_status = 'pending'` using a server-only service-role key. The current local application still persists runtime accounts and alerts in SQLite, so Supabase should be treated as an optional queue until the backend is deliberately migrated.

## Testing

The repository uses Python's standard-library `unittest` runner:

```powershell
c:/Users/ksham/OneDrive/Desktop/major/.venv-1/Scripts/python.exe -m unittest discover -s tests -v
```

The tests cover:

- activity classification
- jumping, dancing, and running heuristics
- stationary-person risk behavior
- overlapping-person deduplication
- heatmap risk bands
- physical-altercation temporal classification
- below-threshold alert behavior
- confirmation frames
- authenticated recipient routing
- cooldown protection
- failed email delivery
- per-user alert isolation

Build the frontend with:

```powershell
npm --prefix frontend run build
```

## End-to-End Verification

1. Start the backend and frontend.
2. Register an operator account with a real email address.
3. Configure valid SMTP credentials in the root `.env`.
4. Log in and allow browser camera permission.
5. Confirm the backend session token is issued by login.
6. Open the normal camera view to inspect detections without tint.
7. Switch to heatmap camera view.
8. Observe blue/green colors for normal and low-risk activity.
9. Observe yellow/orange/red/purple heat as risk increases.
10. Keep a confirmed risk at or above 95% for the configured confirmation frames.
11. Verify the Alert Agent returns an alert ID and email status.
12. Verify the authenticated user's email receives the message.
13. Verify repeated frames remain within cooldown and do not create duplicate emails.
14. Verify failed SMTP delivery is stored as failed without stopping the camera pipeline.

## Current Limitations and Research Considerations

- The live activity classifier is currently pose-geometry and motion based. It is not a validated deep action-recognition model for every human activity.
- `train_activity.py` intentionally blocks activity training until labeled activity sequence folders are supplied.
- The existing threat model metadata contains two image classes: `Normal Class` and `Wall crossing`, with 120 recorded training samples in the current metadata.
- Heatmap risk is a decision-support estimate and should be calibrated against a labeled validation set.
- Physical-altercation detection is a conservative multi-person rapid-motion heuristic, not proof that a fight occurred.
- Camera framing, lighting, occlusion, pose confidence, detector confidence, and missing modalities affect output quality.
- SMTP delivery requires valid provider credentials and may be rejected by providers if app-password or account-security settings are incorrect.
- The Supabase migration is prepared, but the current backend runtime remains SQLite until a deliberate server-side migration is completed.

## Responsible Deployment

ATAS should be evaluated with representative, consented data and measured using precision, recall, false-positive rate, false-negative rate, calibration, and latency. Operators should be able to review the source frame and evidence before taking action. The system should not be used as the sole basis for law-enforcement, employment, housing, education, or other high-impact decisions.
