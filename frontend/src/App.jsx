import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { buildAgentMesh, defaultAgents } from '../../agents/index.js'
import {
  Activity,
  AlertTriangle,
  Bell,
  Camera,
  CheckCircle2,
  Cpu,
  Crosshair,
  Download,
  Eye,
  EyeOff,
  Lock,
  Mail,
  ShieldAlert,
  Upload,
  UserRound,
  Wifi,
  Zap,
} from 'lucide-react'

const scenarioConfig = {
  normal: {
    title: 'Scenario 1 (Normal)',
    threatLevel: 'LOW',
    intent: 'Authorized Patrol',
    alertText: 'Authorized guard walking near perimeter',
    xai: 'Subject remains within authorized patrol path with low-velocity motion and no weapon signatures.',
    overlay: {
      person: { left: '42%', top: '22%', width: '18%', height: '46%' },
      weapon: { left: '58%', top: '39%', width: '7%', height: '11%' },
      points: [
        { left: '48.2%', top: '25%' },
        { left: '49.8%', top: '38%' },
        { left: '46.5%', top: '46%' },
        { left: '52.9%', top: '46%' },
        { left: '45.5%', top: '59%' },
        { left: '53.5%', top: '59%' },
      ],
    },
  },
  suspicious: {
    title: 'Scenario 2 (Suspicious)',
    threatLevel: 'MEDIUM',
    intent: 'Loitering / Infiltration Risk',
    alertText: 'Person loitering near restricted boundary at low illumination',
    xai: 'Loitering behavior near the restricted perimeter increases dwell-time risk and reduces legitimate intent confidence.',
    overlay: {
      person: { left: '36%', top: '20%', width: '22%', height: '52%' },
      weapon: { left: '0%', top: '0%', width: '0%', height: '0%' },
      points: [
        { left: '46.8%', top: '24%' },
        { left: '48.2%', top: '39%' },
        { left: '44.3%', top: '48%' },
        { left: '52.3%', top: '48%' },
        { left: '43.5%', top: '64%' },
        { left: '53.9%', top: '64%' },
      ],
    },
  },
  critical: {
    title: 'Scenario 3 (Critical Threat)',
    threatLevel: 'HIGH',
    intent: 'Concealed Infiltration',
    alertText: 'Crouching individual with detected weapon approaching restricted gate',
    xai: 'Subject detected crawling in restricted Zone A at 02:14 AM. Low-profile posture combined with perimeter trajectory increases infiltration intent confidence to 89%.',
    overlay: {
      person: { left: '39%', top: '27%', width: '24%', height: '58%' },
      weapon: { left: '54%', top: '44%', width: '8%', height: '16%' },
      points: [
        { left: '48.2%', top: '34%' },
        { left: '51.1%', top: '47%' },
        { left: '46.3%', top: '57%' },
        { left: '53.8%', top: '58%' },
        { left: '45.9%', top: '71%' },
        { left: '54.8%', top: '72%' },
      ],
    },
  },
}

const initialLogs = [
  { time: '02:14:11', frame: 462, intent: 'Concealment', level: 'HIGH', action: 'Alarm triggered' },
  { time: '02:13:49', frame: 441, intent: 'Perimeter Approach', level: 'MEDIUM', action: 'Zone A flagged' },
  { time: '02:13:33', frame: 405, intent: 'Loitering', level: 'MEDIUM', action: 'Track persistence' },
  { time: '02:12:51', frame: 366, intent: 'Authorized Patrol', level: 'LOW', action: 'No action' },
]

const historicalAlerts = [
  { id: 'INC-2407', time: '02:14 AM', zone: 'Zone A', intent: 'Infiltration', level: 'HIGH' },
  { id: 'INC-2398', time: '01:49 AM', zone: 'North Gate', intent: 'Loitering', level: 'MEDIUM' },
  { id: 'INC-2391', time: '00:42 AM', zone: 'Perimeter East', intent: 'Unauthorized access', level: 'MEDIUM' },
  { id: 'INC-2385', time: '23:16 PM', zone: 'Service Road', intent: 'Patrol validation', level: 'LOW' },
]

const riskHeatColor = (risk, alpha) => {
  if (risk >= 95) return `rgba(106, 0, 180, ${alpha})`
  if (risk >= 90) return `rgba(235, 30, 35, ${alpha})`
  if (risk >= 75) return `rgba(245, 125, 20, ${alpha})`
  if (risk >= 60) return `rgba(250, 205, 35, ${alpha})`
  if (risk >= 30) return `rgba(35, 190, 90, ${alpha})`
  return `rgba(35, 100, 235, ${alpha})`
}

const poseConnections = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
]

const evaluateLiveFrame = (video) => {
  if (!video || video.readyState < 2) {
    return { status: 'Waiting for camera', action: 'No valid frame', confidence: 0, crimeDetected: false }
  }

  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')
  if (!context) {
    return { status: 'Analysis unavailable', action: 'Browser canvas unavailable', confidence: 0, crimeDetected: false }
  }

  const width = 160
  const height = 90
  canvas.width = width
  canvas.height = height
  context.drawImage(video, 0, 0, width, height)

  const imageData = context.getImageData(0, 0, width, height)
  const pixels = imageData.data
  const previous = video._previousFrame

  let motionPixels = 0
  let lowerBodyPixels = 0
  let midZonePixels = 0

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4
      const red = pixels[index]
      const green = pixels[index + 1]
      const blue = pixels[index + 2]
      const brightness = (red + green + blue) / 3

      if (y > 55 && y < height && x > 35 && x < width - 35 && brightness > 90) {
        lowerBodyPixels += 1
      }

      if (x > 50 && x < 110 && y > 45 && y < height && brightness > 80) {
        midZonePixels += 1
      }

      if (previous) {
        const prevIndex = (y * width + x) * 4
        const prevRed = previous[prevIndex]
        const prevGreen = previous[prevIndex + 1]
        const prevBlue = previous[prevIndex + 2]
        const delta = Math.abs(red - prevRed) + Math.abs(green - prevGreen) + Math.abs(blue - prevBlue)
        if (delta > 90) {
          motionPixels += 1
        }
      }
    }
  }

  video._previousFrame = new Uint8ClampedArray(pixels)

  const personDetected = lowerBodyPixels > 700
  const suspiciousMotion = motionPixels > 1800
  const concealment = midZonePixels > 600 && motionPixels > 1200

  if (personDetected && concealment) {
    return {
      status: 'Crime likely in progress',
      action: 'Crouching near restricted gate / concealment behavior',
      confidence: 92,
      crimeDetected: true,
    }
  }

  if (personDetected && suspiciousMotion) {
    return {
      status: 'Suspicious activity',
      action: 'Loitering or perimeter intrusion',
      confidence: 74,
      crimeDetected: true,
    }
  }

  if (personDetected) {
    return {
      status: 'Normal activity',
      action: 'Person is moving normally',
      confidence: 38,
      crimeDetected: false,
    }
  }

  return {
    status: 'No person detected',
    action: 'No human activity in frame',
    confidence: 10,
    crimeDetected: false,
  }
}

const analyzeLiveFrame = async (video, sessionToken) => {
  if (!video || video.readyState < 2) {
    return { status: 'Waiting for camera', action: 'No valid frame', confidence: 0, crimeDetected: false }
  }

  const canvas = document.createElement('canvas')
  canvas.width = 640
  canvas.height = 360
  canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height)
  const activity = evaluateLiveFrame(video)

  const getMotionMetrics = (detections) => {
    const previous = video._previousDetections || []
    let maximumMovement = 0
    let movingPersons = 0
    let movingObjects = 0

    detections.forEach((detection) => {
      const center = { x: detection.box[0] + detection.box[2] / 2, y: detection.box[1] + detection.box[3] / 2 }
      const candidates = previous.filter((item) => item.isPerson === detection.isPerson && (detection.isPerson || item.label === detection.label))
      const nearest = candidates.reduce((best, item) => {
        const itemCenter = { x: item.box[0] + item.box[2] / 2, y: item.box[1] + item.box[3] / 2 }
        const distance = Math.hypot(center.x - itemCenter.x, center.y - itemCenter.y)
        return distance < best ? distance : best
      }, Number.POSITIVE_INFINITY)
      if (nearest > 3.5 && nearest < Number.POSITIVE_INFINITY) {
        maximumMovement = Math.max(maximumMovement, nearest)
        if (detection.isPerson) movingPersons += 1
        else movingObjects += 1
      }
    })

    video._previousDetections = detections
    return { score: Math.min(maximumMovement / 30, 1), movingPersons, movingObjects }
  }

  try {
    const motionForRequest = video._motionMetrics || { score: 0, movingPersons: 0, movingObjects: 0 }
    const response = await fetch('http://localhost:8000/api/analyze-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${sessionToken}` },
      body: JSON.stringify({
        image: canvas.toDataURL('image/jpeg', 0.7),
        motionScore: motionForRequest.score,
        movingPersons: motionForRequest.movingPersons,
        movingObjects: motionForRequest.movingObjects,
      }),
    })
    const payload = await response.json()
    if (!response.ok || !payload.ok) {
      return { status: activity.status, action: activity.action, confidence: activity.confidence, crimeDetected: activity.crimeDetected, detections: [] }
    }
    const motion = getMotionMetrics(payload.detections || [])
    video._motionMetrics = motion
    const activityHistory = [...(video._activityHistory || []), payload.activity || 'unknown'].slice(-5)
    video._activityHistory = activityHistory
    const activityCounts = activityHistory.reduce((counts, label) => ({ ...counts, [label]: (counts[label] || 0) + 1 }), {})
    const candidateActivity = Object.entries(activityCounts).sort((first, second) => second[1] - first[1])[0]?.[0] || 'unknown'
    const previousActivity = video._stableActivity || 'unknown'
    const candidateFrames = activityHistory.filter((label) => label === candidateActivity).length
    const stableActivity = previousActivity === 'unknown'
      ? candidateFrames >= 3 ? candidateActivity : 'unknown'
      : candidateActivity === previousActivity || candidateFrames >= 3
        ? candidateActivity
        : previousActivity
    video._stableActivity = stableActivity
    const threatHistory = [...(video._threatHistory || []), payload.threatLevel || 'LOW'].slice(-5)
    video._threatHistory = threatHistory
    const previousThreat = video._stableThreatLevel || 'LOW'
    const highFrames = threatHistory.filter((level) => level === 'HIGH' || level === 'CRITICAL').length
    const mediumFrames = threatHistory.filter((level) => level === 'MEDIUM').length
    const personPresent = Boolean(payload.personDetected)
    const threatConfirmed = threatHistory.length >= 3
    const stableThreatLevel = highFrames >= 3
      ? 'HIGH'
      : previousThreat === 'HIGH' && highFrames >= 2
        ? 'HIGH'
        : mediumFrames >= 3
          ? 'MEDIUM'
          : previousThreat === 'MEDIUM' && mediumFrames >= 2
            ? 'MEDIUM'
            : 'LOW'
    video._stableThreatLevel = stableThreatLevel
    const isAssessing = !threatConfirmed && (payload.detected || personPresent || motion.movingPersons || motion.movingObjects)
    const status = isAssessing
      ? `Assessing scene (${threatHistory.length}/3 frames)`
      : payload.detected || personPresent || motion.movingPersons || motion.movingObjects
        ? `${stableThreatLevel} risk assessed`
        : 'Live camera active'
    const strongest = payload.confidence || payload.detections?.reduce((best, detection) => Math.max(best, detection.confidence), 0) || 0
    return {
      status,
      action: payload.detected || payload.aggressionDetected ? payload.message : stableActivity !== 'unknown' ? `Human activity: ${stableActivity}` : motion.movingPersons ? 'Person movement tracked' : motion.movingObjects ? 'Object movement tracked' : personPresent ? 'Monitoring person movement' : 'No person detected in current frame',
      confidence: payload.detected ? strongest : activity.confidence,
      threatLevel: isAssessing ? 'LOW' : stableThreatLevel,
      riskScore: payload.riskScore || payload.confidence || 0,
      riskRegions: payload.riskRegions || [],
      riskThresholdHeatmap: payload.riskThresholdHeatmap ?? 60,
      riskThresholdEmail: payload.riskThresholdEmail ?? 95,
      timestamp: payload.timestamp,
      alert: payload.alert || null,
      threatEvent: payload.threatEvent || null,
      crimeDetected: !isAssessing && stableThreatLevel === 'HIGH',
      riskConfirmed: !isAssessing,
      assessmentFrames: threatHistory.length,
      motion,
      activity: stableActivity,
      rawActivity: payload.rawActivity || stableActivity,
      activitySource: payload.activitySource || 'unknown',
      activityQuality: payload.activityQuality || 'LOW',
      activityClasses: payload.activityClasses || ['standing', 'sitting', 'squatting', 'walking', 'running', 'jumping', 'dancing', 'waving', 'hands_up', 'clapping', 'kicking', 'bending', 'falling', 'lying', 'crawling', 'sleeping', 'fighting'],
      pose: payload.pose || [],
      detections: payload.detections || [],
      personCount: payload.personCount || 0,
      sharpObjects: payload.sharpObjects || [],
      sharpObjectCount: payload.sharpObjectCount || 0,
      agents: payload.agents || [],
      multimodalRisk: payload.multimodalRisk || null,
      temporalContext: payload.temporalContext || null,
      anomalyRisk: payload.anomalyRisk || null,
      facialExpression: payload.facialExpression || null,
      nextActionPrediction: payload.nextActionPrediction || null,
      aggressionDetected: Boolean(payload.aggressionDetected),
      researchSummary: payload.researchSummary || null,
    }
  } catch {
    return { status: activity.status, action: activity.action, confidence: activity.confidence, threatLevel: activity.crimeDetected ? 'HIGH' : 'LOW', crimeDetected: activity.crimeDetected, detections: [] }
  }
}

function App() {
  const videoRef = useRef(null)
  const lastAlertRef = useRef('')
  const alertCooldownRef = useRef('')
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [userName, setUserName] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [email, setEmail] = useState('')
  const [accountEmail, setAccountEmail] = useState('')
  const [sessionToken, setSessionToken] = useState('')
  const [authMode, setAuthMode] = useState('login')
  const [loginError, setLoginError] = useState('')
  const [accountMessage, setAccountMessage] = useState('')
  const [scenario, setScenario] = useState('critical')
  const [selectedAgent, setSelectedAgent] = useState(4)
  const [cameraEnabled, setCameraEnabled] = useState(true)
  const [cameraViewMode, setCameraViewMode] = useState('normal')
  const [cameraError, setCameraError] = useState('')
  const [alarmEnabled, setAlarmEnabled] = useState(true)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [mediaUrl, setMediaUrl] = useState('')
  const [time, setTime] = useState(new Date())
  const [logs, setLogs] = useState(initialLogs)
  const [agentMesh, setAgentMesh] = useState(defaultAgents)
  const [notificationStatus, setNotificationStatus] = useState('')
  const [confirmationStatus, setConfirmationStatus] = useState('')
  const [liveThreatState, setLiveThreatState] = useState({
    status: 'Waiting for camera',
    action: 'No live action detected yet',
    confidence: 0,
    crimeDetected: false,
    threatLevel: 'LOW',
    riskScore: 0,
    riskRegions: [],
    riskThresholdHeatmap: 60,
    riskThresholdEmail: 95,
    activity: 'unknown',
    activityConfidence: 0,
    multimodalRisk: null,
    temporalContext: null,
    anomalyRisk: null,
    facialExpression: null,
    nextActionPrediction: null,
    researchSummary: null,
    alert: null,
  })
  const [liveDetections, setLiveDetections] = useState([])
  const [agentMode, setAgentMode] = useState('MEDIUM')
  const [feedbackStatus, setFeedbackStatus] = useState('')

  const currentScenario = scenarioConfig[scenario]
  const activeAgent = agentMesh[selectedAgent] || defaultAgents[selectedAgent]
  const hasLiveAnalysis = liveThreatState.pose?.length > 0 || liveDetections.length > 0
  const maximumRisk = Math.max(
    Number(liveThreatState.riskScore || 0),
    ...(liveThreatState.riskRegions || []).map((region) => Number(region.riskScore || 0)),
  )
  const heatmapScreenColor = riskHeatColor(maximumRisk, maximumRisk >= 95 ? 0.42 : maximumRisk >= 60 ? 0.24 : 0.12)

  const detectedObjects = useMemo(() => {
    const objects = ['Person', 'Pen', 'Package', 'Vehicle']
    if (currentScenario.threatLevel !== 'LOW') {
      objects.push('Weapon')
    }
    return objects
  }, [currentScenario.threatLevel])

  const threatMessage = useMemo(() => {
    const actionState = liveThreatState.crimeDetected
      ? `CRIME DETECTED: ${liveThreatState.action}.`
      : `Live action: ${liveThreatState.action}.`

    const objectSummary = currentScenario.threatLevel === 'LOW' ? 'No suspicious object pattern detected.' : `Objects tracked: ${detectedObjects.join(', ')}.`

    return `${userName || 'Operator'}: ${actionState} ${objectSummary} ${currentScenario.xai}`
  }, [currentScenario, detectedObjects, liveThreatState, userName])

  useEffect(() => {
    const loadAgents = async () => {
      if (!isLoggedIn) return
      try {
        const response = await fetch(`http://localhost:8000/api/agents?threat=${encodeURIComponent(agentMode)}&action=${encodeURIComponent(liveThreatState.action)}`)
        if (!response.ok) return
        const payload = await response.json()
        if (Array.isArray(payload.agents)) {
          setAgentMesh(payload.agents)
        }
      } catch {
        setAgentMesh(defaultAgents)
      }
    }

    loadAgents()
  }, [agentMode, isLoggedIn, liveThreatState.action])

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!isLoggedIn || !cameraEnabled || uploadedFile) return

    let activeStream = null
    setCameraError('')

    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError(window.isSecureContext
          ? 'Camera access is not available in this browser.'
          : 'Camera access requires HTTPS or localhost. Open http://localhost:4173 on this computer.')
        return
      }
      try {
        let stream
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
          })
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        }
        activeStream = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (error) {
        const errorName = error?.name || 'CameraError'
        const message = errorName === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in the browser address-bar settings, then retry.'
          : errorName === 'NotFoundError'
            ? 'No camera was found. Connect a webcam or use Upload video instead.'
            : errorName === 'NotReadableError'
              ? 'The camera is already in use by another application.'
              : `Camera could not start (${errorName}). Use localhost/HTTPS and check browser permissions.`
        setCameraError(message)
      }
    }

    startCamera()

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop())
      }
      if (videoRef.current) {
        videoRef.current._previousFrame = null
        videoRef.current._previousDetections = []
        videoRef.current._motionMetrics = null
        videoRef.current._activityHistory = []
        videoRef.current._stableActivity = 'unknown'
        videoRef.current._threatHistory = []
        videoRef.current._stableThreatLevel = 'LOW'
      }
    }
  }, [cameraEnabled, isLoggedIn, uploadedFile])

  useEffect(() => {
    if (!isLoggedIn || !cameraEnabled || uploadedFile || !videoRef.current) return

    let active = true
    const analyze = async () => {
      const result = await analyzeLiveFrame(videoRef.current, sessionToken)
      if (!active) return
      setLiveThreatState(result)
      setLiveDetections(result.detections || [])
      if (result.agents?.length) setAgentMesh(result.agents)
      setAgentMode(result.threatLevel === 'HIGH' ? 'HIGH' : result.threatLevel === 'MEDIUM' ? 'SUSPICIOUS' : 'LOW')
    }
    analyze()
    const interval = setInterval(analyze, 2000)

    return () => { active = false; clearInterval(interval) }
  }, [cameraEnabled, isLoggedIn, uploadedFile, sessionToken])

  useEffect(() => {
    const alert = liveThreatState.alert
    if (!alert || alert.emailStatus === 'NOT_ELIGIBLE' || alert.emailStatus === 'AWAITING_CONFIRMATION') return
    setNotificationStatus(`Alert ${alert.alertId || 'pending'}: ${alert.emailStatus.toLowerCase().replaceAll('_', ' ')}`)
    if (['SENT', 'FAILED'].includes(alert.emailStatus) && 'Notification' in window && Notification.permission === 'granted') {
      const notificationKey = alert.alertId || `${liveThreatState.timestamp}:${liveThreatState.riskScore}`
      if (window._lastThreatNotification !== notificationKey) {
        window._lastThreatNotification = notificationKey
        new Notification('ATAS critical threat alert', {
          body: `Risk ${liveThreatState.riskScore}% - ${liveThreatState.action}`,
          tag: notificationKey,
        })
      }
    }
  }, [liveThreatState.alert])

  useEffect(() => {
    if (!isLoggedIn || !('Notification' in window) || Notification.permission !== 'default') return
    Notification.requestPermission().catch(() => {})
  }, [isLoggedIn])

  useEffect(() => {
    if (!videoRef.current) return

    if (uploadedFile) {
      videoRef.current.srcObject = null
      videoRef.current.src = mediaUrl
      videoRef.current.play().catch(() => {})
      return
    }

    if (!cameraEnabled) {
      videoRef.current.srcObject = null
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
  }, [cameraEnabled, uploadedFile, mediaUrl])

  useEffect(() => {
    if (alarmEnabled && currentScenario.threatLevel === 'HIGH') {
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (!AudioCtx) return

      const ctx = new AudioCtx()
      const oscillator = ctx.createOscillator()
      const gain = ctx.createGain()

      oscillator.type = 'sawtooth'
      oscillator.frequency.value = 480
      gain.gain.value = 0.02
      oscillator.connect(gain)
      gain.connect(ctx.destination)
      oscillator.start()
      oscillator.stop(ctx.currentTime + 0.2)
      oscillator.onended = () => ctx.close()
    }
  }, [alarmEnabled, scenario, currentScenario.threatLevel])

  useEffect(() => {
    const nextEvent = {
      time: new Date().toLocaleTimeString('en-GB', { timeZone: 'UTC', hour12: false }),
      frame: Math.floor(Math.random() * 200) + 320,
      intent: currentScenario.intent,
      level: currentScenario.threatLevel,
      action: currentScenario.threatLevel === 'HIGH' ? 'Alarm triggered' : 'Agent mesh stable',
    }

    const timer = setTimeout(() => {
      setLogs((previous) => [nextEvent, ...previous].slice(0, 5))
    }, 1500)

    return () => clearTimeout(timer)
  }, [scenario, currentScenario.intent, currentScenario.threatLevel])

  const threatStyle = useMemo(
    () => ({
      LOW: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40',
      MEDIUM: 'bg-amber-500/15 text-amber-300 border border-amber-500/40',
      HIGH: 'bg-red-500/15 text-red-300 border border-red-500/40',
    }[currentScenario.threatLevel]),
    [currentScenario.threatLevel],
  )

  const handleLogin = async (event) => {
    event.preventDefault()

    if (!email.trim() || password.trim().length < 4) {
      setLoginError('Enter a valid account email and password to enter the command center.')
      return
    }

    try {
      const response = await fetch('http://localhost:8000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      const payload = await response.json()
      if (!response.ok || !payload.ok) {
        setLoginError(payload.message || 'Operator authentication failed.')
        return
      }
      setLoginError('')
      setAccountEmail(payload.email)
      setSessionToken(payload.sessionToken)
      setIsLoggedIn(true)
    } catch {
      setLoginError('Backend unavailable. Start the API on port 8000 and try again.')
    }
  }

  const handleRegister = async (event) => {
    event.preventDefault()
    setLoginError('')
    setAccountMessage('')
    try {
      const response = await fetch('http://localhost:8000/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userName, email, password }),
      })
      const payload = await response.json()
      if (!response.ok || !payload.ok) {
        setLoginError(payload.message || 'Account creation failed.')
        return
      }
      setAccountMessage(payload.message)
      setAuthMode('login')
      setPassword('')
    } catch {
      setLoginError('Backend unavailable. Start the API on port 8000 and try again.')
    }
  }

  const resendConfirmation = async () => {
    setConfirmationStatus('Sending confirmation message...')
    try {
      const response = await fetch('http://localhost:8000/api/resend-confirmation', {
        method: 'POST',
        headers: { Authorization: `Bearer ${sessionToken}` },
      })
      const payload = await response.json()
      if (!response.ok || !payload.ok) throw new Error(payload.message || 'Confirmation resend failed.')
      setConfirmationStatus(payload.message)
    } catch (error) {
      setConfirmationStatus(error.message || 'Confirmation email delivery failed.')
    }
  }

  const handleUpload = (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    const nextUrl = URL.createObjectURL(file)
    setUploadedFile(file)
    setMediaUrl(nextUrl)
    setCameraEnabled(false)
  }

  const exportLogs = () => {
    const payload = JSON.stringify(logs, null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'incident-log.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const submitFeedback = async () => {
    const observedActivity = liveThreatState.activity || liveThreatState.rawActivity || 'standing'
    try {
      const response = await fetch('http://localhost:8000/api/agent-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: accountEmail,
          observedActivity,
          correctedActivity: observedActivity,
          threatLevel: liveThreatState.threatLevel || 'LOW',
        }),
      })
      const payload = await response.json()
      setFeedbackStatus(payload.message || 'Feedback recorded.')
    } catch {
      setFeedbackStatus('Feedback service unavailable.')
    }
  }

  const utcTime = time.toLocaleTimeString('en-GB', { timeZone: 'UTC', hour12: false })
  const istTime = time.toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata', hour12: false })

  if (!isLoggedIn) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black p-6">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md border border-white/15 bg-black p-6 shadow-2xl"
        >
          <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center border border-white/30 text-white">
              <ShieldAlert className="h-6 w-6" />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.28em] text-white/45">Secure access</p>
              <h1 className="text-xl font-semibold text-white">Threat Command Center</h1>
            </div>
          </div>

          <div className="mb-5 border border-white/15 bg-white/[0.04] p-3 text-sm text-white/65">
            {authMode === 'login' ? 'Sign in to monitor the defense grid and review active threat messages.' : 'Create an operator account. A security message will be sent to your email when SMTP is configured.'}
          </div>

          <form onSubmit={authMode === 'login' ? handleLogin : handleRegister} className="space-y-4">
            <label className="block text-sm text-slate-300">
              {authMode === 'login' ? 'Email or operator name' : 'Operator name'}
              <div className="mt-2 flex items-center gap-2 border border-white/15 bg-white/[0.04] px-3 py-2">
                {authMode === 'login' ? <Mail className="h-4 w-4 text-white/45" /> : <UserRound className="h-4 w-4 text-white/45" />}
                <input
                  value={authMode === 'login' ? email : userName}
                  onChange={(event) => authMode === 'login' ? setEmail(event.target.value) : setUserName(event.target.value)}
                  className="w-full bg-transparent text-white outline-none placeholder:text-white/30"
                  placeholder={authMode === 'login' ? 'operator@example.com or operator name' : 'Enter operator name'}
                />
              </div>
            </label>

            {authMode === 'register' && (
              <label className="block text-sm text-white/70">
                Email address
                <div className="mt-2 flex items-center gap-2 border border-white/15 bg-white/[0.04] px-3 py-2">
                  <Mail className="h-4 w-4 text-white/45" />
                  <input
                    value={email}
                    type="email"
                    onChange={(event) => setEmail(event.target.value)}
                    className="w-full bg-transparent text-white outline-none placeholder:text-white/30"
                    placeholder="operator@example.com"
                  />
                </div>
              </label>
            )}

            <label className="block text-sm text-slate-300">
              Password
              <div className="mt-2 flex items-center gap-2 border border-white/15 bg-white/[0.04] px-3 py-2">
                <Lock className="h-4 w-4 text-white/45" />
                <input
                  value={password}
                  type={showPassword ? 'text' : 'password'}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full bg-transparent text-white outline-none placeholder:text-white/30"
                  placeholder="Enter password"
                />
                <button
                  type="button"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  title={showPassword ? 'Hide password' : 'Show password'}
                  onClick={() => setShowPassword((value) => !value)}
                  className="text-white/45 hover:text-white"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </label>

            {loginError && <p className="text-sm text-red-300">{loginError}</p>}
            {accountMessage && <p className="text-sm text-white/70">{accountMessage}</p>}

            <button
              type="submit"
              className="w-full bg-white px-4 py-3 text-sm font-semibold text-black transition hover:bg-white/85"
            >
              {authMode === 'login' ? 'Login to Command Center' : 'Create operator account'}
            </button>
          </form>

          <div className="mt-5 flex items-center justify-between border-t border-white/15 pt-4 text-sm">
            <span className="text-white/45">{authMode === 'login' ? 'New operator?' : 'Already registered?'}</span>
            <button
              type="button"
              onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setLoginError(''); setAccountMessage('') }}
              className="text-white underline underline-offset-4 hover:text-white/70"
            >
              {authMode === 'login' ? 'Create an account' : 'Back to login'}
            </button>
          </div>
        </motion.div>
      </div>
    )
  }

  if (isLoggedIn) {
    return (
      <main className="min-h-screen bg-black p-4 text-white md:p-8">
        <div className="mx-auto max-w-5xl">
          <header className="mb-5 flex items-center justify-between border-b border-white/15 pb-4">
            <div>
              <p className="text-[10px] uppercase tracking-[0.28em] text-white/45">Private camera session</p>
              <h1 className="mt-1 text-xl font-medium">Threat monitor</h1>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={resendConfirmation} className="border border-white/20 px-3 py-2 text-xs text-white/70 hover:bg-white hover:text-black">
                Resend confirmation
              </button>
              <button type="button" onClick={() => { setIsLoggedIn(false); setSessionToken(''); setAccountEmail('') }} className="border border-white/20 px-3 py-2 text-xs text-white/70 hover:bg-white hover:text-black">
                Sign out
              </button>
            </div>
          </header>

          {confirmationStatus && (
            <p className="mb-4 border border-cyan-400/30 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100">
              {confirmationStatus}
            </p>
          )}

          <section className="border border-white/15 bg-black">
            <div className="flex items-center justify-between border-b border-white/15 px-4 py-3 text-xs text-white/55">
              <span>{accountEmail}</span>
              <div className="flex items-center gap-3">
                <div className="flex border border-white/20 text-[10px] uppercase tracking-[0.14em]">
                  <button
                    type="button"
                    onClick={() => setCameraViewMode('normal')}
                    className={`px-2 py-1 ${cameraViewMode === 'normal' ? 'bg-white text-black' : 'text-white/60 hover:text-white'}`}
                  >
                    Normal camera
                  </button>
                  <button
                    type="button"
                    onClick={() => setCameraViewMode('heatmap')}
                    className={`border-l border-white/20 px-2 py-1 ${cameraViewMode === 'heatmap' ? 'bg-red-500 text-white' : 'text-white/60 hover:text-white'}`}
                  >
                    Heatmap camera
                  </button>
                </div>
                <span className="flex items-center gap-2 text-white/75"><span className="status-dot bg-white" /> Webcam · scanning</span>
              </div>
            </div>
            <div className="relative aspect-video bg-black">
              {cameraEnabled && !uploadedFile && !cameraError ? (
                <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-white/45">
                  <span>{cameraError || 'Camera unavailable'}</span>
                  {cameraError && (
                    <button type="button" onClick={() => { setCameraError(''); setCameraEnabled(false); setTimeout(() => setCameraEnabled(true), 0) }} className="border border-white/25 px-3 py-2 text-xs text-white hover:bg-white hover:text-black">
                      Retry camera
                    </button>
                  )}
                </div>
              )}
              <div className="pointer-events-none absolute inset-0">
                {cameraViewMode === 'heatmap' && (
                  <div
                    className="absolute inset-0"
                    style={{
                      background: `radial-gradient(circle at 50% 45%, ${heatmapScreenColor}, transparent 78%), ${heatmapScreenColor}`,
                      mixBlendMode: 'screen',
                    }}
                  />
                )}
                {cameraViewMode === 'heatmap' && liveThreatState.riskRegions?.filter((region) => region.heatmapActive).map((region) => {
                  const intensity = Math.max(0.18, Math.min(0.9, (region.riskScore - liveThreatState.riskThresholdHeatmap) / 40 + 0.18))
                  const heatColor = riskHeatColor(region.riskScore, intensity)
                  return (
                    <div
                      key={`heat-${region.personId}`}
                      className="absolute rounded-full"
                      style={{
                        left: `${region.centerX}%`,
                        top: `${region.centerY}%`,
                        width: `${Math.max(region.width * 1.8, 18)}%`,
                        height: `${Math.max(region.height * 1.2, 18)}%`,
                        transform: 'translate(-50%, -50%)',
                        opacity: intensity,
                        background: `radial-gradient(circle, ${heatColor} 0%, ${riskHeatColor(region.riskScore, intensity * 0.65)} 34%, transparent 78%)`,
                        filter: 'blur(5px)',
                      }}
                    />
                  )
                })}
                {liveThreatState.pose?.map((person, personIndex) => (
                  <svg key={`simple-pose-${personIndex}`} className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Human pose skeleton">
                    {poseConnections.map(([start, end]) => {
                      const first = person.keypoints[start]
                      const second = person.keypoints[end]
                      if (!first || !second) return null
                      return <line key={`${start}-${end}`} x1={first.x} y1={first.y} x2={second.x} y2={second.y} stroke="#22d3ee" strokeWidth="0.7" />
                    })}
                    {person.keypoints.map((point, pointIndex) => (
                      <circle key={pointIndex} cx={point.x} cy={point.y} r="1.1" fill="#f8fafc" stroke="#22d3ee" strokeWidth="0.45" />
                    ))}
                    <text x={person.box[0]} y={Math.max(person.box[1] - 2, 3)} fill="#67e8f9" fontSize="2.8">{(person.activity || 'unknown').toUpperCase()}</text>
                  </svg>
                ))}
                {liveDetections.map((detection, index) => (
                  <div
                    key={`${detection.classId}-${index}`}
                    className="absolute border-2 border-white bg-white/10"
                    style={{
                      left: `${detection.box[0]}%`,
                      top: `${detection.box[1]}%`,
                      width: `${detection.box[2]}%`,
                      height: `${detection.box[3]}%`,
                    }}
                  >
                    <span className="absolute -top-6 left-0 whitespace-nowrap bg-white px-1.5 py-1 text-[10px] font-semibold text-black">
                      {detection.isPerson ? 'Human' : 'Detected object'} {detection.confidence}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <div className="mt-4 flex items-center justify-between border border-white/15 px-4 py-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Current assessment</p>
              <p className="mt-1 text-sm text-white/80">{liveThreatState.status || 'Live camera active'}</p>
              <p className="mt-1 text-xs text-white/45">{liveThreatState.action}</p>
                 <p className="mt-1 text-xs uppercase tracking-[0.16em] text-cyan-300">Activity: {liveThreatState.activity || 'unknown'}</p>
                 <p className="mt-1 text-xs uppercase tracking-[0.16em] text-white/55">
                   Humans: {liveThreatState.personCount || liveThreatState.pose?.length || 0} · Sharp objects: {liveThreatState.sharpObjectCount || 0}
                 </p>
                <p className="mt-1 text-[10px] text-white/40">Raw model inference: {liveThreatState.rawActivity || 'unknown'}</p>
                <p className="mt-1 text-[10px] text-amber-200/75">
                  Heatmap regions: {liveThreatState.riskRegions?.filter((region) => region.heatmapActive).length || 0} active
                </p>
            </div>
            <span className="text-sm text-white/70">{liveThreatState.confidence}%</span>
          </div>
          <div className="mt-3 border border-amber-400/25 bg-amber-500/5 px-4 py-3 text-xs text-amber-100">
            {liveThreatState.riskRegions?.length ? liveThreatState.riskRegions.map((region) => (
              <div key={region.personId} className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-200/10 py-1 last:border-0">
                <span>{region.personId} · {region.label} · {String(region.activity).replaceAll('_', ' ')}</span>
                <span>{region.riskScore}% · {String(region.riskBand).replaceAll('_', ' ')}</span>
              </div>
            )) : 'No person risk regions detected.'}
          </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 border border-white/15 px-4 py-3">
              <span className="text-xs text-white/55">HAR result is model-detected, not user-selected</span>
              <button type="button" onClick={submitFeedback} className="border border-cyan-400/50 px-2 py-1 text-xs text-cyan-200 hover:bg-cyan-400/10">
                Confirm {String(liveThreatState.activity || 'unknown').toUpperCase()}
              </button>
              {feedbackStatus && <span className="text-xs text-white/45">{feedbackStatus}</span>}
            </div>
        </div>
      </main>
    )
  }

  return (
    <div className="min-h-screen bg-[#0d1117] p-4 text-slate-100 md:p-6">
      <div className="mx-auto max-w-[1600px]">
        <header className="command-panel mb-5 rounded-2xl border border-slate-700/70 bg-slate-950/80 px-4 py-3 md:px-6">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-400/40 bg-cyan-500/10 text-cyan-300">
                <ShieldAlert className="h-5 w-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Defense grid</p>
                <h1 className="text-lg font-semibold text-white md:text-xl">
                  Threat Command Center
                </h1>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-300 md:text-sm">
              <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5">
                <span className="status-dot text-emerald-400" style={{ background: '#22c55e' }} />
                Live Feed Active
              </div>
              <div className="flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5">
                <span className="status-dot text-cyan-400" style={{ background: '#06b6d4' }} />
                Agent Mesh Online
              </div>
              <div className="rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1.5">
                Latency: 38 ms
              </div>
              <div className="rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1.5">
                UTC {utcTime}
              </div>
              <div className="rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1.5">
                IST {istTime}
              </div>
            </div>
          </div>
        </header>

        <div className="mb-5 grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
          <div className="command-panel rounded-2xl border border-slate-700/70 bg-slate-950/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Message log</h2>
              </div>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-cyan-200">
                {userName || 'Operator'} logged in
              </span>
            </div>
            <div className="mb-3 flex flex-wrap gap-2">
              <span className={`rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.18em] ${liveThreatState.crimeDetected ? 'border-red-500/40 bg-red-500/10 text-red-300' : 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'}`}>
                Live feed: {liveThreatState.status}
              </span>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-cyan-200">
                Confidence {liveThreatState.confidence}%
              </span>
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-emerald-200">
                Activity {String(liveThreatState.activity || 'unknown').replaceAll('_', ' ')}
              </span>
            </div>
            <p className="text-base text-slate-100">{threatMessage}</p>
            <div className="mt-3 border border-emerald-400/25 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-100">
              <span className="uppercase tracking-[0.2em] text-emerald-300/70">Recognized activity</span>
              <strong className="ml-3 text-sm uppercase">{liveThreatState.activity || 'unknown'}</strong>
              <span className="ml-3 text-emerald-200/70">{liveThreatState.activityConfidence || 0}% confidence</span>
              <span className="ml-3 text-emerald-200/70">{liveThreatState.activityQuality || 'LOW'} quality</span>
            </div>
            {notificationStatus && (
              <p className="mt-3 border border-red-400/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
                {notificationStatus}
              </p>
            )}
            {liveThreatState.alert && (
              <div className="mt-3 grid gap-2 border border-white/15 bg-white/[0.03] p-3 text-xs text-white/70 sm:grid-cols-2">
                <span>Threat status: {liveThreatState.threatLevel}</span>
                <span>Risk score: {liveThreatState.riskScore ?? liveThreatState.confidence}%</span>
                <span>Alert ID: {liveThreatState.alert.alertId || 'Pending confirmation'}</span>
                <span>Email status: {liveThreatState.alert.emailStatus}</span>
                <span className="sm:col-span-2">Timestamp: {liveThreatState.timestamp || 'Not available'}</span>
              </div>
            )}
          </div>

          <div className="command-panel rounded-2xl border border-slate-700/70 bg-slate-950/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Temporal context</h2>
              </div>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-cyan-200">
                {liveThreatState.temporalContext?.movementPattern || 'routine movement'}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Movement</p>
                <p className="mt-2 text-lg font-semibold text-white">
                  {liveThreatState.temporalContext?.movementPattern || 'routine movement'}
                </p>
                <p className="mt-1 text-xs text-slate-300">
                  Zone: {liveThreatState.temporalContext?.restrictedZoneContext || 'standard patrol area'}
                </p>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Temporal anomaly</p>
                <p className="mt-2 text-lg font-semibold text-white">
                  {liveThreatState.temporalContext?.temporalAnomaly || 'normal'}
                </p>
                <p className="mt-1 text-xs text-slate-300">
                  Risk contribution: {liveThreatState.temporalContext?.riskContribution ?? 0.0}
                </p>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Risk evolution</p>
                <p className="mt-2 text-lg font-semibold text-white">
                  {liveThreatState.temporalContext?.riskEvolution?.trend || 'stable'}
                </p>
                <p className="mt-1 text-xs text-slate-300">
                  Confidence {liveThreatState.temporalContext?.riskEvolution?.confidence ?? 0.0}
                </p>
              </div>
            </div>
            <div className="mt-3 rounded-xl border border-slate-700 bg-slate-900/70 p-3 text-xs leading-6 text-slate-200">
              <p className="uppercase tracking-[0.2em] text-slate-400">Context summary</p>
              <p className="mt-2">
                {liveThreatState.temporalContext?.summary || 'Movement and posture are interpreted over time instead of as isolated frame-level events.'}
              </p>
            </div>
          </div>

          <div className="command-panel rounded-2xl border border-slate-700/70 bg-slate-950/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-amber-300" />
                <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Anomaly risk</h2>
              </div>
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-amber-200">
                {liveThreatState.anomalyRisk?.anomalyStatus || 'normal'}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Risk level</p>
                <p className="mt-2 text-lg font-semibold text-white">{liveThreatState.anomalyRisk?.riskLevel || 'LOW'}</p>
                <p className="mt-1 text-xs text-slate-300">Confidence {liveThreatState.anomalyRisk?.confidence ?? 0}%</p>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Sensor reliability</p>
                <p className="mt-2 text-lg font-semibold text-white">{Math.round((liveThreatState.anomalyRisk?.sensorReliability ?? 0.8) * 100)}%</p>
                <p className="mt-1 text-xs text-slate-300">Uncertainty {liveThreatState.anomalyRisk?.uncertainty ?? 0.0}</p>
              </div>
            </div>
            <div className="mt-3 rounded-xl border border-slate-700 bg-slate-900/70 p-3 text-xs leading-6 text-slate-200">
              <p className="uppercase tracking-[0.2em] text-slate-400">Reasoning</p>
              <p className="mt-2">
                {liveThreatState.anomalyRisk?.reasoning || 'The system blends thermal, motion, and contextual evidence and reduces confidence when sensor quality drops.'}
              </p>
            </div>
          </div>

          <div className="command-panel rounded-2xl border border-slate-700/70 bg-slate-950/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Expression and next action</h2>
              </div>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-cyan-200">
                {liveThreatState.facialExpression?.method || 'awaiting frame'}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Recognized emotion</p>
                <p className="mt-2 text-lg font-semibold text-white">{liveThreatState.facialExpression?.emotion || liveThreatState.facialExpression?.expression || 'unknown'}</p>
                <p className="mt-1 text-xs text-slate-300">Confidence {liveThreatState.facialExpression?.emotionConfidence ?? liveThreatState.facialExpression?.confidence ?? 0}%</p>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3 md:col-span-2">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Most likely next action</p>
                <p className="mt-2 text-lg font-semibold text-white">{liveThreatState.nextActionPrediction?.topPrediction || 'awaiting context'}</p>
                <p className="mt-1 text-xs text-slate-300">Prediction confidence {liveThreatState.nextActionPrediction?.confidence ?? 0}%</p>
              </div>
            </div>
            <p className="mt-3 rounded-xl border border-slate-700 bg-slate-900/70 p-3 text-xs leading-6 text-slate-200">
              {liveThreatState.facialExpression?.reason || 'Emotion analysis requires a visible face.'} {liveThreatState.facialExpression?.limitations || ''} {liveThreatState.nextActionPrediction?.decisionPolicy || 'Verify trajectory and context before intervention.'}
            </p>
          </div>

        </div>

        <div className="grid gap-5 xl:grid-cols-[1.7fr_0.9fr]">
          <section className="command-panel rounded-2xl border border-slate-700/70 p-4 md:p-5">
            <div className="mb-4 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
                  <Eye className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-400">Sensor feed</p>
                  <h2 className="text-lg font-semibold text-white">Live Threat Monitor</h2>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {Object.entries(scenarioConfig).map(([key, value]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setScenario(key)}
                    className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                      scenario === key
                        ? 'border-cyan-400 bg-cyan-500/10 text-cyan-200'
                        : 'border-slate-700 bg-slate-900/80 text-slate-300 hover:border-slate-500'
                    }`}
                  >
                    {value.title}
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-4 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => setCameraEnabled((value) => !value)}
                className="inline-flex items-center gap-2 rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-200"
              >
                <Camera className="h-4 w-4" />
                {cameraEnabled ? 'Camera Live' : 'Enable Camera'}
              </button>

              <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-200">
                <Upload className="h-4 w-4" />
                Upload video/image
                <input type="file" accept=".mp4,.jpg,.jpeg,.png" className="hidden" onChange={handleUpload} />
              </label>

              <button
                type="button"
                onClick={() => setAlarmEnabled((value) => !value)}
                className="inline-flex items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
              >
                <Bell className="h-4 w-4" />
                {alarmEnabled ? 'Alarm Enabled' : 'Alarm Muted'}
              </button>
            </div>

            <div className="military-grid relative overflow-hidden rounded-2xl border border-slate-700 bg-[#050a12]">
              <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between bg-slate-950/70 px-4 py-2 text-[10px] uppercase tracking-[0.24em] text-slate-400 backdrop-blur-sm">
                <span>Field of View: Zone A</span>
                <span className="flex items-center gap-2 text-cyan-300">
                  <Wifi className="h-3.5 w-3.5" />
                  Signal Stable
                </span>
              </div>

              <div className="relative aspect-[16/9] w-full overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950">
                {cameraEnabled && !uploadedFile ? (
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_center,_rgba(6,182,212,0.12),_rgba(15,23,42,0.78))]">
                    <div className="relative h-[72%] w-[70%] rounded-[30%] border border-cyan-400/15 bg-slate-900/40 shadow-[0_0_60px_rgba(6,182,212,0.1)]">
                      <div className="absolute left-[24%] top-[20%] h-[44%] w-[20%] rounded-[45%] border border-cyan-400/30 bg-cyan-500/5" />
                      <div className="absolute left-[31%] top-[57%] h-[22%] w-[10%] rounded-full border border-cyan-400/30 bg-cyan-500/5" />
                      <div className="absolute left-[42%] top-[60%] h-[18%] w-[9%] rounded-full border border-cyan-400/30 bg-cyan-500/5" />
                      <div className="absolute left-[24%] top-[62%] h-[18%] w-[9%] rounded-full border border-cyan-400/30 bg-cyan-500/5" />
                    </div>
                  </div>
                )}

                <div className="video-overlay">
                  {liveThreatState.pose?.map((person, personIndex) => (
                    <div key={`pose-${personIndex}`} className="pointer-events-none absolute inset-0">
                      {person.keypoints.map((point, pointIndex) => (
                        <span
                          key={`joint-${personIndex}-${pointIndex}`}
                          className="keypoint"
                          style={{ left: `${point.x}%`, top: `${point.y}%` }}
                          title={`Joint ${pointIndex + 1}: ${point.confidence}%`}
                        />
                      ))}
                      <span
                        className="absolute border border-cyan-300 px-1 py-0.5 text-[9px] uppercase tracking-[0.12em] text-cyan-200"
                        style={{ left: `${person.box[0]}%`, top: `${Math.max(person.box[1] - 4, 1)}%` }}
                      >
                        {person.activity}
                      </span>
                    </div>
                  ))}
                  {!hasLiveAnalysis && (
                    <>
                      <div
                        className={`bounding-box ${currentScenario.threatLevel === 'HIGH' ? 'weapon' : ''}`}
                        style={{
                          left: currentScenario.overlay.person.left,
                          top: currentScenario.overlay.person.top,
                          width: currentScenario.overlay.person.width,
                          height: currentScenario.overlay.person.height,
                        }}
                      >
                        <div className="absolute -top-6 left-0 rounded bg-slate-900/90 px-2 py-1 text-[10px] font-medium text-emerald-300">
                          Demo person overlay
                        </div>
                      </div>
                      {currentScenario.overlay.points.map((point, index) => (
                        <div key={index} className="keypoint" style={{ left: point.left, top: point.top }} />
                      ))}
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto] md:items-center">
              <div>
                <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-[0.26em] text-slate-400">
                  <Crosshair className="h-3.5 w-3.5 text-cyan-300" />
                  Scene analysis
                </div>
                <p className="text-lg font-semibold text-white">{currentScenario.alertText}</p>
              </div>

              <div className={`inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm font-semibold ${threatStyle}`}>
                <AlertTriangle className="h-4 w-4" />
                {currentScenario.threatLevel} Threat
              </div>
            </div>
          </section>

          <aside className="command-panel rounded-2xl border border-slate-700/70 p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-cyan-300" />
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Multi-agent system</h3>
              </div>
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-emerald-300">
                7 agents online
              </span>
            </div>

            <div className="grid gap-3">
              {agentMesh.map((agent, index) => (
                <button
                  key={agent.id}
                  type="button"
                  onClick={() => setSelectedAgent(index)}
                  className={`rounded-2xl border p-3 text-left transition ${
                    selectedAgent === index ? 'border-cyan-500/50 bg-cyan-500/10' : 'border-slate-700 bg-slate-950/70 hover:border-slate-500'
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="flex h-8 w-8 items-center justify-center rounded-lg border text-[10px] font-bold"
                        style={{
                          background: `${agent.color}22`,
                          borderColor: `${agent.color}88`,
                          color: agent.color,
                        }}
                      >
                        {agent.id}
                      </span>
                      <div>
                        <p className="text-[10px] uppercase tracking-[0.18em] text-slate-400">{agent.name}</p>
                        <p className="text-xs font-medium text-slate-200">{agent.status}</p>
                      </div>
                    </div>
                    <span className="rounded-full border border-slate-600 bg-slate-900 px-2 py-1 text-[9px] uppercase tracking-[0.18em] text-slate-300">
                      {agent.payload?.threatLevel || agent.payload?.activity || 'Active'}
                    </span>
                  </div>

                  <pre className="overflow-auto whitespace-pre-wrap text-[10px] leading-5 text-slate-300">
                    {JSON.stringify(agent.payload, null, 2)}
                  </pre>
                </button>
              ))}
            </div>
          </aside>
        </div>

        <div className="mt-5 grid gap-5 xl:grid-cols-[1.05fr_0.95fr_1.1fr]">
          <section className="command-panel rounded-2xl border border-slate-700/70 p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-cyan-300" />
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Threat intelligence logs</h3>
              </div>
              <button
                type="button"
                onClick={exportLogs}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-2 py-1.5 text-xs text-slate-200"
              >
                <Download className="h-3.5 w-3.5" />
                Export JSON
              </button>
            </div>

            <div className="table-scroll max-h-[270px] overflow-auto">
              <table className="w-full border-separate border-spacing-y-2 text-left text-sm">
                <thead className="sticky top-0 bg-slate-950/95 text-[10px] uppercase tracking-[0.2em] text-slate-400">
                  <tr>
                    <th className="px-2 py-2">Time</th>
                    <th className="px-2 py-2">Frame</th>
                    <th className="px-2 py-2">Intent</th>
                    <th className="px-2 py-2">Threat</th>
                    <th className="px-2 py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((entry, index) => (
                    <tr key={`${entry.time}-${index}`} className="rounded-xl bg-slate-900/60 text-slate-200">
                      <td className="rounded-l-xl px-2 py-2">{entry.time}</td>
                      <td className="px-2 py-2">{entry.frame}</td>
                      <td className="px-2 py-2">{entry.intent}</td>
                      <td className="px-2 py-2">
                        <span
                          className={`rounded-full px-2 py-1 text-[10px] font-semibold ${
                            entry.level === 'HIGH'
                              ? 'bg-red-500/15 text-red-300'
                              : entry.level === 'MEDIUM'
                                ? 'bg-amber-500/15 text-amber-300'
                                : 'bg-emerald-500/15 text-emerald-300'
                          }`}
                        >
                          {entry.level}
                        </span>
                      </td>
                      <td className="rounded-r-xl px-2 py-2">{entry.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="command-panel alert-surface rounded-2xl border border-slate-700/70 p-4">
            <div className="mb-3 flex items-center gap-2">
              <Zap className="h-4 w-4 text-cyan-300" />
              <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">XAI explanation</h3>
            </div>

            <div className={`mb-4 inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-semibold ${threatStyle}`}>
              <CheckCircle2 className="h-4 w-4" />
              Active Threat Level: {currentScenario.threatLevel}
            </div>

            <div className="space-y-3 text-sm leading-6 text-slate-200">
              <p>
                Subject detected crawling in restricted Zone A at 02:14 AM. Low-profile posture combined with
                perimeter trajectory increases infiltration intent confidence to 89%.
              </p>
              <ul className="list-disc space-y-2 pl-5 text-slate-300">
                <li>Weapon signature confidence reached 91% in the restricted gate approach corridor.</li>
                <li>Activity profile is consistent with crouching and concealment rather than routine transit.</li>
                <li>Spatial-temporal context triggered rule enforcement for Zone A after dwell time exceeded 2.6 sec.</li>
              </ul>
            </div>
          </section>

          <section className="command-panel rounded-2xl border border-slate-700/70 p-4">
            <div className="mb-3 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-300" />
              <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">Historical alert table</h3>
            </div>

            <div className="table-scroll max-h-[270px] overflow-auto">
              <table className="w-full border-separate border-spacing-y-2 text-left text-sm">
                <thead className="sticky top-0 bg-slate-950/95 text-[10px] uppercase tracking-[0.2em] text-slate-400">
                  <tr>
                    <th className="px-2 py-2">ID</th>
                    <th className="px-2 py-2">Time</th>
                    <th className="px-2 py-2">Zone</th>
                    <th className="px-2 py-2">Intent</th>
                    <th className="px-2 py-2">Level</th>
                  </tr>
                </thead>
                <tbody>
                  {historicalAlerts.map((entry) => (
                    <tr key={entry.id} className="bg-slate-900/60 text-slate-200">
                      <td className="rounded-l-xl px-2 py-2">{entry.id}</td>
                      <td className="px-2 py-2">{entry.time}</td>
                      <td className="px-2 py-2">{entry.zone}</td>
                      <td className="px-2 py-2">{entry.intent}</td>
                      <td className="rounded-r-xl px-2 py-2">
                        <span
                          className={`rounded-full px-2 py-1 text-[10px] font-semibold ${
                            entry.level === 'HIGH'
                              ? 'bg-red-500/15 text-red-300'
                              : entry.level === 'MEDIUM'
                                ? 'bg-amber-500/15 text-amber-300'
                                : 'bg-emerald-500/15 text-emerald-300'
                          }`}
                        >
                          {entry.level}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

export default App
