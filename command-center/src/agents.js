export const defaultAgents = [
  {
    id: 'D1',
    name: 'Detection Agent (YOLOv8)',
    status: 'Active',
    color: '#06b6d4',
    payload: {
      model: 'YOLOv8',
      entity: 'Person',
      confidence: 97,
      bbox: [24, 18, 52, 73],
      classes_detected: ['Person', 'Vehicle', 'Firearm', 'Package'],
      flags: ['perimeter scan', 'person presence'],
    },
  },
  {
    id: 'D2',
    name: 'Pose Extraction Agent (MediaPipe)',
    status: 'Streaming',
    color: '#06b6d4',
    payload: {
      model: 'MediaPipe',
      keypoints: 33,
      pose: 'upright',
      orientation: 'north-east',
      posture: 'steady stance',
      body_joints: ['Head', 'Shoulders', 'Elbows', 'Hands', 'Hips', 'Knees', 'Feet'],
    },
  },
  {
    id: 'D3',
    name: 'Activity Recognition Agent',
    status: 'Evaluating',
    color: '#22c55e',
    payload: {
      model: 'Spatio-temporal classifier',
      activity: 'Standing',
      motion: 'low-velocity',
      confidence: 0.92,
      states: ['Standing', 'Running', 'Crawling', 'Loitering'],
    },
  },
  {
    id: 'D4',
    name: 'Intent Prediction Agent',
    status: 'Tracking',
    color: '#f59e0b',
    payload: {
      model: 'Trajectory predictor',
      trajectory: 'Perimeter Approach',
      confidence: 0.79,
      intent: 'Concealment',
      dwell: 2.6,
      targets: ['Perimeter Approach', 'Infiltration', 'Concealment'],
    },
  },
  {
    id: 'D5',
    name: 'Threat Analysis Agent (LLM Engine)',
    status: 'Assessing',
    color: '#ef4444',
    payload: {
      model: 'LLM engine',
      threatLevel: 'HIGH',
      zone: 'Zone A',
      rulesTriggered: ['restricted boundary', 'concealment', 'weapon detection'],
      context_summary: 'Context, zone policies, and intent history were fused to assign the threat level.',
    },
  },
  {
    id: 'D6',
    name: 'Explainability Agent (XAI)',
    status: 'Explaining',
    color: '#06b6d4',
    payload: {
      model: 'XAI narrative engine',
      rationale: 'Object tracking indicates a person approaching a restricted boundary with concealment and hostile intent patterns. The model combines posture, motion, environment, and dwell time.',
      confidence: 0.89,
      reasoning: 'Human-readable justification generated for operator review.',
    },
  },
  {
    id: 'D7',
    name: 'Alert & Learning Agent',
    status: 'Responding',
    color: '#ef4444',
    payload: {
      model: 'Response policy + feedback loop',
      action: 'alarm_triggered',
      modelUpdate: 'retention_window_2m',
      objectFocus: ['Pen', 'Weapon', 'Vehicle'],
      learning: 'State outputs are logged for model refinement and continuous retraining.',
    },
  },
]

export const buildAgentMesh = (threatLevel = 'MEDIUM', action = 'perimeter scan') => {
  const threat = String(threatLevel).toUpperCase()
  const isHigh = ['HIGH', 'CRITICAL', 'CRIME'].includes(threat)
  const isMedium = ['MEDIUM', 'SUSPICIOUS'].includes(threat)

  return defaultAgents.map((agent, index) => {
    const overrides = {
      D1: { status: isHigh ? 'Alert' : 'Scanning', payload: { ...agent.payload, flags: ['perimeter scan', 'person presence', action] } },
      D2: { status: isHigh ? 'Crouched posture' : 'Steady posture', payload: { ...agent.payload, pose: isHigh ? 'crouched' : 'upright' } },
      D3: { status: 'Analyzing motion', payload: { ...agent.payload, activity: isHigh ? 'concealment' : isMedium ? 'loitering' : 'walking' } },
      D4: { status: 'Predicting intent', payload: { ...agent.payload, trajectory: isHigh ? 'restricted approach' : 'perimeter drift', intent: isHigh ? 'Covert intrusion' : isMedium ? 'Suspicious intent' : 'Authorized patrol' } },
      D5: { status: isHigh ? 'High risk' : isMedium ? 'Medium risk' : 'Low risk', payload: { ...agent.payload, threatLevel: isHigh ? 'HIGH' : isMedium ? 'MEDIUM' : 'LOW' } },
      D6: { status: isHigh ? 'Explaining causal chain' : 'Reasoning over scene', payload: { ...agent.payload, confidence: isHigh ? 0.91 : 0.78 } },
      D7: { status: isHigh ? 'Alarm triggered' : isMedium ? 'Operator warning' : 'Monitor only', payload: { ...agent.payload, action: isHigh ? 'alarm_triggered' : isMedium ? 'notify_operator' : 'monitor_only' } },
    }

    const next = overrides[agent.id] || {}
    return { ...agent, ...next }
  })
}
