import { detectionAgent } from './detectionAgent.js'
import { poseExtractionAgent } from './poseExtractionAgent.js'
import { activityRecognitionAgent } from './activityRecognitionAgent.js'
import { intentPredictionAgent } from './intentPredictionAgent.js'
import { threatAnalysisAgent } from './threatAnalysisAgent.js'
import { explainabilityAgent } from './explainabilityAgent.js'
import { alertLearningAgent } from './alertLearningAgent.js'

export const defaultAgents = [
  detectionAgent,
  poseExtractionAgent,
  activityRecognitionAgent,
  intentPredictionAgent,
  threatAnalysisAgent,
  explainabilityAgent,
  alertLearningAgent,
]

export const buildAgentMesh = (threatLevel = 'MEDIUM', action = 'perimeter scan') => {
  const threat = String(threatLevel).toUpperCase()
  const isHigh = ['HIGH', 'CRITICAL', 'CRIME'].includes(threat)
  const isMedium = ['MEDIUM', 'SUSPICIOUS'].includes(threat)
  const level = isHigh ? 'HIGH' : isMedium ? 'MEDIUM' : 'LOW'

  return defaultAgents.map((agent) => {
    const payload = { ...agent.payload, frame_action: action, threat_level: level }
    if (agent.id === 'D1') payload.flags = ['person scan', action]
    if (agent.id === 'D3') payload.activity = isHigh ? 'high-risk movement' : isMedium ? 'suspicious movement' : 'routine movement'
    if (agent.id === 'D5') payload.threatLevel = level
    if (agent.id === 'D7') payload.action = isHigh ? 'alarm_triggered' : isMedium ? 'notify_operator' : 'monitor_only'
    return { ...agent, status: level, payload }
  })
}
