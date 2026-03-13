import { useState, useCallback, useEffect, useRef, createContext } from 'react'
import { Flex, Box, Button } from '@radix-ui/themes'
import UploadPanel from './components/UploadPanel'
import ViewerPanel from './components/ViewerPanel'
import ControlPanel from './components/ControlPanel'
import MeasurementOverlay from './components/MeasurementOverlay'
import IntroAnimation from './components/IntroAnimation'
import { translations } from './i18n'
import './App.css'

export const LanguageContext = createContext('en')

const SESSION_CACHE_KEY = 'sam3d-body-session-v1'

const A_POSE_ROTATIONS = {
  left_shoulder: { x: 0, y: 0, z: 0.541052 },
  right_shoulder: { x: 0, y: 0, z: -0.558505 },
  left_elbow: { x: 0, y: 0, z: 0 },
  right_elbow: { x: 0, y: 0, z: 0 }
}

const getAPoseRotations = () => {
  const rotations = {}
  Object.entries(A_POSE_ROTATIONS).forEach(([joint, rot]) => {
    rotations[joint] = { x: rot.x || 0, y: rot.y || 0, z: rot.z || 0 }
  })
  return rotations
}

const readFileAsDataUrl = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

const dataUrlToFile = (dataUrl, filename, fallbackType = 'image/png') => {
  const arr = dataUrl.split(',')
  const mimeMatch = arr[0]?.match(/:(.*?);/)
  const mime = mimeMatch ? mimeMatch[1] : fallbackType
  const bstr = atob(arr[1])
  const len = bstr.length
  const u8arr = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    u8arr[i] = bstr.charCodeAt(i)
  }
  return new File([u8arr], filename, { type: mime })
}

function App() {
  const [rigData, setRigData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedPerson, setSelectedPerson] = useState(0)
  // Store joint rotations per person: { personIndex: { jointName: { x, y, z } } }
  const [jointRotationsByPerson, setJointRotationsByPerson] = useState({})
  const [language, setLanguage] = useState('zh')
  const [showJoints, setShowJoints] = useState(true)
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null)
  const [sessionMeta, setSessionMeta] = useState(null)
  const [cachedImageInfo, setCachedImageInfo] = useState(null)
  const [restoringSession, setRestoringSession] = useState(true)
  const [measurementsByPerson, setMeasurementsByPerson] = useState({})
  const [targetHeightInputs, setTargetHeightInputs] = useState({})
  const [measurementLoading, setMeasurementLoading] = useState(false)
  const [measurementError, setMeasurementError] = useState(null)
  const [isMeasurementOverlayOpen, setIsMeasurementOverlayOpen] = useState(false)
  const [showIntro, setShowIntro] = useState(true)
  const [thetaScaleByPerson, setThetaScaleByPerson] = useState({})
  const [rawPoseTextByPerson, setRawPoseTextByPerson] = useState({})
  const [rawPoseError, setRawPoseError] = useState(null)
  const [rawPoseLoading, setRawPoseLoading] = useState(false)
  const [rawPoseBaselineByPerson, setRawPoseBaselineByPerson] = useState({})
  const [rawPoseLengthByPerson, setRawPoseLengthByPerson] = useState({})
  const pollAttemptRef = useRef(0)
  const viewerRef = useRef(null)

  const persistSessionState = useCallback((payload) => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(payload))
  }, [])

  const clearSessionCache = useCallback(() => {
    if (typeof window === 'undefined') return
    window.localStorage.removeItem(SESSION_CACHE_KEY)
  }, [])

  const handleImageUpload = useCallback(async (file) => {
    setLoading(true)
    setError(null)
    setRigData(null)
    setSessionMeta(null)
    setMeasurementsByPerson({})
    setTargetHeightInputs({})
    setMeasurementLoading(false)
    setMeasurementError(null)
    setThetaScaleByPerson({})
    setRawPoseTextByPerson({})
    setRawPoseError(null)
    setRawPoseBaselineByPerson({})
    setRawPoseLengthByPerson({})

    try {
      const imageDataUrl = await readFileAsDataUrl(file)
      setUploadedImageUrl(imageDataUrl)
      setCachedImageInfo({
        dataUrl: imageDataUrl,
        name: file.name,
        type: file.type || 'image/png',
        size: file.size,
        lastModified: file.lastModified
      })

      const formData = new FormData()
      formData.append('image', file)

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      })

      if (!response.ok && response.status !== 202) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to submit image')
      }

      const data = await response.json()
      if (!data.session_id) {
        throw new Error('Missing session id from server')
      }

      setSelectedPerson(0)
      setJointRotationsByPerson({})
      setSessionMeta({
        sessionId: data.session_id,
        status: data.status || 'queued',
        numPersons: data.num_persons || 0
      })
      pollAttemptRef.current = 0
      setRestoringSession(false)
    } catch (err) {
      setError(err.message)
      console.error('Upload error:', err)
      setLoading(false)
      setRestoringSession(false)
    }
  }, [])

  const handleJointRotationChange = useCallback((jointName, axis, value) => {
    setJointRotationsByPerson(prev => ({
      ...prev,
      [selectedPerson]: {
        ...(prev[selectedPerson] || {}),
        [jointName]: {
          ...(prev[selectedPerson]?.[jointName] || { x: 0, y: 0, z: 0 }),
          [axis]: value
        }
      }
    }))
  }, [selectedPerson])

  const handleResetPose = useCallback(() => {
    setJointRotationsByPerson(prev => ({
      ...prev,
      [selectedPerson]: {}
    }))
  }, [selectedPerson])

  // Get current person's joint rotations & measurements
  const currentJointRotations = jointRotationsByPerson[selectedPerson] || {}
  const currentThetaScale = thetaScaleByPerson[selectedPerson] ?? 1
  const currentMeasurement = measurementsByPerson[selectedPerson]
  const targetHeightValue = targetHeightInputs[selectedPerson] ??
    (currentMeasurement?.target_height_cm ? Number(currentMeasurement.target_height_cm).toFixed(1) : '')

  const toggleLanguage = useCallback(() => {
    setLanguage(prev => prev === 'en' ? 'zh' : 'en')
  }, [])

  const handleClearSession = useCallback(() => {
    clearSessionCache()
    setRigData(null)
    setSessionMeta(null)
    setSelectedPerson(0)
    setJointRotationsByPerson({})
    setMeasurementsByPerson({})
    setTargetHeightInputs({})
    setUploadedImageUrl(null)
    setCachedImageInfo(null)
    setError(null)
    setLoading(false)
    setMeasurementLoading(false)
    setMeasurementError(null)
    setIsMeasurementOverlayOpen(false)
    setRestoringSession(false)
    setThetaScaleByPerson({})
    setRawPoseTextByPerson({})
    setRawPoseError(null)
    pollAttemptRef.current = 0
  }, [clearSessionCache])

  const handleReprocess = useCallback(() => {
    if (!cachedImageInfo?.dataUrl || loading) return
    const file = dataUrlToFile(
      cachedImageInfo.dataUrl,
      cachedImageInfo.name || 'cached-upload.png',
      cachedImageInfo.type || 'image/png'
    )
    handleImageUpload(file)
  }, [cachedImageInfo, handleImageUpload, loading])

  const sendMeasurements = useCallback(async (personIndex, targetHeightCm, bakedVertices = null) => {
    if (!sessionMeta?.sessionId) return
    setMeasurementLoading(true)
    setMeasurementError(null)

    try {
      const res = await fetch('/api/measurements', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionMeta.sessionId,
          person_index: personIndex,
          target_height_cm: targetHeightCm,
          baked_vertices: bakedVertices
        })
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to compute measurements')
      }

      const data = await res.json()
      setMeasurementsByPerson(prev => ({
        ...prev,
        [personIndex]: {
          ...data,
          fetchedAt: Date.now()
        }
      }))
      const normalizedHeight = typeof data.target_height_cm === 'number'
        ? data.target_height_cm
        : Number.parseFloat(data.target_height_cm ?? '')
      setTargetHeightInputs(prev => ({
        ...prev,
        [personIndex]: Number.isFinite(normalizedHeight) ? normalizedHeight.toFixed(1) : ''
      }))
    } catch (err) {
      console.error('Measurement error:', err)
      setMeasurementError(err.message)
    } finally {
      setMeasurementLoading(false)
    }
  }, [sessionMeta?.sessionId])

  const fetchMeasurements = useCallback(async (personIndex, targetHeightCm) => {
    if (!sessionMeta?.sessionId) return

    // Check if user has applied slider rotations
    const rotations = jointRotationsByPerson[personIndex]
    const hasCustomPose = rotations && Object.values(rotations).some(
      r => r.x !== 0 || r.y !== 0 || r.z !== 0
    )

    let bakedVertices = null
    if (hasCustomPose && viewerRef.current?.bakeSkinnedVertices) {
      bakedVertices = viewerRef.current.bakeSkinnedVertices(personIndex)
    }

    await sendMeasurements(personIndex, targetHeightCm, bakedVertices)
  }, [sessionMeta?.sessionId, jointRotationsByPerson, sendMeasurements])

  const handleMeasurementHeightChange = useCallback((personIndex, value) => {
    setTargetHeightInputs(prev => ({
      ...prev,
      [personIndex]: value
    }))
    setMeasurementError(null)
  }, [])

  const parseTargetHeight = useCallback((value) => {
    const trimmed = (value ?? '').trim()
    if (!trimmed) {
      setMeasurementError(language === 'zh' ? '请输入目标身高' : 'Please enter a target height')
      return null
    }

    const parsed = parseFloat(trimmed)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setMeasurementError(language === 'zh' ? '目标身高必须为正数' : 'Target height must be a positive number')
      return null
    }
    return parsed
  }, [language])

  const handleMeasurementApply = useCallback((personIndex, value) => {
    const parsed = parseTargetHeight(value)
    if (parsed == null) return
    fetchMeasurements(personIndex, parsed)
  }, [fetchMeasurements, parseTargetHeight])

  const handleNormalizeToAPose = useCallback((personIndex) => {
    const aPose = getAPoseRotations()
    setJointRotationsByPerson(prev => ({
      ...prev,
      [personIndex]: aPose
    }))
  }, [])

  const handleMeasureInAPose = useCallback(async (personIndex, targetHeightCm) => {
    if (!viewerRef.current?.bakeSkinnedVertices) {
      setMeasurementError(language === 'zh' ? '无法读取网格数据' : 'Unable to read mesh data')
      return
    }

    const previous = jointRotationsByPerson[personIndex] || {}
    const aPose = getAPoseRotations()
    setJointRotationsByPerson(prev => ({
      ...prev,
      [personIndex]: aPose
    }))

    await new Promise((resolve) => requestAnimationFrame(resolve))

    const baked = viewerRef.current.bakeSkinnedVertices(personIndex)
    setJointRotationsByPerson(prev => ({
      ...prev,
      [personIndex]: previous
    }))

    if (!baked) {
      setMeasurementError(language === 'zh' ? '无法生成烘焙网格' : 'Failed to bake mesh')
      return
    }

    await sendMeasurements(personIndex, targetHeightCm, baked)
  }, [jointRotationsByPerson, language, sendMeasurements])

  const handleMeasureInAPoseApply = useCallback((personIndex, value) => {
    const parsed = parseTargetHeight(value)
    if (parsed == null) return
    handleMeasureInAPose(personIndex, parsed)
  }, [handleMeasureInAPose, parseTargetHeight])

  const handleThetaScaleChange = useCallback((personIndex, value) => {
    setThetaScaleByPerson(prev => ({
      ...prev,
      [personIndex]: value
    }))
  }, [])

  const handleThetaScaleCommit = useCallback(async (personIndex, value) => {
    if (!sessionMeta?.sessionId || !rigData) return

    try {
      const res = await fetch('/api/mhr/theta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionMeta.sessionId,
          person_index: personIndex,
          theta_scale: value
        })
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to update theta mesh')
      }

      const data = await res.json()
      if (!data?.rig_data) return

      if (viewerRef.current?.applyRigUpdate) {
        viewerRef.current.applyRigUpdate(personIndex, data.rig_data)
      }

      setRigData(prev => {
        if (!prev?.rig_data?.[personIndex]) return prev
        const nextRig = prev.rig_data.slice()
        const existing = nextRig[personIndex]
        const updated = data.rig_data
        nextRig[personIndex] = {
          ...existing,
          ...updated,
          mesh: {
            ...existing.mesh,
            ...updated.mesh
          },
          skeleton: {
            ...existing.skeleton,
            ...updated.skeleton
          },
          metadata: {
            ...existing.metadata,
            ...updated.metadata
          },
          keypoints: updated.keypoints || existing.keypoints
        }
        return { ...prev, rig_data: nextRig }
      })
    } catch (err) {
      console.error('Theta update error:', err)
    }
  }, [rigData, sessionMeta?.sessionId])

  const handleRawPoseTextChange = useCallback((personIndex, value) => {
    setRawPoseTextByPerson(prev => ({
      ...prev,
      [personIndex]: value
    }))
    setRawPoseError(null)
  }, [])

  const parseRawPose = (text) => {
    const raw = String(text || '')
    const matches = raw.match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g) || []
    let tokens = matches
    if (tokens.length >= 5) {
      const headerA = ['1', '2', '3', '4', '5']
      const headerB = ['2', '3', '4', '5', '6']
      const firstFive = tokens.slice(0, 5)
      const isHeaderA = headerA.every((v, i) => firstFive[i] === v)
      const isHeaderB = headerB.every((v, i) => firstFive[i] === v)
      if (isHeaderA || isHeaderB) {
        tokens = tokens.slice(5)
      }
    }
    const values = tokens.map(v => Number.parseFloat(v))
    if (values.some(v => !Number.isFinite(v))) {
      return { values: null, error: 'Pose contains non-numeric values.' }
    }
    return { values, error: null }
  }

  const handleLoadRawPose = useCallback(async (personIndex) => {
    if (!sessionMeta?.sessionId) return
    setRawPoseLoading(true)
    setRawPoseError(null)

    try {
      const res = await fetch(`/api/mhr/pose?session_id=${sessionMeta.sessionId}&person_index=${personIndex}`)
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to load pose')
      }
      const data = await res.json()
      const poseText = Array.isArray(data.body_pose_params)
        ? data.body_pose_params.join(', ')
        : ''
      setRawPoseTextByPerson(prev => ({
        ...prev,
        [personIndex]: poseText
      }))
      if (Array.isArray(data.body_pose_params)) {
        setRawPoseBaselineByPerson(prev => ({
          ...prev,
          [personIndex]: data.body_pose_params
        }))
        const lengthValue = Number.isFinite(data.length)
          ? data.length
          : data.body_pose_params.length
        setRawPoseLengthByPerson(prev => ({
          ...prev,
          [personIndex]: lengthValue
        }))
      }
    } catch (err) {
      setRawPoseError(err.message)
    } finally {
      setRawPoseLoading(false)
    }
  }, [sessionMeta?.sessionId])

  const getCurrentPoseBaseline = useCallback(async (personIndex) => {
    const cached = rawPoseBaselineByPerson[personIndex]
    const expectedLength = rawPoseLengthByPerson[personIndex]
    if (Array.isArray(cached) && (!expectedLength || cached.length === expectedLength)) {
      return cached
    }
    if (!sessionMeta?.sessionId) return null
    try {
      const res = await fetch(`/api/mhr/pose?session_id=${sessionMeta.sessionId}&person_index=${personIndex}`)
      if (!res.ok) return null
      const data = await res.json()
      if (Array.isArray(data.body_pose_params)) {
        setRawPoseBaselineByPerson(prev => ({
          ...prev,
          [personIndex]: data.body_pose_params
        }))
        const lengthValue = Number.isFinite(data.length)
          ? data.length
          : data.body_pose_params.length
        setRawPoseLengthByPerson(prev => ({
          ...prev,
          [personIndex]: lengthValue
        }))
        return data.body_pose_params
      }
      return null
    } catch {
      return null
    }
  }, [rawPoseBaselineByPerson, rawPoseLengthByPerson, sessionMeta?.sessionId])

  const handleApplyRawPose = useCallback(async (personIndex, text) => {
    if (!sessionMeta?.sessionId || !rigData) return

    const { values, error } = parseRawPose(text || '')
    if (error) {
      setRawPoseError(error)
      return
    }

    let expectedLength = rawPoseLengthByPerson[personIndex]
    if (!expectedLength) {
      const baseline = await getCurrentPoseBaseline(personIndex)
      if (Array.isArray(baseline)) {
        expectedLength = baseline.length
      }
    }
    if (!expectedLength && Array.isArray(values)) {
      expectedLength = values.length
    }
    if (!expectedLength) expectedLength = 130

    let normalizedValues = values
    if (normalizedValues && normalizedValues.length > expectedLength) {
      normalizedValues = normalizedValues.slice(0, expectedLength)
    } else if (normalizedValues && normalizedValues.length < expectedLength) {
      const baseline = await getCurrentPoseBaseline(personIndex)
      if (!baseline || baseline.length !== expectedLength) {
        setRawPoseError(`Pose must have exactly ${expectedLength} values.`)
        return
      }
      const filled = baseline.slice()
      for (let i = 0; i < normalizedValues.length; i++) {
        filled[i] = normalizedValues[i]
      }
      normalizedValues = filled
    }

    if (!normalizedValues || normalizedValues.length !== expectedLength) {
      setRawPoseError(`Pose must have exactly ${expectedLength} values.`)
      return
    }

    setRawPoseTextByPerson(prev => ({
      ...prev,
      [personIndex]: normalizedValues.join(', ')
    }))
    setRawPoseLengthByPerson(prev => ({
      ...prev,
      [personIndex]: expectedLength
    }))

    setRawPoseLoading(true)
    setRawPoseError(null)

    try {
      const res = await fetch('/api/mhr/pose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionMeta.sessionId,
          person_index: personIndex,
          body_pose_params: normalizedValues
        })
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || 'Failed to apply pose')
      }

      const data = await res.json()
      if (!data?.rig_data) return

      if (viewerRef.current?.applyRigUpdate) {
        viewerRef.current.applyRigUpdate(personIndex, data.rig_data)
      }

      setRawPoseBaselineByPerson(prev => ({
        ...prev,
        [personIndex]: normalizedValues
      }))

      setRigData(prev => {
        if (!prev?.rig_data?.[personIndex]) return prev
        const nextRig = prev.rig_data.slice()
        const existing = nextRig[personIndex]
        const updated = data.rig_data
        nextRig[personIndex] = {
          ...existing,
          ...updated,
          mesh: {
            ...existing.mesh,
            ...updated.mesh
          },
          skeleton: {
            ...existing.skeleton,
            ...updated.skeleton
          },
          metadata: {
            ...existing.metadata,
            ...updated.metadata
          },
          keypoints: updated.keypoints || existing.keypoints
        }
        return { ...prev, rig_data: nextRig }
      })
    } catch (err) {
      setRawPoseError(err.message)
    } finally {
      setRawPoseLoading(false)
    }
  }, [rigData, sessionMeta?.sessionId, getCurrentPoseBaseline, rawPoseLengthByPerson])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const cachedRaw = window.localStorage.getItem(SESSION_CACHE_KEY)
    if (!cachedRaw) {
      setRestoringSession(false)
      return
    }

    try {
      const cached = JSON.parse(cachedRaw)
      if (cached.language) {
        setLanguage(cached.language)
      }
      if (typeof cached.showJoints === 'boolean') {
        setShowJoints(cached.showJoints)
      }
      if (typeof cached.selectedPerson === 'number') {
        setSelectedPerson(cached.selectedPerson)
      }
      if (cached.jointRotationsByPerson) {
        setJointRotationsByPerson(cached.jointRotationsByPerson)
      }
      if (cached.uploadedImage?.dataUrl) {
        setUploadedImageUrl(cached.uploadedImage.dataUrl)
        setCachedImageInfo(cached.uploadedImage)
      }
      if (cached.sessionId) {
        setLoading(true)
        setSessionMeta({
          sessionId: cached.sessionId,
          status: cached.status || 'queued',
          numPersons: cached.numPersons || 0
        })
        return
      }
    } catch (err) {
      console.error('Invalid cached session data', err)
      clearSessionCache()
    }

    setRestoringSession(false)
  }, [clearSessionCache])

  useEffect(() => {
    pollAttemptRef.current = 0
  }, [sessionMeta?.sessionId])

  useEffect(() => {
    const sessionId = sessionMeta?.sessionId
    const status = sessionMeta?.status

    if (!sessionId) return
    if (status === 'failed') {
      setLoading(false)
      setRestoringSession(false)
      return
    }
    if (status === 'completed' && rigData?.rig_data?.length) {
      setLoading(false)
      setRestoringSession(false)
      return
    }

    let cancelled = false
    let timeoutId

    const poll = async () => {
      try {
        const res = await fetch(`/api/sessions/${sessionId}`)
        if (!res.ok) {
          if (res.status === 404) {
            clearSessionCache()
            if (!cancelled) {
              setError('Session expired or not found')
              setLoading(false)
              setRestoringSession(false)
              setSessionMeta(null)
            }
            return
          }
          throw new Error('Failed to fetch session status')
        }

        const data = await res.json()
        if (cancelled) {
          return
        }

        setSessionMeta(prev => ({
          sessionId: data.session_id,
          status: data.status,
          numPersons: data.num_persons ?? prev?.numPersons ?? 0
        }))

        if (data.status === 'completed' && data.rig_data) {
          setRigData({
            success: true,
            session_id: data.session_id,
            num_persons: data.num_persons,
            rig_data: data.rig_data
          })
          setSelectedPerson(0)
          setJointRotationsByPerson({})
          setLoading(false)
          setRestoringSession(false)
          pollAttemptRef.current = 0
          return
        }

        if (data.status === 'failed') {
          setError(data.error || 'Processing failed')
          setLoading(false)
          setRestoringSession(false)
          pollAttemptRef.current = 0
          return
        }
      } catch (pollErr) {
        if (!cancelled) {
          console.error('Session polling failed', pollErr)
        }
      }

      if (!cancelled) {
        const baseDelay = status === 'processing' ? 3000 : 4000
        const multiplier = Math.pow(1.5, pollAttemptRef.current)
        const delay = Math.min(baseDelay * multiplier, 15000)
        pollAttemptRef.current += 1
        timeoutId = setTimeout(poll, delay)
      }
    }

    poll()

    return () => {
      cancelled = true
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [sessionMeta?.sessionId, sessionMeta?.status, rigData, clearSessionCache])

  useEffect(() => {
    if (!sessionMeta) return
    persistSessionState({
      sessionId: sessionMeta.sessionId,
      status: sessionMeta.status,
      numPersons: sessionMeta.numPersons,
      selectedPerson,
      jointRotationsByPerson,
      language,
      showJoints,
      uploadedImage: cachedImageInfo || null
    })
  }, [
    sessionMeta,
    cachedImageInfo,
    selectedPerson,
    jointRotationsByPerson,
    language,
    showJoints,
    persistSessionState
  ])

  const handleToggleMeasurementOverlay = useCallback((nextState) => {
    setIsMeasurementOverlayOpen(prev => typeof nextState === 'boolean' ? nextState : !prev)
  }, [])

  const handleMeasurementExport = useCallback(() => {
    const payload = measurementsByPerson[selectedPerson]
    if (!payload) {
      setMeasurementError(language === 'zh' ? '请先生成测量数据' : 'Generate measurements first')
      return
    }
    const t = translations[language]
    const rows = [['key', 'label', 'value', 'unit']]
    Object.entries(payload.measurements || {}).forEach(([key, value]) => {
      const meta = payload.schema?.[key]
      const unitLabel = meta?.unit === 'deg'
        ? t.measurementUnits.deg
        : t.measurementUnits.cm
      const label = t.measurementLabels[key] || key
      rows.push([
        key,
        label,
        typeof value === 'number' ? value : '',
        unitLabel
      ])
    })

    const csvContent = rows.map(row => row.map(field => {
      if (typeof field === 'number') return field
      const text = String(field ?? '')
      if (text.includes(',') || text.includes('"')) {
        return `"${text.replace(/"/g, '""')}"`
      }
      return text
    }).join(',')).join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `person-${selectedPerson + 1}-measurements.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  }, [measurementsByPerson, selectedPerson, language])

  useEffect(() => {
    if (!rigData) {
      setIsMeasurementOverlayOpen(false)
    }
  }, [rigData])

  useEffect(() => {
    if (!sessionMeta?.sessionId || !rigData) return
    if (rawPoseLengthByPerson[selectedPerson]) return

    let active = true
    const fetchLength = async () => {
      try {
        const res = await fetch(`/api/mhr/pose?session_id=${sessionMeta.sessionId}&person_index=${selectedPerson}`)
        if (!res.ok) return
        const data = await res.json()
        if (!Array.isArray(data.body_pose_params)) return
        const lengthValue = Number.isFinite(data.length)
          ? data.length
          : data.body_pose_params.length
        if (!active) return
        setRawPoseLengthByPerson(prev => ({
          ...prev,
          [selectedPerson]: lengthValue
        }))
      } catch {
        // Ignore fetch errors; will be retried on demand.
      }
    }

    fetchLength()
    return () => {
      active = false
    }
  }, [sessionMeta?.sessionId, rigData, selectedPerson, rawPoseLengthByPerson])

  const t = translations[language]
  const currentYear = new Date().getFullYear()

  const handleIntroComplete = useCallback(() => {
    setShowIntro(false)
  }, [])

  return (
    <LanguageContext.Provider value={language}>
      {showIntro && <IntroAnimation onComplete={handleIntroComplete} />}
      <div className="app-shell">
        <Flex className="app-container" gap="0">
          <Box className="left-panel">
            <UploadPanel
              onUpload={handleImageUpload}
              loading={loading}
              error={error}
              language={language}
              onToggleLanguage={toggleLanguage}
              imagePreviewUrl={uploadedImageUrl}
              onClearSession={handleClearSession}
              onReprocess={handleReprocess}
              hasCachedResult={Boolean(sessionMeta?.status === 'completed' && rigData)}
              restoringSession={restoringSession}
              sessionStatus={sessionMeta?.status}
            />

            {rigData && (
              <ControlPanel
                rigData={rigData}
                selectedPerson={selectedPerson}
                onPersonSelect={setSelectedPerson}
                jointRotations={currentJointRotations}
                onJointRotationChange={handleJointRotationChange}
                onResetPose={handleResetPose}
                onNormalizeAPose={() => handleNormalizeToAPose(selectedPerson)}
                showJoints={showJoints}
                onToggleJoints={setShowJoints}
                language={language}
                thetaScale={currentThetaScale}
                onThetaScaleChange={(value) => handleThetaScaleChange(selectedPerson, value)}
                onThetaScaleCommit={(value) => handleThetaScaleCommit(selectedPerson, value)}
                rawPoseText={rawPoseTextByPerson[selectedPerson]}
                rawPoseError={rawPoseError}
                rawPoseLoading={rawPoseLoading}
                rawPoseExpectedLength={rawPoseLengthByPerson[selectedPerson]}
                onRawPoseTextChange={(value) => handleRawPoseTextChange(selectedPerson, value)}
                onLoadRawPose={() => handleLoadRawPose(selectedPerson)}
                onApplyRawPose={() => handleApplyRawPose(selectedPerson, rawPoseTextByPerson[selectedPerson])}
              />
            )}
          </Box>

          <Box className="right-panel">
            <div className="viewer-stage">
              <ViewerPanel
                ref={viewerRef}
                allRigData={rigData?.rig_data}
                selectedPerson={selectedPerson}
                onPersonSelect={setSelectedPerson}
                jointRotations={currentJointRotations}
                jointRotationsByPerson={jointRotationsByPerson}
                showJoints={showJoints}
                language={language}
                measurementData={currentMeasurement}
              />

              {rigData && (
                <div className="viewer-toolbar">
                  <Button
                    size="2"
                    variant={isMeasurementOverlayOpen ? 'solid' : 'surface'}
                    onClick={() => handleToggleMeasurementOverlay()}
                  >
                    {t.measurementPanel.openButton}
                  </Button>
                </div>
              )}

              <MeasurementOverlay
                language={language}
                selectedPerson={selectedPerson}
                visible={isMeasurementOverlayOpen}
                onClose={() => handleToggleMeasurementOverlay(false)}
                measurementData={currentMeasurement}
                measurementError={measurementError}
                measurementLoading={measurementLoading}
                targetHeightValue={targetHeightValue}
                onTargetHeightChange={(value) => handleMeasurementHeightChange(selectedPerson, value)}
                onApply={(value) => handleMeasurementApply(selectedPerson, value)}
                onApplyAPose={(value) => handleMeasureInAPoseApply(selectedPerson, value)}
                onExport={handleMeasurementExport}
              />
            </div>
          </Box>
        </Flex>
        <footer className="site-footer">
          <span>© {currentYear} 小白客 · 版权所有</span>
          <span className="footer-links">
            友情链接：
            <a href="https://www.asmo.top/home" target="_blank" rel="noreferrer">杂货铺</a>
          </span>
        </footer>
      </div>
    </LanguageContext.Provider>
  )
}

export default App
