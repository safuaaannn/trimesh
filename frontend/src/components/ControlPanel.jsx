import { Box, Heading, Text, Button, Flex, ScrollArea, Tabs, Switch, Select } from '@radix-ui/themes'
import * as Slider from '@radix-ui/react-slider'
import { RotateCcw, User, ChevronDown, ChevronUp } from 'lucide-react'
import JointControl from './JointControl'
import { translations, getJointDisplayName } from '../i18n'
import './ControlPanel.css'
import { useCallback, useEffect, useMemo, useState } from 'react'

const BODY_PARTS = {
  upperBody: ['head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
  lowerBody: ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
}

// Finger joint patterns - these might have different indices
const FINGER_PATTERNS = {
  leftHand: [
    'left_thumb',
    'left_forefinger',
    'left_middle_finger',
    'left_ring_finger',
    'left_pinky_finger'
  ],
  rightHand: [
    'right_thumb',
    'right_forefinger',
    'right_middle_finger',
    'right_ring_finger',
    'right_pinky_finger'
  ]
}

export default function ControlPanel({
  rigData,
  selectedPerson,
  onPersonSelect,
  jointRotations,
  onJointRotationChange,
  onResetPose,
  onNormalizeAPose,
  showJoints,
  onToggleJoints,
  language,
  thetaScale,
  onThetaScaleChange,
  onThetaScaleCommit,
  rawPoseText,
  rawPoseError,
  rawPoseLoading,
  rawPoseExpectedLength,
  onRawPoseTextChange,
  onLoadRawPose,
  onApplyRawPose
}) {
  const [showDebug, setShowDebug] = useState(false)
  const [showRawPose, setShowRawPose] = useState(false)
  const [poseEditorJoint, setPoseEditorJoint] = useState(0)
  const [poseEditorValues, setPoseEditorValues] = useState(Array(5).fill(''))

  if (!rigData) return null

  const t = translations[language]
  const currentRig = rigData.rig_data[selectedPerson]
  const animationTargets = currentRig?.animation_targets || {}
  const thetaPercent = Math.round((thetaScale ?? 1) * 100)
  const poseJointNames = currentRig?.skeleton?.joint_names || []

  const stripHeaderTokens = useCallback((tokens) => {
    if (!tokens || tokens.length < 5) return tokens || []
    const headerA = ['1', '2', '3', '4', '5']
    const headerB = ['2', '3', '4', '5', '6']
    const firstFive = tokens.slice(0, 5)
    const isHeaderA = headerA.every((v, i) => firstFive[i] === v)
    const isHeaderB = headerB.every((v, i) => firstFive[i] === v)
    if (isHeaderA || isHeaderB) {
      return tokens.slice(5)
    }
    return tokens
  }, [])

  const rawPoseTokens = useMemo(() => {
    if (!rawPoseText) return []
    const matches = String(rawPoseText).match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g) || []
    return stripHeaderTokens(matches)
  }, [rawPoseText, stripHeaderTokens])
  const expectedLength = Number.isFinite(rawPoseExpectedLength) ? rawPoseExpectedLength : 130
  const expectedJointCount = Math.ceil(expectedLength / 5)
  const rawPoseCount = rawPoseTokens.length
  const rawPoseDiff = rawPoseCount - expectedLength
  const rawPoseStatus = rawPoseDiff === 0 ? 'ok' : rawPoseDiff > 0 ? 'extra' : 'missing'
  const rawPoseMissing = rawPoseDiff < 0 ? Math.abs(rawPoseDiff) : 0
  const rawPoseExtra = rawPoseDiff > 0 ? rawPoseDiff : 0
  const rawPoseDetail = useMemo(() => {
    if (rawPoseStatus === 'missing') {
      const start = rawPoseCount + 1
      return `${t.rawPoseMissing} ${rawPoseMissing} (${t.rawPosePositions} ${start}-${expectedLength})`
    }
    if (rawPoseStatus === 'extra') {
      const extras = rawPoseTokens.slice(expectedLength, expectedLength + Math.min(5, rawPoseExtra)).join(', ')
      return `${t.rawPoseExtra} ${rawPoseExtra} (${t.rawPosePositions} ${expectedLength + 1}-${rawPoseCount}: ${extras})`
    }
    return null
  }, [rawPoseStatus, rawPoseCount, rawPoseMissing, rawPoseExtra, rawPoseTokens, t, expectedLength])

  const templateHint = useMemo(() => {
    const base = expectedLength % 5 === 0 ? t.rawPoseTemplateHint : t.rawPoseTemplateHintSimple
    return base
      .replace('{count}', String(expectedLength))
      .replace('{joints}', String(expectedJointCount))
  }, [t, expectedLength, expectedJointCount])

  const insertZerosLabel = useMemo(() => {
    return t.rawPoseInsertZeros.replace('{count}', String(expectedLength))
  }, [t, expectedLength])

  const poseGroupSize = 5
  const poseJointCount = Math.ceil(Math.max(rawPoseTokens.length, expectedLength) / poseGroupSize)
  const poseJointLabels = useMemo(() => {
    const count = Math.max(expectedJointCount, poseJointCount)
    return Array.from({ length: count }, (_, idx) => {
      const jointName = poseJointNames[idx] || `joint_${idx}`
      return getJointDisplayName(jointName, language)
    })
  }, [poseJointCount, poseJointNames, language, expectedJointCount])
  const poseValueStrings = useMemo(() => {
    const base = rawPoseTokens.slice(0, expectedLength)
    const filled = Array.from({ length: expectedLength }, (_, idx) => base[idx] ?? '0')
    return filled
  }, [rawPoseTokens, expectedLength])
  const poseRows = useMemo(() => {
    if (!rawPoseTokens.length) return []
    const rows = []
    for (let i = 0; i < poseJointCount; i++) {
      const jointName = poseJointNames[i] || `joint_${i}`
      const label = getJointDisplayName(jointName, language)
      const start = i * poseGroupSize
      rows.push({
        label,
        values: rawPoseTokens.slice(start, start + poseGroupSize)
      })
    }
    return rows
  }, [rawPoseTokens, poseJointCount, poseJointNames, language])

  const handlePoseEditorValueChange = (index, value) => {
    setPoseEditorValues(prev => {
      const next = prev.slice()
      next[index] = value
      return next
    })
  }

  const handleApplyPoseEditor = () => {
    const next = poseValueStrings.slice()
    const start = poseEditorJoint * poseGroupSize
    for (let i = 0; i < poseGroupSize; i++) {
      const raw = poseEditorValues[i]
      const parsed = Number.parseFloat(raw)
      next[start + i] = Number.isFinite(parsed) ? String(parsed) : '0'
    }
    onRawPoseTextChange?.(next.join(', '))
  }

  const handleCleanRawPose = () => {
    const text = String(rawPoseText || '')
    const matches = text.match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g) || []
    const cleaned = stripHeaderTokens(matches)
    const trimmed = cleaned.slice(0, expectedLength)
    onRawPoseTextChange?.(trimmed.join(', '))
  }

  const handleInsertZeros = () => {
    const zeros = Array.from({ length: expectedLength }, () => '0')
    onRawPoseTextChange?.(zeros.join(', '))
  }

  useEffect(() => {
    const start = poseEditorJoint * poseGroupSize
    const values = poseValueStrings.slice(start, start + poseGroupSize)
    setPoseEditorValues(values)
  }, [poseEditorJoint, poseValueStrings])

  console.log('[Control] All animation_targets:', animationTargets)

  // Find finger joints that exist in animation_targets
  const availableFingerJoints = {}

  // Check for any finger-related joints
  const allJointNames = Object.keys(animationTargets)
  console.log('[Control] All joint names:', allJointNames)

  const fingerKeywords = ['thumb', 'finger', 'forefinger', 'middle_finger', 'ring_finger', 'pinky']
  const fingerJoints = allJointNames.filter(name =>
    fingerKeywords.some(keyword => name.toLowerCase().includes(keyword))
  )

  console.log('[Control] Found finger joints:', fingerJoints)

  // Group by left/right hand
  availableFingerJoints.leftHand = fingerJoints.filter(name => name.includes('left'))
  availableFingerJoints.rightHand = fingerJoints.filter(name => name.includes('right'))

  const hasFingers = availableFingerJoints.leftHand.length > 0 || availableFingerJoints.rightHand.length > 0

  console.log('[Control] Available finger joints:', availableFingerJoints)
  console.log('[Control] Has fingers:', hasFingers)

  return (
    <Box className="control-panel">
      <ScrollArea className="control-panel-scroll" type="auto">
        <Box p="4">
          <Flex justify="between" align="center" mb="3">
            <Heading size="4">{t.poseControls}</Heading>
            <Flex gap="2">
              <Button size="1" variant="soft" onClick={onNormalizeAPose}>
                {t.normalizeAPose}
              </Button>
              <Button size="1" variant="soft" onClick={onResetPose}>
                <RotateCcw size={14} />
                {t.reset}
              </Button>
            </Flex>
          </Flex>

          {/* Display Options */}
          <Box mb="4" p="3" style={{ background: 'var(--gray-3)', borderRadius: 'var(--radius-2)' }}>
            <Text size="2" weight="medium" mb="2" style={{ display: 'block' }}>
              {t.displayOptions}
            </Text>
            <Flex direction="column" gap="2">
              <Flex justify="between" align="center">
                <Text size="2">{t.showJoints}</Text>
                <Switch checked={showJoints} onCheckedChange={onToggleJoints} />
              </Flex>
            </Flex>
          </Box>

          {/* MHR Theta Control */}
          <Box mb="4" p="3" style={{ background: 'var(--gray-3)', borderRadius: 'var(--radius-2)' }}>
            <Text size="2" weight="medium" mb="2" style={{ display: 'block' }}>
              {t.thetaControl}
            </Text>
            <Flex align="center" gap="2">
              <Text size="1" className="axis-label">θ</Text>
              <Slider.Root
                className="slider-root"
                value={[thetaPercent]}
                onValueChange={(values) => onThetaScaleChange?.(values[0] / 100)}
                onValueCommit={(values) => onThetaScaleCommit?.(values[0] / 100)}
                min={0}
                max={200}
                step={1}
              >
                <Slider.Track className="slider-track">
                  <Slider.Range className="slider-range" />
                </Slider.Track>
                <Slider.Thumb className="slider-thumb" />
              </Slider.Root>
              <Text size="1" className="value-display">
                {(thetaPercent / 100).toFixed(2)}x
              </Text>
            </Flex>
          </Box>

          {/* Advanced Raw Pose Editor */}
          <Box mb="4" p="3" style={{ background: 'var(--gray-3)', borderRadius: 'var(--radius-2)' }}>
            <Flex justify="between" align="center" mb="2">
              <Text size="2" weight="medium">
                {t.rawPoseTitle}
              </Text>
              <Button
                size="1"
                variant="ghost"
                onClick={() => setShowRawPose(prev => !prev)}
              >
                {showRawPose ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </Button>
            </Flex>
            {showRawPose && (
              <>
                <Text size="1" color="gray" mb="2" style={{ display: 'block' }}>
                  {t.rawPoseHint}
                </Text>
                <textarea
                  className="raw-pose-textarea"
                  value={rawPoseText || ''}
                  onChange={(e) => onRawPoseTextChange?.(e.target.value)}
                  placeholder={t.rawPoseHint}
                />
                <Text size="1" color="gray" mt="2" style={{ display: 'block' }}>
                  {templateHint}
                </Text>
                <Text size="1" color="gray" mt="1" style={{ display: 'block' }}>
                  {t.rawPoseAutoFixHint}
                </Text>
                <Text
                  size="1"
                  className={`raw-pose-count raw-pose-count-${rawPoseStatus}`}
                  mt="2"
                  style={{ display: 'block' }}
                >
                  {t.rawPoseCount}: {rawPoseCount}/{expectedLength}
                  {rawPoseMissing > 0 && ` · ${t.rawPoseMissing} ${rawPoseMissing}`}
                  {rawPoseExtra > 0 && ` · ${t.rawPoseExtra} ${rawPoseExtra}`}
                </Text>
                {rawPoseDetail && (
                  <Text size="1" className="raw-pose-detail" mt="1" style={{ display: 'block' }}>
                    {rawPoseDetail}
                  </Text>
                )}
                {poseRows.length > 0 && (
                  <Box className="raw-pose-grid" mt="2">
                    {poseRows.map((row, idx) => (
                      <div key={`${row.label}-${idx}`} className="raw-pose-row">
                        <span className="raw-pose-label">{row.label}</span>
                        <div className="raw-pose-values">
                          {Array.from({ length: poseGroupSize }).map((_, valueIdx) => (
                            <span key={valueIdx} className="raw-pose-value">
                              {row.values[valueIdx] ?? ''}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </Box>
                )}
                <Box className="raw-pose-editor" mt="3">
                  <Text size="1" weight="medium" mb="2" style={{ display: 'block' }}>
                    {t.rawPoseEditorTitle}
                  </Text>
                  <div className="raw-pose-editor-row">
                    <label className="raw-pose-editor-label">{t.rawPoseSelectJoint}</label>
                    <select
                      className="raw-pose-editor-select"
                      value={String(poseEditorJoint)}
                      onChange={(e) => setPoseEditorJoint(Number(e.target.value))}
                    >
                      {poseJointLabels.map((label, idx) => (
                        <option key={idx} value={String(idx)}>
                          {idx + 1}. {label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="raw-pose-editor-values">
                    {poseEditorValues.map((value, idx) => (
                      <div key={idx} className="raw-pose-editor-cell">
                        <span className="raw-pose-editor-index">{idx + 1}</span>
                        <input
                          className="raw-pose-editor-input"
                          type="text"
                          inputMode="decimal"
                          value={value}
                          onChange={(e) => handlePoseEditorValueChange(idx, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                  <Button
                    size="1"
                    variant="soft"
                    onClick={handleApplyPoseEditor}
                    disabled={rawPoseLoading}
                  >
                    {t.rawPoseUpdateJoint}
                  </Button>
                </Box>
                {rawPoseError && (
                  <Text size="1" className="raw-pose-error" mt="2" style={{ display: 'block' }}>
                    {rawPoseError}
                  </Text>
                )}
                <Flex gap="2" mt="2">
                  <Button
                    size="1"
                    variant="soft"
                    onClick={onLoadRawPose}
                    disabled={rawPoseLoading}
                  >
                    {t.rawPoseLoad}
                  </Button>
                  <Button
                    size="1"
                    variant="soft"
                    onClick={handleInsertZeros}
                    disabled={rawPoseLoading}
                  >
                    {insertZerosLabel}
                  </Button>
                  <Button
                    size="1"
                    variant="soft"
                    onClick={handleCleanRawPose}
                    disabled={rawPoseLoading}
                  >
                    {t.rawPoseClean}
                  </Button>
                  <Button
                    size="1"
                    onClick={onApplyRawPose}
                    disabled={rawPoseLoading}
                  >
                    {t.rawPoseApply}
                  </Button>
                </Flex>
              </>
            )}
          </Box>

          {rigData.num_persons > 1 && (
            <Box mb="4">
              <Flex justify="between" align="center" mb="2">
                <Text size="2" weight="medium">
                  {t.selectPerson}
                </Text>
                <Text size="1" color="gray">
                  {t.person} {selectedPerson + 1}/{rigData.num_persons}
                </Text>
              </Flex>

              {rigData.num_persons >= 6 && (
                <Box mb="2">
                  <Select.Root
                    value={String(selectedPerson)}
                    onValueChange={(value) => onPersonSelect(Number(value))}
                  >
                    <Select.Trigger variant="soft" size="2" style={{ width: '100%' }}>
                      <User size={14} />
                      {t.person} {selectedPerson + 1}
                    </Select.Trigger>
                    <Select.Content>
                      {rigData.rig_data.map((_, idx) => (
                        <Select.Item key={idx} value={String(idx)}>
                          {t.person} {idx + 1}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Box>
              )}

              <Box className="person-selector-row-wrapper">
                <Flex gap="2" wrap="wrap" className="person-selector-row">
                  {rigData.rig_data.map((_, idx) => (
                    <Button
                      key={idx}
                      size="2"
                      variant={selectedPerson === idx ? 'solid' : 'soft'}
                      onClick={() => onPersonSelect(idx)}
                      className="person-selector-chip"
                    >
                      <User size={14} />
                      {t.person} {idx + 1}
                    </Button>
                  ))}
                </Flex>
              </Box>
            </Box>
          )}

          <Tabs.Root defaultValue="upper">
            <Tabs.List>
              <Tabs.Trigger value="upper">{t.upperBody}</Tabs.Trigger>
              <Tabs.Trigger value="lower">{t.lowerBody}</Tabs.Trigger>
              {hasFingers && <Tabs.Trigger value="fingers">{t.fingers}</Tabs.Trigger>}
            </Tabs.List>

            <Box pt="3">
              {/* Upper Body */}
              <Tabs.Content value="upper">
                <Box className="joint-controls-grid">
                  {BODY_PARTS.upperBody.map(jointName => {
                    if (!animationTargets[jointName]) {
                      console.log(`[Control] Joint not found: ${jointName}`)
                      return null
                    }

                    return (
                      <JointControl
                        key={jointName}
                        jointName={jointName}
                        displayName={getJointDisplayName(jointName, language)}
                        rotation={jointRotations[jointName] || { x: 0, y: 0, z: 0 }}
                        onChange={onJointRotationChange}
                        language={language}
                      />
                    )
                  })}
                </Box>
              </Tabs.Content>

              {/* Lower Body */}
              <Tabs.Content value="lower">
                <Box className="joint-controls-grid">
                  {BODY_PARTS.lowerBody.map(jointName => {
                    if (!animationTargets[jointName]) return null

                    return (
                      <JointControl
                        key={jointName}
                        jointName={jointName}
                        displayName={getJointDisplayName(jointName, language)}
                        rotation={jointRotations[jointName] || { x: 0, y: 0, z: 0 }}
                        onChange={onJointRotationChange}
                        language={language}
                      />
                    )
                  })}
                </Box>
              </Tabs.Content>

              {/* Fingers */}
              {hasFingers && (
                <Tabs.Content value="fingers">
                  <Box className="joint-controls-grid">
                    {/* Left Hand */}
                    {availableFingerJoints.leftHand.length > 0 && (
                      <>
                        <Text size="3" weight="bold" mb="2" style={{ display: 'block' }}>
                          {t.leftHand} ({availableFingerJoints.leftHand.length})
                        </Text>
                        {availableFingerJoints.leftHand.map(jointName => (
                          <JointControl
                            key={jointName}
                            jointName={jointName}
                            displayName={getJointDisplayName(jointName, language)}
                            rotation={jointRotations[jointName] || { x: 0, y: 0, z: 0 }}
                            onChange={onJointRotationChange}
                            language={language}
                          />
                        ))}
                      </>
                    )}

                    {/* Right Hand */}
                    {availableFingerJoints.rightHand.length > 0 && (
                      <>
                        <Text size="3" weight="bold" mb="2" mt="3" style={{ display: 'block' }}>
                          {t.rightHand} ({availableFingerJoints.rightHand.length})
                        </Text>
                        {availableFingerJoints.rightHand.map(jointName => (
                          <JointControl
                            key={jointName}
                            jointName={jointName}
                            displayName={getJointDisplayName(jointName, language)}
                            rotation={jointRotations[jointName] || { x: 0, y: 0, z: 0 }}
                            onChange={onJointRotationChange}
                            language={language}
                          />
                        ))}
                      </>
                    )}
                  </Box>
                </Tabs.Content>
              )}
            </Box>
          </Tabs.Root>

          {/* Debug: Show all available joints - Collapsible */}
          {process.env.NODE_ENV === 'development' && (
            <Box mt="4">
              <Button
                size="1"
                variant="ghost"
                onClick={() => setShowDebug(!showDebug)}
                style={{ width: '100%', justifyContent: 'space-between' }}
              >
                <Text size="1" weight="medium">
                  Debug Info ({Object.keys(animationTargets).length} joints)
                </Text>
                {showDebug ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </Button>
              {showDebug && (
                <Box mt="2" p="3" style={{ background: 'var(--gray-3)', borderRadius: 'var(--radius-2)' }}>
                  <Text size="1" color="gray" weight="bold" mb="2" style={{ display: 'block' }}>
                    Available joints:
                  </Text>
                  <Text size="1" color="gray" style={{ fontFamily: 'monospace', wordBreak: 'break-all', maxHeight: '200px', overflowY: 'auto', display: 'block' }}>
                    {Object.keys(animationTargets).join(', ')}
                  </Text>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </ScrollArea>
    </Box>
  )
}
