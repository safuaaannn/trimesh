import { Box, Text, Flex } from '@radix-ui/themes'
import { useEffect, useMemo, useState } from 'react'
import * as Slider from '@radix-ui/react-slider'
import { translations } from '../i18n'
import './JointControl.css'

export default function JointControl({ jointName, displayName, rotation, onChange, language }) {
  const t = translations[language]
  const degrees = useMemo(() => ({
    x: rotation.x * (180 / Math.PI),
    y: rotation.y * (180 / Math.PI),
    z: rotation.z * (180 / Math.PI)
  }), [rotation])

  const [inputValues, setInputValues] = useState({
    x: degrees.x.toFixed(0),
    y: degrees.y.toFixed(0),
    z: degrees.z.toFixed(0)
  })

  useEffect(() => {
    setInputValues({
      x: degrees.x.toFixed(0),
      y: degrees.y.toFixed(0),
      z: degrees.z.toFixed(0)
    })
  }, [degrees])

  const handleChange = (axis, values) => {
    const value = (values[0] / 100) * Math.PI * 2 - Math.PI
    onChange(jointName, axis, value)
  }

  const handleInputChange = (axis, value) => {
    setInputValues(prev => ({ ...prev, [axis]: value }))
  }

  const commitInput = (axis) => {
    const raw = inputValues[axis]
    const parsed = parseFloat(raw)
    if (!Number.isFinite(parsed)) {
      setInputValues(prev => ({ ...prev, [axis]: degrees[axis].toFixed(0) }))
      return
    }
    const clamped = Math.max(-180, Math.min(180, parsed))
    onChange(jointName, axis, (clamped * Math.PI) / 180)
  }

  const getSliderValue = (radians) => {
    return ((radians + Math.PI) / (Math.PI * 2)) * 100
  }

  return (
    <Box className="joint-control">
      <Text size="2" weight="medium" mb="2" style={{ display: 'block' }}>
        {displayName || jointName}
      </Text>

      <Box className="axis-control">
        <Flex align="center" gap="2" mb="1">
          <Text size="1" className="axis-label axis-x">X</Text>
          <Slider.Root
            className="slider-root"
            value={[getSliderValue(rotation.x)]}
            onValueChange={(values) => handleChange('x', values)}
            min={0}
            max={100}
            step={0.5}
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" />
          </Slider.Root>
          <input
            className="axis-input"
            type="number"
            step="1"
            min="-180"
            max="180"
            value={inputValues.x}
            onChange={(e) => handleInputChange('x', e.target.value)}
            onBlur={() => commitInput('x')}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
          />
        </Flex>

        <Flex align="center" gap="2" mb="1">
          <Text size="1" className="axis-label axis-y">Y</Text>
          <Slider.Root
            className="slider-root"
            value={[getSliderValue(rotation.y)]}
            onValueChange={(values) => handleChange('y', values)}
            min={0}
            max={100}
            step={0.5}
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" />
          </Slider.Root>
          <input
            className="axis-input"
            type="number"
            step="1"
            min="-180"
            max="180"
            value={inputValues.y}
            onChange={(e) => handleInputChange('y', e.target.value)}
            onBlur={() => commitInput('y')}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
          />
        </Flex>

        <Flex align="center" gap="2">
          <Text size="1" className="axis-label axis-z">Z</Text>
          <Slider.Root
            className="slider-root"
            value={[getSliderValue(rotation.z)]}
            onValueChange={(values) => handleChange('z', values)}
            min={0}
            max={100}
            step={0.5}
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" />
          </Slider.Root>
          <input
            className="axis-input"
            type="number"
            step="1"
            min="-180"
            max="180"
            value={inputValues.z}
            onChange={(e) => handleInputChange('z', e.target.value)}
            onBlur={() => commitInput('z')}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
          />
        </Flex>
      </Box>
    </Box>
  )
}
