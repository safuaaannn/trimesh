import { useCallback, useRef, useState, useEffect } from 'react'
import { Box, Button, Text, Flex, Heading, Card, IconButton, Dialog } from '@radix-ui/themes'
import { Upload, AlertCircle, Languages, RefreshCcw, Trash2, Camera, X, Circle } from 'lucide-react'
import { translations } from '../i18n'
import './UploadPanel.css'

export default function UploadPanel({
  onUpload,
  loading,
  error,
  language,
  onToggleLanguage,
  imagePreviewUrl,
  onClearSession,
  onReprocess,
  hasCachedResult,
  restoringSession,
  sessionStatus
}) {
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(null)
  const countdownTimerRef = useRef(null)
  const [showCamera, setShowCamera] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [countdown, setCountdown] = useState(0)
  const [isCapturing, setIsCapturing] = useState(false)
  const t = translations[language]
  const statusLabel = sessionStatus ? t.processingStatuses?.[sessionStatus] || sessionStatus : null

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      onUpload(file)
    }
  }, [onUpload])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      onUpload(file)
    }
  }, [onUpload])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
  }, [])

  const startCamera = useCallback(async () => {
    try {
      setCameraError(null)
      setCountdown(0)
      setIsCapturing(false)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user', // 前置摄像头
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      streamRef.current = stream
      setShowCamera(true)
      // 使用 setTimeout 确保对话框已渲染
      setTimeout(() => {
        if (videoRef.current && streamRef.current) {
          videoRef.current.srcObject = streamRef.current
          videoRef.current.muted = true
          videoRef.current.play().catch(err => {
            console.error('Video play error:', err)
          })
        }
      }, 200)
    } catch (err) {
      console.error('Camera access error:', err)
      setCameraError(t.cameraAccessError)
      setShowCamera(true) // 显示对话框以展示错误信息
    }
  }, [t])

  const stopCamera = useCallback(() => {
    // 清除倒计时
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current)
      countdownTimerRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setShowCamera(false)
    setCameraError(null)
    setCountdown(0)
    setIsCapturing(false)
  }, [])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || isCapturing) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    // 检查视频是否已加载
    if (video.readyState < 2) {
      console.warn('Video not ready')
      return
    }

    // 设置画布尺寸与视频一致
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480

    // 绘制当前视频帧到画布
    context.drawImage(video, 0, 0)

    // 将画布转换为 Blob，然后转换为 File
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera-${Date.now()}.jpg`, {
          type: 'image/jpeg',
          lastModified: Date.now()
        })
        stopCamera()
        onUpload(file)
      }
    }, 'image/jpeg', 0.95)
  }, [onUpload, stopCamera, isCapturing])

  const startCountdown = useCallback(() => {
    if (isCapturing || countdown > 0) return
    
    setIsCapturing(true)
    setCountdown(5)
    
    countdownTimerRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          if (countdownTimerRef.current) {
            clearInterval(countdownTimerRef.current)
            countdownTimerRef.current = null
          }
          setIsCapturing(false)
          // 倒计时结束后自动拍摄
          setTimeout(() => {
            capturePhoto()
          }, 100)
          return 0
        }
        return prev - 1
      })
    }, 1000)
  }, [isCapturing, countdown, capturePhoto])

  // 当对话框打开且视频流准备好时，设置视频源
  useEffect(() => {
    if (showCamera && streamRef.current && videoRef.current && !cameraError) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.muted = true
      videoRef.current.play().catch(err => {
        console.error('Video play error:', err)
      })
    }
  }, [showCamera, cameraError])

  // 清理摄像头资源
  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  return (
    <Box p="4" style={{ borderBottom: '1px solid var(--gray-6)' }}>
      <Flex justify="between" align="center" mb="3">
        <Heading size="5">{t.title}</Heading>
        <IconButton
          size="2"
          variant="soft"
          onClick={onToggleLanguage}
          title={language === 'en' ? '切换到中文' : 'Switch to English'}
        >
          <Languages size={18} />
        </IconButton>
      </Flex>

      {imagePreviewUrl ? (
        <Card>
          <Box position="relative">
            <img
              src={imagePreviewUrl}
              alt="Uploaded preview"
              style={{
                width: '100%',
                height: 'auto',
                maxHeight: '200px',
                objectFit: 'contain',
                borderRadius: '8px',
                display: 'block'
              }}
            />
            <Button
              size="2"
              variant="soft"
              style={{
                position: 'absolute',
                bottom: '8px',
                right: '8px'
              }}
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              <Upload size={16} />
              {language === 'en' ? 'Replace Image' : '替换图片'}
            </Button>
          </Box>
        </Card>
      ) : (
        <Card>
          <div
            className="upload-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={32} className="upload-icon" />
            <Text size="2" color="gray" mt="2">
              {loading ? t.processing : t.uploadZone}
            </Text>
            <Text size="1" color="gray" mt="1">
              {t.uploadFormats}
            </Text>
          </div>
          <Flex gap="2" mt="3" justify="center">
            <Button
              size="2"
              variant="outline"
              onClick={startCamera}
              disabled={loading}
            >
              <Camera size={16} />
              {t.takePhoto}
            </Button>
          </Flex>
        </Card>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {error && (
        <Flex mt="3" gap="2" align="center" style={{ color: 'var(--red-9)' }}>
          <AlertCircle size={16} />
          <Text size="2">{error}</Text>
        </Flex>
      )}

      {statusLabel && (
        <Text
          size="2"
          mt="3"
          style={{
            color: sessionStatus === 'failed'
              ? 'var(--red-10)'
              : sessionStatus === 'completed'
                ? 'var(--green-10)'
                : 'var(--gray-11)'
          }}
        >
          {statusLabel}
        </Text>
      )}

      {restoringSession && (
        <Text size="2" color="gray" mt="3">
          {t.restoringSession}
        </Text>
      )}

      {hasCachedResult && !restoringSession && (
        <Box mt="3">
          <Flex gap="2" wrap="wrap">
            <Button
              size="2"
              variant="solid"
              onClick={onReprocess}
              disabled={loading}
            >
              <RefreshCcw size={16} />
              {t.reprocess}
            </Button>
            <Button
              size="2"
              variant="outline"
              onClick={onClearSession}
              disabled={loading}
            >
              <Trash2 size={16} />
              {t.clearResult}
            </Button>
          </Flex>
          <Text size="1" color="gray" mt="2">
            {t.cachedSessionHint}
          </Text>
        </Box>
      )}

      {loading && (
        <Box mt="3">
          <div className="loading-bar"></div>
          <Text size="1" color="gray" mt="2">{t.detectingPose}</Text>
        </Box>
      )}

      {/* 摄像头拍照对话框 */}
      <Dialog.Root open={showCamera} onOpenChange={(open) => {
        if (!open) {
          stopCamera()
        }
      }}>
        <Dialog.Content style={{ maxWidth: '90vw', width: '640px', padding: '0' }}>
          <Box p="4">
            <Flex justify="between" align="center" mb="3">
              <Dialog.Title>{t.cameraTitle}</Dialog.Title>
              <IconButton
                size="2"
                variant="ghost"
                onClick={stopCamera}
              >
                <X size={18} />
              </IconButton>
            </Flex>

            {cameraError ? (
              <Box p="4" style={{ textAlign: 'center' }}>
                <AlertCircle size={32} style={{ color: 'var(--red-9)', margin: '0 auto 1rem' }} />
                <Text size="3" color="red">{cameraError}</Text>
                <Button
                  size="2"
                  variant="outline"
                  mt="3"
                  onClick={stopCamera}
                >
                  {t.close}
                </Button>
              </Box>
            ) : (
              <>
                <Box
                  style={{
                    position: 'relative',
                    width: '100%',
                    backgroundColor: '#000',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    aspectRatio: '4/3',
                    minHeight: '300px'
                  }}
                >
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      transform: 'scaleX(-1)' // 镜像翻转，更符合自拍习惯
                    }}
                  />
                  <canvas ref={canvasRef} style={{ display: 'none' }} />
                  {countdown > 0 && (
                    <Box
                      style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        zIndex: 10
                      }}
                    >
                      <Text
                        size="9"
                        weight="bold"
                        style={{
                          color: '#fff',
                          textShadow: '0 0 20px rgba(0,0,0,0.8)',
                          fontSize: '120px',
                          lineHeight: 1
                        }}
                      >
                        {countdown}
                      </Text>
                    </Box>
                  )}
                </Box>

                <Flex gap="2" mt="4" justify="center" align="center">
                  <Button
                    size="3"
                    variant="solid"
                    onClick={startCountdown}
                    disabled={isCapturing || countdown > 0}
                    style={{
                      width: '64px',
                      height: '64px',
                      borderRadius: '50%',
                      padding: 0,
                      cursor: (isCapturing || countdown > 0) ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <Circle size={48} fill="currentColor" />
                  </Button>
                </Flex>
                <Text size="1" color="gray" mt="2" style={{ textAlign: 'center' }}>
                  {countdown > 0 
                    ? `${t.countdown}: ${countdown}`
                    : (isCapturing 
                        ? t.capturing
                        : t.cameraHint)
                  }
                </Text>
              </>
            )}
          </Box>
        </Dialog.Content>
      </Dialog.Root>
    </Box>
  )
}
