import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import './IntroAnimation.css'

export default function IntroAnimation({ onComplete }) {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const rendererRef = useRef(null)
  const frameIdRef = useRef(null)
  const particlesRef = useRef(null)
  const startTimeRef = useRef(Date.now())
  const [fadeOut, setFadeOut] = useState(false)

  useEffect(() => {
    if (!mountRef.current) return

    const width = window.innerWidth
    const height = window.innerHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    camera.position.z = 5
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Particles system
    const particleCount = 3000
    const particles = new THREE.BufferGeometry()
    const positions = new Float32Array(particleCount * 3)
    const velocities = new Float32Array(particleCount * 3)
    const colors = new Float32Array(particleCount * 3)
    const sizes = new Float32Array(particleCount)

    const color1 = new THREE.Color(0x00ffff) // Cyan
    const color2 = new THREE.Color(0x0080ff) // Blue
    const color3 = new THREE.Color(0xff00ff) // Magenta

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3
      
      // Random positions in a sphere
      const radius = Math.random() * 12 + 3
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(Math.random() * 2 - 1)
      
      positions[i3] = radius * Math.sin(phi) * Math.cos(theta)
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
      positions[i3 + 2] = radius * Math.cos(phi)
      
      // Random velocities (towards center)
      const speed = Math.random() * 0.03 + 0.01
      velocities[i3] = -positions[i3] * speed * 0.1 + (Math.random() - 0.5) * 0.01
      velocities[i3 + 1] = -positions[i3 + 1] * speed * 0.1 + (Math.random() - 0.5) * 0.01
      velocities[i3 + 2] = -positions[i3 + 2] * speed * 0.1 + (Math.random() - 0.5) * 0.01
      
      // Random colors
      const colorChoice = Math.random()
      let color
      if (colorChoice < 0.33) {
        color = color1
      } else if (colorChoice < 0.66) {
        color = color2
      } else {
        color = color3
      }
      
      colors[i3] = color.r
      colors[i3 + 1] = color.g
      colors[i3 + 2] = color.b
      
      // Random sizes
      sizes[i] = Math.random() * 0.03 + 0.02
    }

    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    particles.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    particles.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    })

    const particleSystem = new THREE.Points(particles, particleMaterial)
    scene.add(particleSystem)
    particlesRef.current = { geometry: particles, velocities, system: particleSystem, count: particleCount }

    // Central geometric shapes
    const shapesGroup = new THREE.Group()
    
    // Torus
    const torusGeometry = new THREE.TorusGeometry(0.8, 0.3, 16, 100)
    const torusMaterial = new THREE.MeshStandardMaterial({
      color: 0x00ffff,
      emissive: 0x004444,
      emissiveIntensity: 0.5,
      metalness: 0.9,
      roughness: 0.1,
      transparent: true,
      opacity: 0.8
    })
    const torus = new THREE.Mesh(torusGeometry, torusMaterial)
    shapesGroup.add(torus)

    // Icosahedron
    const icoGeometry = new THREE.IcosahedronGeometry(0.6, 0)
    const icoMaterial = new THREE.MeshStandardMaterial({
      color: 0xff00ff,
      emissive: 0x440044,
      emissiveIntensity: 0.5,
      metalness: 0.9,
      roughness: 0.1,
      transparent: true,
      opacity: 0.7,
      wireframe: true
    })
    const icosahedron = new THREE.Mesh(icoGeometry, icoMaterial)
    shapesGroup.add(icosahedron)

    // Octahedron
    const octaGeometry = new THREE.OctahedronGeometry(0.4, 0)
    const octaMaterial = new THREE.MeshStandardMaterial({
      color: 0x0080ff,
      emissive: 0x002244,
      emissiveIntensity: 0.5,
      metalness: 0.9,
      roughness: 0.1,
      transparent: true,
      opacity: 0.6
    })
    const octahedron = new THREE.Mesh(octaGeometry, octaMaterial)
    shapesGroup.add(octahedron)

    scene.add(shapesGroup)

    // Light rays
    const raysGroup = new THREE.Group()
    const rayCount = 12
    
    for (let i = 0; i < rayCount; i++) {
      const angle = (i / rayCount) * Math.PI * 2
      const rayGeometry = new THREE.BoxGeometry(0.02, 3, 0.02)
      const rayMaterial = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(0.6 + (i / rayCount) * 0.3, 1, 0.5),
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
      })
      const ray = new THREE.Mesh(rayGeometry, rayMaterial)
      ray.position.set(
        Math.cos(angle) * 2,
        Math.sin(angle) * 2,
        0
      )
      ray.lookAt(0, 0, 0)
      raysGroup.add(ray)
    }
    scene.add(raysGroup)

    // Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x00ffff, 0x004444)
    gridHelper.material.opacity = 0.2
    gridHelper.material.transparent = true
    scene.add(gridHelper)

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3)
    scene.add(ambientLight)

    const pointLight1 = new THREE.PointLight(0x00ffff, 1, 100)
    pointLight1.position.set(5, 5, 5)
    scene.add(pointLight1)

    const pointLight2 = new THREE.PointLight(0xff00ff, 1, 100)
    pointLight2.position.set(-5, -5, 5)
    scene.add(pointLight2)

    const pointLight3 = new THREE.PointLight(0x0080ff, 1, 100)
    pointLight3.position.set(0, 5, -5)
    scene.add(pointLight3)

    // Animation
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate)
      
      const elapsed = (Date.now() - startTimeRef.current) / 1000
      const progress = Math.min(elapsed / 4, 1) // 4 seconds animation

      // Rotate shapes
      torus.rotation.x += 0.01
      torus.rotation.y += 0.015
      icosahedron.rotation.x -= 0.012
      icosahedron.rotation.y += 0.01
      octahedron.rotation.x += 0.008
      octahedron.rotation.y -= 0.014

      // Animate particles
      if (particlesRef.current) {
        const { geometry, velocities, count } = particlesRef.current
        const positions = geometry.attributes.position.array
        
        for (let i = 0; i < count; i++) {
          const i3 = i * 3
          
          // Update position
          positions[i3] += velocities[i3]
          positions[i3 + 1] += velocities[i3 + 1]
          positions[i3 + 2] += velocities[i3 + 2]
          
          // Wrap around with smooth transition
          const dist = Math.sqrt(
            positions[i3] ** 2 + 
            positions[i3 + 1] ** 2 + 
            positions[i3 + 2] ** 2
          )
          
          if (dist > 15 || dist < 1) {
            const radius = Math.random() * 8 + 4
            const theta = Math.random() * Math.PI * 2
            const phi = Math.acos(Math.random() * 2 - 1)
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta)
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
            positions[i3 + 2] = radius * Math.cos(phi)
            
            // Reset velocity
            const speed = Math.random() * 0.03 + 0.01
            velocities[i3] = -positions[i3] * speed * 0.1 + (Math.random() - 0.5) * 0.01
            velocities[i3 + 1] = -positions[i3 + 1] * speed * 0.1 + (Math.random() - 0.5) * 0.01
            velocities[i3 + 2] = -positions[i3 + 2] * speed * 0.1 + (Math.random() - 0.5) * 0.01
          }
        }
        
        geometry.attributes.position.needsUpdate = true
      }

      // Rotate rays
      raysGroup.rotation.z += 0.005

      // Camera animation - smooth orbital motion
      const cameraRadius = 6
      const cameraSpeed = 0.3
      camera.position.x = Math.sin(elapsed * cameraSpeed) * cameraRadius
      camera.position.y = Math.cos(elapsed * cameraSpeed * 0.7) * 2 + 1
      camera.position.z = Math.cos(elapsed * cameraSpeed) * cameraRadius
      camera.lookAt(0, 0, 0)
      
      // Animate point lights
      pointLight1.position.x = Math.sin(elapsed * 0.8) * 6
      pointLight1.position.y = Math.cos(elapsed * 0.6) * 6
      pointLight2.position.x = Math.cos(elapsed * 0.7) * 6
      pointLight2.position.y = Math.sin(elapsed * 0.9) * 6
      pointLight3.position.z = Math.sin(elapsed * 0.5) * 6

      // Fade out effect
      if (progress >= 0.75 && !fadeOut) {
        setFadeOut(true)
      }

      const opacity = fadeOut ? Math.max(0, 1 - (elapsed - 3) * 2.5) : Math.min(1, progress * 1.5)

      // Apply fade to all objects
      particleMaterial.opacity = opacity * 0.8
      torusMaterial.opacity = opacity * 0.8
      icoMaterial.opacity = opacity * 0.7
      octaMaterial.opacity = opacity * 0.6
      raysGroup.children.forEach(ray => {
        ray.material.opacity = opacity * 0.6
      })
      gridHelper.material.opacity = opacity * 0.2

      renderer.render(scene, camera)

      // Complete animation
      if (elapsed >= 4 && onComplete) {
        onComplete()
      }
    }

    animate()

    // Handle resize
    const handleResize = () => {
      const newWidth = window.innerWidth
      const newHeight = window.innerHeight

      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current)
      }

      if (rendererRef.current && mountRef.current) {
        if (mountRef.current.contains(rendererRef.current.domElement)) {
          mountRef.current.removeChild(rendererRef.current.domElement)
        }
        rendererRef.current.dispose()
      }

      // Dispose geometries and materials
      torusGeometry.dispose()
      torusMaterial.dispose()
      icoGeometry.dispose()
      icoMaterial.dispose()
      octaGeometry.dispose()
      octaMaterial.dispose()
      particles.dispose()
      particleMaterial.dispose()
      gridHelper.geometry.dispose()
      gridHelper.material.dispose()
      
      raysGroup.children.forEach(ray => {
        ray.geometry.dispose()
        ray.material.dispose()
      })

      sceneRef.current = null
      cameraRef.current = null
      rendererRef.current = null
      particlesRef.current = null
    }
  }, [fadeOut, onComplete])

  return (
    <div className={`intro-animation ${fadeOut ? 'fade-out' : ''}`}>
      <div ref={mountRef} className="intro-canvas" />
      <div className="intro-content">
        <div className="intro-title">
          <h1 className="intro-title-main">Monocular 3D BODY</h1>
          <h2 className="intro-title-sub">EDITOR</h2>
        </div>
        <div className="intro-tagline">
          <span className="tagline-text">未来科技 · 3D人体姿态编辑</span>
        </div>
      </div>
    </div>
  )
}

