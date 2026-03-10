import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Box, Text } from '@radix-ui/themes'
import { translations } from '../i18n'
import './ViewerPanel.css'

const ViewerPanel = forwardRef(function ViewerPanel({
  allRigData,
  selectedPerson,
  onPersonSelect,
  jointRotations,
  jointRotationsByPerson,
  showJoints,
  language,
  measurementData
}, ref) {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const rendererRef = useRef(null)
  const controlsRef = useRef(null)
  const personsRef = useRef([]) // Array of {mesh, bones, helpers} for each person
  const frameIdRef = useRef(null)
  const raycasterRef = useRef(null)
  const mouseRef = useRef(new THREE.Vector2())
  const chestVisualsRef = useRef(null) // Chest ring + landmark spheres
  const waistVisualsRef = useRef(null) // Waist ring + landmark spheres
  const hipVisualsRef = useRef(null)   // Hip ring + pubic bone landmark sphere
  const thighVisualsRef = useRef(null) // Left thigh ring + landmark sphere
  const inseamVisualsRef = useRef(null) // Inseam line + crotch/heel spheres
  const armVisualsRef = useRef(null)   // Arm length line + 3 vertex spheres
  const shoulderBreadthVisualsRef = useRef(null) // Shoulder breadth geodesic path over back surface
  const frontBodyVisualsRef = useRef(null) // Front body geodesic path + endpoint dots
  const shoulderRingVisualsRef = useRef(null) // Shoulder-level horizontal ring (vertex 7953)
  const waistCutVisualsRef = useRef(null) // Waist slab vertex point cloud (debug)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showWireframe, setShowWireframe] = useState(false)
  const [showWaistCut, setShowWaistCut] = useState(false)
  const [hoveredVertex, setHoveredVertex] = useState(null) // { index, x, y }

  const t = translations[language]

  // Expose bakeSkinnedVertices to parent via ref
  useImperativeHandle(ref, () => ({
    /**
     * Extract deformed vertex positions from the GPU-skinned mesh.
     * Uses Three.js SkinnedMesh.boneTransform() to read back each
     * vertex position after bone rotations have been applied.
     * Returns an array of [x, y, z] triples in LOCAL mesh space
     * (same coordinate system as the original rig vertices).
     */
    bakeSkinnedVertices(personIndex) {
      const personData = personsRef.current[personIndex]
      if (!personData?.mesh) return null

      const mesh = personData.mesh
      const posAttr = mesh.geometry.getAttribute('position')
      const vertexCount = posAttr.count
      const bakedVertices = []
      const target = new THREE.Vector3()

      // Ensure all bone world matrices are current
      mesh.updateMatrixWorld(true)

      for (let i = 0; i < vertexCount; i++) {
        // Load the original vertex position into target first
        target.fromBufferAttribute(posAttr, i)

        // Apply bone transforms (skinning) — this modifies target in-place
        // Returns position in local mesh space (same coord system as original vertices)
        if (mesh.applyBoneTransform) {
          mesh.applyBoneTransform(i, target)
        } else if (mesh.boneTransform) {
          mesh.boneTransform(i, target)
        }
        // else: target already has the raw position from fromBufferAttribute

        bakedVertices.push([
          parseFloat(target.x.toFixed(6)),
          parseFloat(target.y.toFixed(6)),
          parseFloat(target.z.toFixed(6))
        ])
      }

      console.log(`[Viewer] Baked ${vertexCount} vertices for person ${personIndex}`)
      return bakedVertices
    }
  }), [])

  // Initialize Three.js scene once
  useEffect(() => {
    if (!mountRef.current || sceneRef.current) return

    console.log('[Viewer] Initializing Three.js scene')

    try {
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0a0b0d)
      sceneRef.current = scene

      const width = mountRef.current.clientWidth
      const height = mountRef.current.clientHeight
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100)
      camera.position.set(0, 1.5, 4)
      cameraRef.current = camera

      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      mountRef.current.appendChild(renderer.domElement)
      rendererRef.current = renderer

      const controls = new OrbitControls(camera, renderer.domElement)
      controls.target.set(0, 1, 0)
      controls.enableDamping = true
      controls.dampingFactor = 0.05
      controls.minDistance = 0.5
      controls.maxDistance = 15
      controls.update()
      controlsRef.current = controls

      // Raycaster for click detection
      raycasterRef.current = new THREE.Raycaster()

      // Lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
      scene.add(ambientLight)

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8)
      dirLight1.position.set(5, 10, 7)
      scene.add(dirLight1)

      const dirLight2 = new THREE.DirectionalLight(0x6b8cff, 0.3)
      dirLight2.position.set(-5, 5, -5)
      scene.add(dirLight2)

      // Grid
      const grid = new THREE.GridHelper(8, 40, 0x444444, 0x222222)
      scene.add(grid)

      // Mouse click handler - use ref to access latest onPersonSelect
      const handleClick = (event) => {
        if (!personsRef.current.length || !cameraRef.current) return

        const rect = renderer.domElement.getBoundingClientRect()
        mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
        mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current)

        // Check intersection with all meshes
        const meshes = personsRef.current.map(p => p.mesh).filter(Boolean)
        const intersects = raycasterRef.current.intersectObjects(meshes, false)

        if (intersects.length > 0) {
          const clickedMesh = intersects[0].object
          const personIndex = personsRef.current.findIndex(p => p.mesh === clickedMesh)

          if (personIndex !== -1) {
            console.log('[Viewer] Clicked person:', personIndex)
            onPersonSelect(personIndex)
          }
        }
      }

      renderer.domElement.addEventListener('click', handleClick)

      // Mouse-move handler — find nearest vertex under cursor
      const handleMouseMove = (event) => {
        if (!personsRef.current.length || !cameraRef.current || !raycasterRef.current) {
          setHoveredVertex(null)
          return
        }
        const rect = renderer.domElement.getBoundingClientRect()
        const mx = ((event.clientX - rect.left) / rect.width) * 2 - 1
        const my = -((event.clientY - rect.top) / rect.height) * 2 + 1
        raycasterRef.current.setFromCamera(new THREE.Vector2(mx, my), cameraRef.current)
        const meshes = personsRef.current.map(p => p.mesh).filter(Boolean)
        const hits = raycasterRef.current.intersectObjects(meshes, false)
        if (hits.length > 0 && hits[0].face) {
          const { face, point, object: hitMesh } = hits[0]
          const pos = hitMesh.geometry.getAttribute('position')
          const va = new THREE.Vector3().fromBufferAttribute(pos, face.a).applyMatrix4(hitMesh.matrixWorld)
          const vb = new THREE.Vector3().fromBufferAttribute(pos, face.b).applyMatrix4(hitMesh.matrixWorld)
          const vc = new THREE.Vector3().fromBufferAttribute(pos, face.c).applyMatrix4(hitMesh.matrixWorld)
          const da = va.distanceToSquared(point)
          const db = vb.distanceToSquared(point)
          const dc = vc.distanceToSquared(point)
          let vtx
          if (da <= db && da <= dc) vtx = face.a
          else if (db <= dc) vtx = face.b
          else vtx = face.c
          setHoveredVertex({ index: vtx, x: event.clientX, y: event.clientY })
        } else {
          setHoveredVertex(null)
        }
      }
      renderer.domElement.addEventListener('mousemove', handleMouseMove)

      // Animation loop
      const animate = () => {
        frameIdRef.current = requestAnimationFrame(animate)

        if (controlsRef.current) {
          controlsRef.current.update()
        }

        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current)
        }
      }
      animate()

      console.log('[Viewer] Scene initialized successfully')

      // Resize handler
      const handleResize = () => {
        if (!mountRef.current) return
        const width = mountRef.current.clientWidth
        const height = mountRef.current.clientHeight

        if (cameraRef.current) {
          cameraRef.current.aspect = width / height
          cameraRef.current.updateProjectionMatrix()
        }

        if (rendererRef.current) {
          rendererRef.current.setSize(width, height)
        }
      }
      window.addEventListener('resize', handleResize)

      return () => {
        console.log('[Viewer] Cleaning up scene...')
        window.removeEventListener('resize', handleResize)
        renderer.domElement.removeEventListener('click', handleClick)
        renderer.domElement.removeEventListener('mousemove', handleMouseMove)

        if (frameIdRef.current) {
          cancelAnimationFrame(frameIdRef.current)
          frameIdRef.current = null
        }

        if (controlsRef.current) {
          controlsRef.current.dispose()
          controlsRef.current = null
        }

        if (rendererRef.current && mountRef.current) {
          if (mountRef.current.contains(rendererRef.current.domElement)) {
            mountRef.current.removeChild(rendererRef.current.domElement)
          }
          rendererRef.current.dispose()
          rendererRef.current = null
        }

        // Clear refs
        sceneRef.current = null
        cameraRef.current = null
      }
    } catch (err) {
      console.error('[Viewer] Scene initialization error:', err)
      setError(`Failed to initialize viewer: ${err.message}`)
    }
  }, []) // Remove selectedPerson and onPersonSelect from dependencies

  // Load all persons
  useEffect(() => {
    if (!allRigData || !sceneRef.current) return

    console.log('[Viewer] Loading', allRigData.length, 'person(s)')
    setIsLoading(true)
    setError(null)

    try {
      const scene = sceneRef.current

      // Clear measurement visuals from previous session
      if (chestVisualsRef.current) {
        scene.remove(chestVisualsRef.current)
        chestVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        chestVisualsRef.current = null
      }
      if (waistVisualsRef.current) {
        scene.remove(waistVisualsRef.current)
        waistVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        waistVisualsRef.current = null
      }
      if (hipVisualsRef.current) {
        scene.remove(hipVisualsRef.current)
        hipVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        hipVisualsRef.current = null
      }
      if (thighVisualsRef.current) {
        scene.remove(thighVisualsRef.current)
        thighVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        thighVisualsRef.current = null
      }
      if (inseamVisualsRef.current) {
        scene.remove(inseamVisualsRef.current)
        inseamVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        inseamVisualsRef.current = null
      }
      if (armVisualsRef.current) {
        scene.remove(armVisualsRef.current)
        armVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        armVisualsRef.current = null
      }
      if (shoulderBreadthVisualsRef.current) {
        scene.remove(shoulderBreadthVisualsRef.current)
        shoulderBreadthVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        shoulderBreadthVisualsRef.current = null
      }
      if (frontBodyVisualsRef.current) {
        scene.remove(frontBodyVisualsRef.current)
        frontBodyVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        frontBodyVisualsRef.current = null
      }
      if (shoulderRingVisualsRef.current) {
        scene.remove(shoulderRingVisualsRef.current)
        shoulderRingVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        shoulderRingVisualsRef.current = null
      }
      if (waistCutVisualsRef.current) {
        scene.remove(waistCutVisualsRef.current)
        waistCutVisualsRef.current.traverse(child => {
          if (child.geometry) child.geometry.dispose()
          if (child.material) child.material.dispose()
        })
        waistCutVisualsRef.current = null
      }

      // Clear previous persons - with proper cleanup
      console.log('[Viewer] Clearing', personsRef.current.length, 'previous person(s)')
      personsRef.current.forEach((person, idx) => {
        console.log(`[Viewer] Cleaning up person ${idx}`)
        if (person.mesh) {
          scene.remove(person.mesh)
          if (person.mesh.geometry) {
            person.mesh.geometry.dispose()
          }
          if (person.mesh.material) {
            if (Array.isArray(person.mesh.material)) {
              person.mesh.material.forEach(mat => mat.dispose())
            } else {
              person.mesh.material.dispose()
            }
          }
          // Dispose skeleton
          if (person.mesh.skeleton) {
            person.mesh.skeleton.dispose()
          }
        }
        if (person.helpers) {
          scene.remove(person.helpers)
          person.helpers.traverse(child => {
            if (child.geometry) child.geometry.dispose()
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach(mat => mat.dispose())
              } else {
                child.material.dispose()
              }
            }
          })
        }
      })
      personsRef.current = []
      console.log('[Viewer] Cleanup complete')

      // First pass: collect all root translations to calculate center
      const rootTranslations = []
      allRigData.forEach((rigData) => {
        if (rigData.metadata && rigData.metadata.root_translation) {
          rootTranslations.push(rigData.metadata.root_translation)
        }
      })

      // Calculate center of all positions (for relative positioning)
      let centerX = 0, centerZ = 0
      if (rootTranslations.length > 0) {
        rootTranslations.forEach(rt => {
          centerX += rt[0]
          centerZ += rt[2]  // Use original Z before flipping
        })
        centerX /= rootTranslations.length
        centerZ /= rootTranslations.length
      }

      // Load each person
      allRigData.forEach((rigData, personIndex) => {
        const { mesh, skeleton: skel, metadata } = rigData

        console.log(`[Viewer] Loading person ${personIndex}, metadata:`, metadata)

        // Create geometry
        const geometry = new THREE.BufferGeometry()
        const vertices = new Float32Array(mesh.vertices.flat())
        const indices = new Uint32Array(mesh.faces.flat())
        const skinIndices = new Uint16Array(mesh.skinIndices.flat())
        const skinWeights = new Float32Array(mesh.skinWeights.flat())

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
        geometry.setIndex(new THREE.BufferAttribute(indices, 1))
        geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4))
        geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4))
        geometry.computeVertexNormals()

        // Create bones
        const bones = []
        skel.joint_names.forEach((name, idx) => {
          const bone = new THREE.Bone()
          bone.name = name
          bone.rotation.order = 'XYZ' // Set consistent rotation order
          const parentIdx = skel.parents[idx]
          const pos = skel.joint_positions[idx]

          if (parentIdx >= 0) {
            const parentPos = skel.joint_positions[parentIdx]
            bone.position.set(
              pos[0] - parentPos[0],
              pos[1] - parentPos[1],
              pos[2] - parentPos[2]
            )
          } else {
            bone.position.set(pos[0], pos[1], pos[2])
          }

          bones.push(bone)
        })

        // Build hierarchy
        bones.forEach((bone, idx) => {
          const parentIdx = skel.parents[idx]
          if (parentIdx >= 0) {
            bones[parentIdx].add(bone)
          }
        })

        // Helper to resolve head bone index even if not explicitly named "head"
        const resolveHeadBoneIndex = (neckIdx) => {
          if (neckIdx < 0) return -1
          const headNameCandidates = ['head', 'joint_113', 'joint_112', 'joint_111', 'nose']
          for (const candidate of headNameCandidates) {
            const idx = skel.joint_names.indexOf(candidate)
            if (idx >= 0) return idx
          }

          const faceKeywords = ['nose', 'eye', 'ear', 'joint_11', 'joint_12', 'jaw']
          const neckParent = skel.parents[neckIdx]

          // Head may share the same parent as the neck (sibling)
          if (neckParent >= 0) {
            for (let i = 0; i < skel.parents.length; i += 1) {
              if (skel.parents[i] === neckParent && i !== neckIdx) {
                const name = skel.joint_names[i]
                if (faceKeywords.some(keyword => name.includes(keyword))) {
                  return i
                }
              }
            }
          }

          // Head may be a child of the neck
          for (let i = 0; i < skel.parents.length; i += 1) {
            if (skel.parents[i] === neckIdx) {
              const name = skel.joint_names[i]
              if (faceKeywords.some(keyword => name.includes(keyword))) {
                return i
              }
            }
          }

          return -1
        }

        // Debug: Log neck and head bone hierarchy for this person
        // Note: personIndex is 0-based, but UI shows 1-based (Person 1, Person 2, etc.)
        const displayPersonNumber = personIndex + 1
        const neckBoneIdx = skel.joint_names.indexOf('neck')
        let headBoneIdx = skel.joint_names.indexOf('head')
        if (headBoneIdx < 0) {
          headBoneIdx = resolveHeadBoneIndex(neckBoneIdx)
        }

        console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Looking for 'neck', found at index: ${neckBoneIdx}`)
        console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Looking for 'head', found at index: ${headBoneIdx}`)
        console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Total joints: ${skel.joint_names.length}`)
        console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Sample joint names:`, skel.joint_names.slice(0, 10))

        if (neckBoneIdx >= 0) {
          const neckBone = bones[neckBoneIdx]
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - ✓ Neck bone found at index ${neckBoneIdx}, name: ${neckBone.name}`)
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Neck parent: ${skel.parents[neckBoneIdx]} (${skel.parents[neckBoneIdx] >= 0 ? skel.joint_names[skel.parents[neckBoneIdx]] : 'root'})`)
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Neck children count: ${neckBone.children.length}`)
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Neck children:`, neckBone.children.map(c => c.name))
        } else {
          console.warn(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - ✗ 'neck' bone NOT found in joint_names!`)
        }

        if (headBoneIdx >= 0) {
          const headBone = bones[headBoneIdx]
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - ✓ Head bone resolved at index ${headBoneIdx}, name: ${headBone.name}`)
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Head parent: ${skel.parents[headBoneIdx]} (${skel.parents[headBoneIdx] >= 0 ? skel.joint_names[skel.parents[headBoneIdx]] : 'root'})`)
        } else {
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Unable to resolve a head bone (only neck will be controllable)`)
        }

        // Material (different color for selected)
        const isSelected = personIndex === selectedPerson
        const material = new THREE.MeshStandardMaterial({
          color: isSelected ? 0x4aefff : 0x4a9eff,
          metalness: 0.3,
          roughness: 0.4,
          side: THREE.DoubleSide,
          emissive: isSelected ? 0x002244 : 0x000000,
          emissiveIntensity: isSelected ? 0.3 : 0
        })

        // Create skinned mesh
        const skinnedMesh = new THREE.SkinnedMesh(geometry, material)
        const rootBones = bones.filter((_, idx) => skel.parents[idx] === -1)
        rootBones.forEach(root => skinnedMesh.add(root))

        const skeleton = new THREE.Skeleton(bones)
        skinnedMesh.bind(skeleton)
        skinnedMesh.userData.personIndex = personIndex

        // Apply root translation to position the person correctly in space
        // Keep relative positions but center them and place all on ground
        if (metadata && metadata.root_translation) {
          const rootTrans = metadata.root_translation

          // Calculate relative position from center (for X and Z)
          // Keep relative positions but center them around origin
          const relativeX = rootTrans[0] - centerX
          const relativeZ = rootTrans[2] - centerZ  // Use original Z before flipping

          // Set position: relative X/Z, Y will be adjusted based on mesh bounds
          skinnedMesh.position.set(
            relativeX,         // X: relative to center (keep relative positions)
            0,                 // Y: temporary, will be adjusted below
            -relativeZ        // Z: flip Z and use relative position
          )

          // Calculate mesh bounding box to find the lowest point (feet)
          // This is in local space (mesh origin is at neck)
          geometry.computeBoundingBox()
          const bbox = geometry.boundingBox
          if (bbox) {
            const minY = bbox.min.y  // This is negative (feet are below neck/origin)
            // Adjust Y position so that the lowest point (feet) is at ground level (Y=0)
            skinnedMesh.position.y = -minY
            console.log(`[Viewer] Person ${personIndex} positioned at:`, {
              original: rootTrans,
              center: [centerX, centerZ],
              relative: [relativeX, relativeZ],
              meshMinY: minY,
              finalPosition: skinnedMesh.position.toArray(),
              note: 'Centered and placed on ground, keeping relative positions'
            })
          } else {
            console.log(`[Viewer] Person ${personIndex} positioned at:`, {
              original: rootTrans,
              relative: [relativeX, relativeZ],
              note: 'Could not compute bounding box, using relative position'
            })
          }
        }

        scene.add(skinnedMesh)

        // Skeleton helpers
        const helpersGroup = new THREE.Group()

        // Joint spheres
        skel.joint_positions.forEach((pos) => {
          const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.02, 12, 12),
            new THREE.MeshBasicMaterial({
              color: isSelected ? 0xff6688 : 0xff4466
            })
          )
          sphere.position.set(pos[0], pos[1], pos[2])
          helpersGroup.add(sphere)
        })

        // Bone lines
        const linePairs = []
        skel.parents.forEach((parentIdx, childIdx) => {
          if (parentIdx >= 0) {
            linePairs.push([parentIdx, childIdx])
          }
        })

        if (linePairs.length > 0) {
          const linePositions = new Float32Array(linePairs.length * 6)
          linePairs.forEach(([pIdx, cIdx], i) => {
            const pPos = skel.joint_positions[pIdx]
            const cPos = skel.joint_positions[cIdx]
            const offset = i * 6
            linePositions[offset + 0] = pPos[0]
            linePositions[offset + 1] = pPos[1]
            linePositions[offset + 2] = pPos[2]
            linePositions[offset + 3] = cPos[0]
            linePositions[offset + 4] = cPos[1]
            linePositions[offset + 5] = cPos[2]
          })

          const lineGeometry = new THREE.BufferGeometry()
          lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3))
          const lines = new THREE.LineSegments(
            lineGeometry,
            new THREE.LineBasicMaterial({
              color: isSelected ? 0x77ccff : 0x55aaff
            })
          )
          helpersGroup.add(lines)
        }

        // Apply same position offset to helpers
        if (metadata && metadata.root_translation) {
          const rootTrans = metadata.root_translation
          const relativeX = rootTrans[0] - centerX
          const relativeZ = rootTrans[2] - centerZ
          // Use same Y as mesh (after adjustment)
          helpersGroup.position.set(relativeX, skinnedMesh.position.y, -relativeZ)
        }

        scene.add(helpersGroup)

        // Ensure animation targets include head mapping if we have a resolved head bone
        const animationTargets = { ...rigData.animation_targets }
        if (headBoneIdx >= 0 && animationTargets.head === undefined) {
          animationTargets.head = headBoneIdx
          console.log(`[Viewer] Person ${displayPersonNumber} (index ${personIndex}) - Added fallback head animation target at bone ${headBoneIdx}`)
        }

        personsRef.current.push({
          mesh: skinnedMesh,
          bones,
          helpers: helpersGroup,
          animationTargets
        })
      })

      console.log('[Viewer] Loaded', personsRef.current.length, 'person(s)')

      // Focus camera on all meshes
      if (cameraRef.current && controlsRef.current && personsRef.current.length > 0) {
        const box = new THREE.Box3()
        personsRef.current.forEach(p => {
          if (p.mesh) box.expandByObject(p.mesh)
        })

        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())

        controlsRef.current.target.copy(center)

        const maxDim = Math.max(size.x, size.y, size.z)
        const fov = cameraRef.current.fov * (Math.PI / 180)
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 2.5

        cameraRef.current.position.set(
          center.x,
          center.y + size.y * 0.3,
          center.z + cameraZ
        )

        cameraRef.current.lookAt(center)
        controlsRef.current.update()

        console.log('[Viewer] Camera positioned')
      }

      setIsLoading(false)
    } catch (err) {
      console.error('[Viewer] Load error:', err)
      setError(`Failed to load models: ${err.message}`)
      setIsLoading(false)
    }
  }, [allRigData])

  // Apply saved rotations to a specific person
  const applyPersonRotations = useCallback((personIndex, rotations) => {
    const personData = personsRef.current[personIndex]
    if (!personData || !personData.bones || !personData.animationTargets) {
      return
    }

    const { animationTargets } = personData

    // Reset to bind pose first
    if (personData.mesh && personData.mesh.skeleton) {
      personData.mesh.skeleton.pose()
    }

    // Apply saved rotations
    Object.entries(rotations || {}).forEach(([jointName, rotation]) => {
      const boneIdx = animationTargets[jointName]
      if (boneIdx !== undefined && personData.bones[boneIdx]) {
        const bone = personData.bones[boneIdx]
        bone.rotation.set(
          rotation.x || 0,
          rotation.y || 0,
          rotation.z || 0
        )
      }
    })

    // Update skeleton
    if (personData.mesh && personData.mesh.skeleton) {
      personData.bones.forEach(bone => {
        bone.updateMatrixWorld(false)
      })
      personData.mesh.skeleton.update()
    }
  }, [])

  // Update selection highlight - only when selectedPerson changes
  useEffect(() => {
    console.log('[Viewer] Selection highlight effect triggered, selectedPerson:', selectedPerson)
    console.log('[Viewer] Persons count:', personsRef.current.length)

    personsRef.current.forEach((person, idx) => {
      const isSelected = idx === selectedPerson
      console.log(`[Viewer] Person ${idx} - isSelected:`, isSelected)

      if (person.mesh && person.mesh.material) {
        console.log(`[Viewer] Person ${idx} - updating mesh material`)
        person.mesh.material.color.setHex(isSelected ? 0x4aefff : 0x4a9eff)
        person.mesh.material.emissive.setHex(isSelected ? 0x002244 : 0x000000)
        person.mesh.material.emissiveIntensity = isSelected ? 0.3 : 0
        person.mesh.material.needsUpdate = true
      } else {
        console.warn(`[Viewer] Person ${idx} - mesh or material missing`)
      }

      // Apply saved rotations to non-selected persons to maintain their poses
      // Don't apply to selected person here - it will be handled by the jointRotations effect
      if (!isSelected) {
        const savedRotations = jointRotationsByPerson?.[idx]
        if (savedRotations && Object.keys(savedRotations).length > 0) {
          applyPersonRotations(idx, savedRotations)
        }
      }

      // Update helper colors
      if (person.helpers) {
        person.helpers.children.forEach(child => {
          if (child.type === 'Mesh' && child.material) {
            child.material.color.setHex(isSelected ? 0xff6688 : 0xff4466)
            child.material.needsUpdate = true
          } else if (child.type === 'LineSegments' && child.material) {
            child.material.color.setHex(isSelected ? 0x77ccff : 0x55aaff)
            child.material.needsUpdate = true
          }
        })
      }
    })
  }, [selectedPerson, applyPersonRotations, jointRotationsByPerson])

  // Apply joint rotations to selected person only
  useEffect(() => {
    console.log('[Viewer] Apply rotations effect triggered')
    console.log('[Viewer] selectedPerson:', selectedPerson)
    console.log('[Viewer] personsRef.current.length:', personsRef.current.length)
    console.log('[Viewer] jointRotations:', jointRotations)

    if (selectedPerson < 0 || selectedPerson >= personsRef.current.length) {
      console.warn('[Viewer] Selected person out of range')
      return
    }

    const selectedPersonData = personsRef.current[selectedPerson]
    if (!selectedPersonData) {
      console.warn('[Viewer] Selected person data not found')
      return
    }

    if (!selectedPersonData.bones) {
      console.warn('[Viewer] Selected person has no bones')
      return
    }

    console.log('[Viewer] Selected person has', selectedPersonData.bones.length, 'bones')

    const { animationTargets } = selectedPersonData
    if (!animationTargets) {
      console.warn('[Viewer] Selected person has no animation targets')
      return
    }

    console.log('[Viewer] Animation targets:', animationTargets)

    // Reset all bones to bind pose first
    if (selectedPersonData.mesh && selectedPersonData.mesh.skeleton) {
      selectedPersonData.mesh.skeleton.pose() // Reset to bind pose
    }

    // Apply rotations to bones
    Object.entries(jointRotations).forEach(([jointName, rotation]) => {
      const boneIdx = animationTargets[jointName]

      if (boneIdx === undefined) {
        console.warn(`[Viewer] Joint "${jointName}" not in animation targets`)
        return
      }

      if (!selectedPersonData.bones[boneIdx]) {
        console.warn(`[Viewer] Bone ${boneIdx} not found for joint "${jointName}"`)
        return
      }

      const bone = selectedPersonData.bones[boneIdx]
      console.log(`[Viewer] Applying rotation to ${jointName} (bone ${boneIdx}):`, rotation)
      console.log(`[Viewer]   - Bone name: ${bone.name}, Parent: ${bone.parent?.name || 'none'}`)

      // Apply rotation in local space
      bone.rotation.set(
        rotation.x || 0,
        rotation.y || 0,
        rotation.z || 0
      )
    })

    // Update skeleton and mesh after all rotations are applied
    if (selectedPersonData.mesh && selectedPersonData.mesh.skeleton) {
      // Update bone matrices
      selectedPersonData.bones.forEach(bone => {
        bone.updateMatrixWorld(false) // Update from local to world
      })
      // Update skeleton
      selectedPersonData.mesh.skeleton.update()
    }
  }, [jointRotations, selectedPerson])

  // Control visibility
  useEffect(() => {
    personsRef.current.forEach(person => {
      if (!person.helpers) return

      person.helpers.visible = showJoints

      person.helpers.children.forEach(child => {
        // Show both joints (spheres) and bones (lines) based on showJoints
        child.visible = showJoints
      })
    })
  }, [showJoints])

  // Wireframe toggle — switches all person meshes between solid and wireframe
  useEffect(() => {
    personsRef.current.forEach((person, idx) => {
      const mat = person.mesh?.material
      if (!mat) return
      const isSelected = idx === selectedPerson
      mat.wireframe = showWireframe
      if (showWireframe) {
        mat.color.setHex(isSelected ? 0x00ffff : 0x44aaff)
        if (mat.emissive) mat.emissive.setHex(0x000000)
        mat.emissiveIntensity = 0
      } else {
        mat.color.setHex(isSelected ? 0x4aefff : 0x4a9eff)
        if (mat.emissive) mat.emissive.setHex(isSelected ? 0x002244 : 0x000000)
        mat.emissiveIntensity = isSelected ? 0.3 : 0
      }
      mat.needsUpdate = true
    })
  }, [showWireframe, selectedPerson])

  // Chest ring + landmark visualization (SMPLX-style measurement)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Always clear stale visuals first
    if (chestVisualsRef.current) {
      scene.remove(chestVisualsRef.current)
      chestVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      chestVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const ringPoints = landmarks.chest_ring_points
    const leftPt = landmarks.left_chest_landmark
    const rightPt = landmarks.right_chest_landmark
    if (!ringPoints?.length && !leftPt && !rightPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    // Match the world-space offset of the person's mesh
    group.position.copy(personData.mesh.position)

    // 1. Closed measurement ring through the convex hull vertices
    if (ringPoints && ringPoints.length >= 3) {
      // Close the loop by appending the first point at the end
      const closed = [...ringPoints, ringPoints[0]]
      const posArr = new Float32Array(closed.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      const ring = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xffcc00, linewidth: 2 })
      )
      group.add(ring)
    }

    // 2. Left chest landmark sphere (vertex 7380) — red
    if (leftPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xff3333 })
      )
      sphere.position.set(leftPt[0], leftPt[1], leftPt[2])
      group.add(sphere)
    }

    // 3. Right chest landmark sphere (vertex 6156) — blue
    if (rightPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x44aaff })
      )
      sphere.position.set(rightPt[0], rightPt[1], rightPt[2])
      group.add(sphere)
    }

    scene.add(group)
    chestVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Waist ring + landmark visualization (SMPLX-style measurement)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Always clear stale visuals first
    if (waistVisualsRef.current) {
      scene.remove(waistVisualsRef.current)
      waistVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      waistVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const ringPoints = landmarks.waist_ring_points
    const frontPt = landmarks.belly_button_landmark
    const backPt = landmarks.back_belly_button_landmark
    if (!ringPoints?.length && !frontPt && !backPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // 1. Closed waist ring — teal/mint to distinguish from gold chest ring
    if (ringPoints && ringPoints.length >= 3) {
      const closed = [...ringPoints, ringPoints[0]]
      const posArr = new Float32Array(closed.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      const ring = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0x00ffcc, linewidth: 2 })
      )
      group.add(ring)
    }

    // 2. Front (belly button) landmark sphere — green
    if (frontPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x00ff44 })
      )
      sphere.position.set(frontPt[0], frontPt[1], frontPt[2])
      group.add(sphere)
    }

    // 3. Back (lumbar) landmark sphere — orange
    if (backPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xff8800 })
      )
      sphere.position.set(backPt[0], backPt[1], backPt[2])
      group.add(sphere)
    }

    // 4. spine3 joint — purple circumference ring (body cross-section at that level)
    //    + sphere marker at the joint centre, matching the chest/waist/hip ring style
    const spine3Pt = landmarks.waist_spine3_joint
    const pelvisPt = landmarks.waist_pelvis_joint
    const spine3Ring = landmarks.spine3_ring_points

    if (spine3Ring && spine3Ring.length >= 3) {
      const closed = [...spine3Ring, spine3Ring[0]]
      const posArr = new Float32Array(closed.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      const ring = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xcc44ff, linewidth: 2 })
      )
      group.add(ring)
    }

    if (spine3Pt) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.016, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xcc44ff })
      )
      s.position.set(spine3Pt[0], spine3Pt[1], spine3Pt[2])
      group.add(s)
    }

    // 5. Pelvis joint — smaller purple sphere
    if (pelvisPt) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.012, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x7722cc })
      )
      s.position.set(pelvisPt[0], pelvisPt[1], pelvisPt[2])
      group.add(s)
    }

    // 6. Wrist joints — small cyan spheres (hand positions at waist level)
    for (const wPt of [landmarks.waist_left_wrist_joint, landmarks.waist_right_wrist_joint]) {
      if (wPt) {
        const s = new THREE.Mesh(
          new THREE.SphereGeometry(0.010, 12, 12),
          new THREE.MeshBasicMaterial({ color: 0x00ccff })
        )
        s.position.set(wPt[0], wPt[1], wPt[2])
        group.add(s)
      }
    }

    scene.add(group)
    waistVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Hip ring + pubic bone landmark (SMPLX-style measurement)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (hipVisualsRef.current) {
      scene.remove(hipVisualsRef.current)
      hipVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      hipVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const ringPoints = landmarks.hip_ring_points
    const pubicPt = landmarks.pubic_bone_landmark
    if (!ringPoints?.length && !pubicPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // 1. Closed hip ring — warm orange to distinguish from chest (gold) and waist (teal)
    if (ringPoints && ringPoints.length >= 3) {
      const closed = [...ringPoints, ringPoints[0]]
      const posArr = new Float32Array(closed.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      const ring = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xff6600, linewidth: 2 })
      )
      group.add(ring)
    }

    // 2. Pubic bone landmark sphere — yellow
    if (pubicPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xffff00 })
      )
      sphere.position.set(pubicPt[0], pubicPt[1], pubicPt[2])
      group.add(sphere)
    }

    scene.add(group)
    hipVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Left thigh ring (SMPLX-style — UV-projected, leg-axis plane)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (thighVisualsRef.current) {
      scene.remove(thighVisualsRef.current)
      thighVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      thighVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const ringPoints = landmarks.left_thigh_ring_points
    const thighPt = landmarks.left_thigh_landmark
    if (!ringPoints?.length && !thighPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Closed left thigh ring — magenta to distinguish from all other rings
    if (ringPoints && ringPoints.length >= 3) {
      const closed = [...ringPoints, ringPoints[0]]
      const posArr = new Float32Array(closed.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      const ring = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xff00aa, linewidth: 2 })
      )
      group.add(ring)
    }

    // Left thigh landmark sphere — bright pink
    if (thighPt) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.016, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xff44cc })
      )
      sphere.position.set(thighPt[0], thighPt[1], thighPt[2])
      group.add(sphere)
    }

    scene.add(group)
    thighVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Inseam line + crotch sphere + heel sphere
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (inseamVisualsRef.current) {
      scene.remove(inseamVisualsRef.current)
      inseamVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      inseamVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const crotchPt = landmarks.inseam_crotch_landmark
    const heelPt = landmarks.inseam_heel_landmark
    if (!crotchPt && !heelPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Line from crotch to heel — cyan to stand out from other markers
    if (crotchPt && heelPt) {
      const pts = new Float32Array([...crotchPt, ...heelPt])
      const lineGeo = new THREE.BufferGeometry()
      lineGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3))
      group.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x00eeff, linewidth: 2 })))
    }

    // Crotch sphere — bright cyan
    if (crotchPt) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x00eeff })
      )
      s.position.set(crotchPt[0], crotchPt[1], crotchPt[2])
      group.add(s)
    }

    // Heel sphere — white
    if (heelPt) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      )
      s.position.set(heelPt[0], heelPt[1], heelPt[2])
      group.add(s)
    }

    scene.add(group)
    inseamVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Arm length line + shoulder / elbow / wrist spheres (green)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (armVisualsRef.current) {
      scene.remove(armVisualsRef.current)
      armVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      armVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const waypoints = landmarks.arm_waypoints
    if (!waypoints?.length) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Line shoulder → elbow → wrist (lime green)
    const pts = new Float32Array(waypoints.flatMap(p => p))
    const lineGeo = new THREE.BufferGeometry()
    lineGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3))
    group.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x44ff88, linewidth: 2 })))

    // Distinct sphere per anchor: shoulder=cyan, elbow=yellow, wrist=cyan
    const DOT_COLORS = [0x00ddff, 0xffee00, 0x00ddff]
    for (let i = 0; i < waypoints.length; i++) {
      const pt = waypoints[i]
      const color = DOT_COLORS[i] ?? 0x44ff88
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.013, 16, 16),
        new THREE.MeshBasicMaterial({ color })
      )
      s.position.set(pt[0], pt[1], pt[2])
      group.add(s)
    }

    scene.add(group)
    armVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Shoulder breadth — geodesic path over the back surface (right acromion → left acromion)
  // Cyan-green line hugging the posterior shoulder/back; cyan endpoint dots.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (shoulderBreadthVisualsRef.current) {
      scene.remove(shoulderBreadthVisualsRef.current)
      shoulderBreadthVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      shoulderBreadthVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const pathPts = landmarks.shoulder_breadth_path
    if (!pathPts?.length) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Path line — cyan-green over the back surface
    const positions = new Float32Array(pathPts.flat())
    const lineGeo = new THREE.BufferGeometry()
    lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    group.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x00ddaa, depthTest: false, linewidth: 2 })))

    // Endpoint dots — left (lime) and right (yellow)
    const addDot = (pt, color, r = 0.013) => {
      if (!pt) return
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(r, 10, 10),
        new THREE.MeshBasicMaterial({ color, depthTest: false })
      )
      m.position.set(pt[0], pt[1], pt[2])
      group.add(m)
    }
    addDot(landmarks.shoulder_left_landmark, 0x44ff44)  // left acromion — lime
    addDot(landmarks.shoulder_right_landmark, 0xffff00)  // right acromion — yellow

    scene.add(group)
    shoulderBreadthVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Front body length — geodesic surface path (neck top → crotch)
  // Draws the actual surface path the tape measure would follow on the front body.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (frontBodyVisualsRef.current) {
      scene.remove(frontBodyVisualsRef.current)
      frontBodyVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      frontBodyVisualsRef.current = null
    }

    const landmarks = measurementData?.landmarks
    if (!landmarks) return

    const pathPts = landmarks.shoulder_to_crotch_path
    const neckPt = landmarks.neck_top_landmark
    const crotchPt = landmarks.inseam_crotch_landmark
    const waistMidPt = landmarks.front_waist_midline_landmark  // belly button (confirmed front)
    if (!pathPts?.length && !neckPt) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Geodesic polyline — front-constrained, orange-red
    if (pathPts && pathPts.length >= 2) {
      const posArr = new Float32Array(pathPts.flatMap(p => p))
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff5500, linewidth: 2 })))
    }

    const addDot = (pt, color, radius = 0.013) => {
      if (!pt) return
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 16, 16),
        new THREE.MeshBasicMaterial({ color })
      )
      s.position.set(pt[0], pt[1], pt[2])
      group.add(s)
    }

    addDot(neckPt, 0xffffff, 0.015)  // neck top — white
    addDot(crotchPt, 0xff5500, 0.015)  // crotch — orange-red
    addDot(waistMidPt, 0xffaa00)         // belly button mid-anchor — amber

    scene.add(group)
    frontBodyVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Shoulder ring — horizontal cross-section through vertex 7953 (bright purple)
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    if (shoulderRingVisualsRef.current) {
      scene.remove(shoulderRingVisualsRef.current)
      shoulderRingVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      shoulderRingVisualsRef.current = null
    }

    const ringPts = measurementData?.landmarks?.shoulder_ring_points
    const landmarkPt = measurementData?.landmarks?.shoulder_ring_landmark
    if (!ringPts?.length) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    const addDot = (pt, color, r = 0.012) => {
      if (!pt) return
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(r, 10, 10),
        new THREE.MeshBasicMaterial({ color, depthTest: false })
      )
      mesh.position.set(pt[0], pt[1], pt[2])
      group.add(mesh)
    }

    // Ring line
    const positions = new Float32Array(ringPts.flat())
    // Close the loop
    const closed = new Float32Array(positions.length + 3)
    closed.set(positions)
    closed[positions.length] = positions[0]
    closed[positions.length + 1] = positions[1]
    closed[positions.length + 2] = positions[2]
    const ringGeo = new THREE.BufferGeometry()
    ringGeo.setAttribute('position', new THREE.BufferAttribute(closed, 3))
    group.add(new THREE.Line(ringGeo, new THREE.LineBasicMaterial({ color: 0xaa44ff, depthTest: false, linewidth: 2 })))

    // Landmark dot at vertex 7953
    addDot(landmarkPt, 0xcc88ff, 0.014)

    scene.add(group)
    shoulderRingVisualsRef.current = group
  }, [measurementData, selectedPerson])

  // Waist slab vertex debug visualisation — shows every mesh vertex that lies
  // inside the horizontal cutting slab used for the waist circumference.
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    // Always clear previous slab visuals
    if (waistCutVisualsRef.current) {
      scene.remove(waistCutVisualsRef.current)
      waistCutVisualsRef.current.traverse(child => {
        if (child.geometry) child.geometry.dispose()
        if (child.material) child.material.dispose()
      })
      waistCutVisualsRef.current = null
    }

    if (!showWaistCut) return

    const slabVerts = measurementData?.landmarks?.waist_slab_vertices
    if (!slabVerts?.length) return

    const personData = personsRef.current[selectedPerson]
    if (!personData?.mesh) return

    const group = new THREE.Group()
    group.position.copy(personData.mesh.position)

    // Render all slab vertices as a magenta point cloud
    const positions = new Float32Array(slabVerts.flat())
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const points = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ color: 0xff00ff, size: 0.012, sizeAttenuation: true })
    )
    group.add(points)

    // Also draw a thin semi-transparent horizontal disc at the waist level
    // to make the cutting plane obvious
    const frontPt = measurementData?.landmarks?.belly_button_landmark
    const backPt = measurementData?.landmarks?.back_belly_button_landmark
    if (frontPt && backPt) {
      const waistY = (frontPt[1] + backPt[1]) / 2
      const waistX = (frontPt[0] + backPt[0]) / 2
      const waistZ = (frontPt[2] + backPt[2]) / 2
      const discGeo = new THREE.CircleGeometry(0.30, 64)
      discGeo.rotateX(Math.PI / 2)
      const discMat = new THREE.MeshBasicMaterial({
        color: 0xff00ff,
        transparent: true,
        opacity: 0.10,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const disc = new THREE.Mesh(discGeo, discMat)
      disc.position.set(waistX, waistY, waistZ)
      group.add(disc)
    }

    scene.add(group)
    waistCutVisualsRef.current = group
  }, [showWaistCut, measurementData, selectedPerson])

  return (
    <div className="viewer-container">
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {!allRigData && !error && (
        <div className="viewer-placeholder">
          <div className="placeholder-content">
            <div className="placeholder-icon">🎭</div>
            <h2>{t.uploadToStart}</h2>
            <p>{t.dragAndDrop}</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="viewer-loading">
          <Text size="3" weight="bold">{t.loadingModel}</Text>
        </div>
      )}

      {error && (
        <div className="viewer-error">
          <Box p="4" style={{ background: 'var(--red-3)', borderRadius: '8px', border: '1px solid var(--red-6)' }}>
            <Text size="2" color="red" weight="bold" mb="2" style={{ display: 'block' }}>
              {t.errorLoading}
            </Text>
            <Text size="2" color="red">
              {error}
            </Text>
          </Box>
        </div>
      )}

      {allRigData && allRigData.length > 1 && (
        <div className="viewer-hint">
          <Text size="1" color="gray">
            {language === 'zh' ? '点击模型以选择人物' : 'Click on a model to select person'}
          </Text>
        </div>
      )}

      {allRigData && (
        <button
          className={`viewer-wireframe-btn${showWireframe ? ' active' : ''}`}
          onClick={() => setShowWireframe(v => !v)}
          title={showWireframe
            ? (language === 'zh' ? '切换为实体模型' : 'Switch to solid mesh')
            : (language === 'zh' ? '切换为线框模型' : 'Switch to wireframe')}
        >
          {showWireframe
            ? (language === 'zh' ? '◼ 实体' : '◼ Solid')
            : (language === 'zh' ? '⬡ 线框' : '⬡ Wire')}
        </button>
      )}

      {allRigData && measurementData && (
        <button
          className={`viewer-wireframe-btn${showWaistCut ? ' active' : ''}`}
          style={{ top: '46px' }}
          onClick={() => setShowWaistCut(v => !v)}
          title={language === 'zh' ? '显示腰部截面顶点' : 'Show waist cut vertices'}
        >
          {showWaistCut
            ? (language === 'zh' ? '◉ 腰截面' : '◉ Waist Cut')
            : (language === 'zh' ? '○ 腰截面' : '○ Waist Cut')}
        </button>
      )}

      {hoveredVertex !== null && (
        <div
          className="vertex-tooltip"
          style={{ left: hoveredVertex.x + 14, top: hoveredVertex.y - 12 }}
        >
          #{hoveredVertex.index}
        </div>
      )}
    </div>
  )
})

export default ViewerPanel
