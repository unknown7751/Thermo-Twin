/**
 * HVACScene — interactive 3D digital twin of the HVAC unit.
 *
 * Renders 4 component meshes (compressor, condenser+fan, evaporator,
 * expansion valve) wired to live data:
 *   • health glow      ← Phase-2 UKF state estimates
 *   • rotation speed    ← live compressor power / fan RPM
 *   • refrigerant flow ← speed ∝ compressor power
 *   • hover / click    ← raycasting → React callbacks
 *
 * THREE.js mental model:
 *   Scene  : the world container
 *   Camera : the viewpoint (PerspectiveCamera here)
 *   Renderer: draws Scene from Camera onto the <canvas> every frame
 *   Raycaster: shoots a ray from the mouse into the scene for picking
 *
 * The class owns its own requestAnimationFrame loop. React only feeds it
 * data via update*() and is notified of interaction via callbacks. All GPU
 * resources are released in dispose().
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  createCompressor,
  createCondenser,
  createEvaporator,
  createExpansionValve,
  buildCircuitPath,
  createPipes,
  createRefrigerantFlow,
} from './hvacGeometries.js'
import { healthToRGB, healthToGlow } from './colorUtils.js'

// Which health metric drives each component's glow
const HEALTH_KEY = {
  compressor: 'compressor_efficiency_pct',
  condenser:  'fan_health_pct',
  evaporator: 'refrigerant_charge_pct',
  valve:      null, // not health-critical
}

export class HVACScene {
  constructor(canvasElement, initialHealthState = null) {
    if (!canvasElement) {
      throw new Error('HVACScene: a valid HTMLCanvasElement is required')
    }
    this.canvas = canvasElement

    // ── Renderer ─────────────────────────────────────────────────────────
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      precision: 'highp',
      powerPreference: 'high-performance',
    })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this._resize()

    // ── Scene + camera ───────────────────────────────────────────────────
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0f172a)

    const aspect = this._aspect()
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000)
    this.camera.position.set(0.5, 2.2, 5.5)
    this.camera.lookAt(0, 0.2, 0)

    // ── State ────────────────────────────────────────────────────────────
    this.currentSensorData = {
      compressor_power_kw: 3.5, discharge_pressure_psi: 245,
      fan_rpm: 1190, supply_air_temp_c: 11,
    }
    this.currentHealthState = initialHealthState || {
      refrigerant_charge_pct: 100, compressor_efficiency_pct: 100, fan_health_pct: 100,
    }
    this.currentPrediction = { ...this.currentSensorData }

    this.onComponentClickCallback = null
    this.onHoverCallback = null
    this._hovered = null
    this._disposed = false
    this._animationId = null

    // Bound handlers (so removeEventListener works)
    this._onMove   = this._handleMouseMove.bind(this)
    this._onClick  = this._handleClick.bind(this)
    this._onResize = this._onWindowResize.bind(this)

    this._setupScene()
    this._setupLighting()
    this._setupControls()
    this._setupRaycaster()

    window.addEventListener('resize', this._onResize)
    this._clock = new THREE.Clock()
    this._animate()
  }

  // ── Public API ─────────────────────────────────────────────────────────
  updateSensorData(d)  { this.currentSensorData = { ...this.currentSensorData, ...d } }
  updateHealthState(h) { this.currentHealthState = { ...this.currentHealthState, ...h } }
  updatePrediction(p)  { this.currentPrediction = { ...this.currentPrediction, ...p } }

  setComponentClickCallback(cb) { this.onComponentClickCallback = cb }
  setHoverCallback(cb)          { this.onHoverCallback = cb }

  dispose() {
    this._disposed = true
    if (this._animationId) cancelAnimationFrame(this._animationId)
    window.removeEventListener('resize', this._onResize)
    this.canvas.removeEventListener('mousemove', this._onMove)
    this.canvas.removeEventListener('click', this._onClick)
    if (this._controls) this._controls.dispose()
    this.scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose()
      if (obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
        mats.forEach((m) => m.dispose())
      }
    })
    this.renderer.dispose()
    this.scene.clear()
  }

  // ── Scene construction ─────────────────────────────────────────────────
  _setupScene() {
    this.unit = new THREE.Group()

    const comp = createCompressor()
    const cond = createCondenser()
    const evap = createEvaporator()
    const valve = createExpansionValve()

    this.compressorMesh     = comp.mesh
    this.compressorMaterial = comp.material
    this.condenserGroup     = cond.group
    this.condenserBody      = cond.body
    this.condenserFanMesh   = cond.fan
    this.condenserMaterial  = cond.material
    this.evaporatorGroup    = evap.group
    this.evaporatorMesh     = evap.core
    this.evaporatorMaterial = evap.material
    this.valveMesh          = valve.mesh
    this.valveMaterial      = valve.material

    this.unit.add(comp.mesh, cond.group, evap.group, valve.mesh)

    // Refrigerant circuit through the 4 component centres
    const positions = {
      compressor: comp.mesh.position,
      condenser:  cond.group.position,
      evaporator: evap.group.position,
      valve:      valve.mesh.position,
    }
    const curve = buildCircuitPath(positions)
    this._circuitCurve = curve
    this.unit.add(createPipes(curve))

    const flow = createRefrigerantFlow(140)
    this.flow = flow
    this.unit.add(flow.points)

    this.scene.add(this.unit)

    // Meshes eligible for raycasting (name → component id).
    // condenserGroup is included so its fan-blade children are also hit-tested
    // (recursive=true in intersectObjects).  Any hit inside the group resolves
    // to 'condenser' via _resolveComponentName().
    this._pickables = [
      this.compressorMesh,
      this.condenserGroup,  // ← whole group, fan blades hit-tested recursively
      this.evaporatorMesh,  // ← core mesh only; fins are decorative, not pickable
      this.valveMesh,
    ]
  }

  _setupLighting() {
    const key = new THREE.DirectionalLight(0xffffff, 1.1)
    key.position.set(4, 6, 5)
    this.scene.add(key)
    this.scene.add(new THREE.AmbientLight(0x4b5563, 0.7))
    // Soft point lights near the two health-critical components
    const p1 = new THREE.PointLight(0xfbbf24, 0.4, 6); p1.position.set(-2, 1, 1)
    const p2 = new THREE.PointLight(0x3b82f6, 0.4, 6); p2.position.set(2, 1, 1)
    this.scene.add(p1, p2)
  }

  _setupControls() {
    try {
      this._controls = new OrbitControls(this.camera, this.canvas)
      this._controls.enableDamping = true
      this._controls.dampingFactor = 0.08
      this._controls.minDistance = 3
      this._controls.maxDistance = 12
      this._controls.target.set(0, 0.2, 0)
    } catch {
      this._controls = null // OrbitControls optional; scene still renders
    }
  }

  _setupRaycaster() {
    this.raycaster = new THREE.Raycaster()
    this.mouse = new THREE.Vector2()
    this.canvas.addEventListener('mousemove', this._onMove)
    this.canvas.addEventListener('click', this._onClick)
  }

  // ── Interaction ────────────────────────────────────────────────────────
  _pick(event) {
    const rect = this.canvas.getBoundingClientRect()
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    this.raycaster.setFromCamera(this.mouse, this.camera)
    // recursive=true so fan blades / evaporator fins (child meshes) are tested
    const hits = this.raycaster.intersectObjects(this._pickables, true)
    if (!hits.length) return null
    return this._resolveComponentName(hits[0].object)
  }

  /**
   * Walk up the Three.js parent chain from the hit object until we find a node
   * whose name is one of the four known component ids.  This lets any blade,
   * fin, or sub-mesh inside a group resolve back to the top-level component.
   */
  _resolveComponentName(obj) {
    const KNOWN = new Set(['compressor', 'condenser', 'evaporator', 'valve'])
    let cur = obj
    while (cur) {
      if (KNOWN.has(cur.name)) return cur.name
      cur = cur.parent
    }
    return null
  }

  _handleMouseMove(event) {
    const name = this._pick(event)
    if (name !== this._hovered) {
      this._hovered = name
      this.canvas.style.cursor = name ? 'pointer' : 'default'
      if (this.onHoverCallback) {
        this.onHoverCallback(name ? this._buildHoverInfo(name) : null)
      }
    }
  }

  _handleClick(event) {
    const name = this._pick(event)
    if (name && this.onComponentClickCallback) {
      this.onComponentClickCallback(name)
    }
  }

  /** Real vs predicted sensor for the sensor associated with a component. */
  _buildHoverInfo(component) {
    // Must match HVAC3D / backend /twin/component-history sensor_map
    const sensorOf = {
      compressor: 'compressor_power_kw',
      condenser:  'fan_rpm',
      evaporator: 'discharge_pressure_psi',
      valve:      'supply_air_temp_c',
    }
    const key = sensorOf[component]
    const real = this.currentSensorData[key]
    const pred = this.currentPrediction[key]
    const div  = (real ?? 0) - (pred ?? 0)
    const healthKey = HEALTH_KEY[component]
    return {
      component,
      sensorKey: key,
      real, predicted: pred, divergence: div,
      health: healthKey ? this.currentHealthState[healthKey] : null,
      status: Math.abs(div) > Math.abs((pred ?? 1) * 0.05) ? 'diverging' : 'normal',
    }
  }

  // ── Health glow ────────────────────────────────────────────────────────
  updateHealthGlow(h) {
    const apply = (mat, pct) => {
      const { r, g, b } = healthToRGB(pct)
      mat.emissive.setRGB(r, g, b)
      mat.emissiveIntensity = healthToGlow(pct)
    }
    apply(this.compressorMaterial, h.compressor_efficiency_pct ?? 100)
    apply(this.condenserMaterial,  h.fan_health_pct ?? 100)
    apply(this.evaporatorMaterial, h.refrigerant_charge_pct ?? 100)
    // valve: faint constant glow, not health-driven
    this.valveMaterial.emissiveIntensity = 0.12
  }

  // ── Refrigerant flow ───────────────────────────────────────────────────
  updateRefrigerantFlow(elapsed) {
    const { geometry, count } = this.flow
    const pos = geometry.attributes.position.array
    const col = geometry.attributes.color.array

    // 3.5 kW → one circuit / 2 s; scales linearly with power
    const power = Math.max(0.3, this.currentSensorData.compressor_power_kw || 3.5)
    const speed = (power / 3.5) / 2.0 // circuits per second
    const base  = (elapsed * speed) % 1

    const tmp = new THREE.Vector3()
    for (let i = 0; i < count; i++) {
      const t = (base + i / count) % 1
      this._circuitCurve.getPointAt(t, tmp)
      pos[i * 3] = tmp.x; pos[i * 3 + 1] = tmp.y; pos[i * 3 + 2] = tmp.z
      // hot (red) leaving compressor → cold (blue) after expansion
      const hue = (1 - t) * 0.66 // 0=red .. 0.66=blue
      const c = new THREE.Color().setHSL(hue, 1.0, 0.55)
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b
    }
    geometry.attributes.position.needsUpdate = true
    geometry.attributes.color.needsUpdate = true
  }

  // ── Animation loop ─────────────────────────────────────────────────────
  _animate() {
    if (this._disposed) return
    this._animationId = requestAnimationFrame(() => this._animate())
    const elapsed = this._clock.getElapsedTime()

    // Compressor spin ∝ power
    const powerFrac = (this.currentSensorData.compressor_power_kw || 3.5) / 5.0
    this.compressorMesh.rotation.y += 0.02 * (0.5 + 1.5 * powerFrac)

    // Condenser fan spin ∝ rpm
    const rpmFrac = (this.currentSensorData.fan_rpm || 1190) / 1500.0
    this.condenserFanMesh.rotation.x += 0.06 * (0.4 + 1.6 * rpmFrac)

    // Evaporator gentle heat-exchange bob
    this.evaporatorGroup.position.y = 1.0 + Math.sin(elapsed * 3) * 0.04

    this.updateRefrigerantFlow(elapsed)
    this.updateHealthGlow(this.currentHealthState)

    if (this._controls) this._controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  // ── Resize ─────────────────────────────────────────────────────────────
  _aspect() {
    return (this.canvas.clientWidth || 1) / (this.canvas.clientHeight || 1)
  }

  _resize() {
    const w = this.canvas.clientWidth || 800
    const h = this.canvas.clientHeight || 600
    this.renderer.setSize(w, h, false)
  }

  _onWindowResize() {
    if (this._disposed) return
    this._resize()
    this.camera.aspect = this._aspect()
    this.camera.updateProjectionMatrix()
  }
}

export default HVACScene
