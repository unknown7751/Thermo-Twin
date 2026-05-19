/**
 * Geometry/mesh factory helpers for the HVAC 3D scene.
 *
 * Kept separate from hvacScene.js so the scene class stays focused on
 * orchestration (lighting, raycasting, animation) while the actual mesh
 * construction lives here and is independently testable.
 *
 * THREE.js primer for readers new to it:
 *   • Geometry  = the shape (vertices/faces), e.g. CylinderGeometry
 *   • Material  = how the surface reacts to light (colour, metalness,
 *                 emissive glow). MeshStandardMaterial is physically based.
 *   • Mesh      = Geometry + Material, the actual object placed in the scene.
 *   • Group     = a transform container; moving the group moves its children.
 */

import * as THREE from 'three'

/** Standard PBR material with an emissive channel used for the health glow. */
export function makeHealthMaterial(baseColorHex) {
  return new THREE.MeshStandardMaterial({
    color: baseColorHex,
    metalness: 0.7,
    roughness: 0.3,
    emissive: baseColorHex,
    emissiveIntensity: 0.0, // driven each frame by updateHealthGlow()
  })
}

/** Compressor — vertical cylinder, yellow-orange metal. */
export function createCompressor() {
  const geo = new THREE.CylinderGeometry(0.3, 0.3, 0.6, 32)
  const mat = makeHealthMaterial(0xfbbf24)
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(-2, 0, 0)
  mesh.name = 'compressor'
  return { mesh, material: mat }
}

/**
 * Condenser — a blue box housing plus a separate fan-blade group that
 * spins independently (returned so the animation loop can rotate just it).
 */
export function createCondenser() {
  const group = new THREE.Group()
  group.position.set(2, 0, 0)
  group.name = 'condenser'

  const bodyMat = makeHealthMaterial(0x3b82f6)
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.5, 0.5), bodyMat)
  body.name = 'condenser'
  group.add(body)

  // Fan: a hub + 4 flat blades, parented to a group that rotates on Z.
  const fan = new THREE.Group()
  const bladeMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8, metalness: 0.6, roughness: 0.4,
  })
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.1, 16), bladeMat)
  hub.rotation.x = Math.PI / 2
  fan.add(hub)
  for (let i = 0; i < 4; i++) {
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.1, 0.02), bladeMat)
    blade.position.x = 0.18
    const pivot = new THREE.Group()
    pivot.rotation.z = (i * Math.PI) / 2
    pivot.add(blade)
    fan.add(pivot)
  }
  fan.position.set(0.55, 0, 0)
  fan.name = 'condenserFan'
  group.add(fan)

  return { group, body, fan, material: bodyMat }
}

/** Evaporator — finned cyan box (fins are thin slabs for a coil look). */
export function createEvaporator() {
  const group = new THREE.Group()
  group.position.set(0, 1.0, 0)
  group.name = 'evaporator'

  const mat = makeHealthMaterial(0x06b6d4)
  const core = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.4, 0.8), mat)
  core.name = 'evaporator'
  group.add(core)

  const finMat = new THREE.MeshStandardMaterial({
    color: 0x0e7490, metalness: 0.5, roughness: 0.5,
  })
  for (let i = 0; i < 7; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.5, 0.86), finMat)
    fin.position.x = -0.5 + i * 0.16
    group.add(fin)
  }
  return { group, core, material: mat }
}

/** Expansion valve — small grey cylinder, not health-critical. */
export function createExpansionValve() {
  const geo = new THREE.CylinderGeometry(0.1, 0.1, 0.15, 16)
  const mat = makeHealthMaterial(0x64748b)
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(0, -0.5, 0)
  mesh.name = 'valve'
  return { mesh, material: mat }
}

/**
 * Build the closed refrigerant circuit as an ordered list of waypoints
 * (compressor → condenser → evaporator → valve → compressor) plus a
 * CatmullRom curve so particles can be sampled at any 0–1 progress.
 */
export function buildCircuitPath(positions) {
  const pts = [
    positions.compressor.clone(),
    positions.condenser.clone(),
    positions.evaporator.clone(),
    positions.valve.clone(),
    positions.compressor.clone(), // close the loop
  ]
  const curve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.2)
  return curve
}

/** Static dark-grey pipes following the circuit curve. */
export function createPipes(curve) {
  const geo = new THREE.TubeGeometry(curve, 120, 0.05, 12, true)
  const mat = new THREE.MeshStandardMaterial({
    color: 0x404040, metalness: 0.8, roughness: 0.4,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.name = 'pipes'
  return mesh
}

/**
 * Refrigerant flow as a single Points cloud (one BufferGeometry, updated
 * in place each frame — no per-frame allocation, per the perf spec).
 */
export function createRefrigerantFlow(count = 140) {
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const mat = new THREE.PointsMaterial({
    size: 0.07,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
  })
  const points = new THREE.Points(geo, mat)
  points.name = 'refrigerant'
  points.frustumCulled = false
  return { points, geometry: geo, count }
}
