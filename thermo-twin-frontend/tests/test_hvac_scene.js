/**
 * Phase 5 — HVAC 3D scene tests.
 *
 * Run with vitest:
 *   npm i -D vitest
 *   npx vitest run tests/test_hvac_scene.js
 *
 * These cover the deterministic, GPU-free logic: colour mapping, geometry
 * construction, circuit path, and HVACScene guard clauses. The live render
 * loop / raycasting needs a real WebGL context (browser e2e), so those are
 * intentionally out of scope for unit tests.
 */

import { describe, it, expect } from 'vitest'
import * as THREE from 'three'

import {
  healthToHex, healthToRGB, healthToGlow, healthStatus,
  clamp01, hexToRgb, HEALTH_GREEN, HEALTH_AMBER, HEALTH_RED,
} from '../src/utils/colorUtils.js'

import {
  createCompressor, createCondenser, createEvaporator,
  createExpansionValve, buildCircuitPath, createRefrigerantFlow,
} from '../src/utils/hvacGeometries.js'

import { HVACScene } from '../src/utils/hvacScene.js'

// ── colorUtils ────────────────────────────────────────────────────────────────
describe('colorUtils', () => {
  it('1. healthToHex maps traffic-light bands correctly', () => {
    expect(healthToHex(100)).toBe(HEALTH_GREEN)
    expect(healthToHex(50)).toBe(HEALTH_AMBER)
    expect(healthToHex(10)).toBe(HEALTH_RED)
  })

  it('2. healthToGlow scales 0→0, 100→0.8 and clamps', () => {
    expect(healthToGlow(0)).toBeCloseTo(0)
    expect(healthToGlow(100)).toBeCloseTo(0.8)
    expect(healthToGlow(250)).toBeCloseTo(0.8) // clamped
  })

  it('3. healthToRGB interpolates red→amber→green', () => {
    const red = healthToRGB(0)
    const grn = healthToRGB(100)
    expect(red.r).toBeGreaterThan(red.g)        // red dominant
    expect(grn.g).toBeGreaterThan(grn.r)        // green dominant
  })

  it('4. healthStatus labels and clamp01/hexToRgb helpers', () => {
    expect(healthStatus(90)).toBe('healthy')
    expect(healthStatus(50)).toBe('degrading')
    expect(healthStatus(5)).toBe('critical')
    expect(clamp01(-1)).toBe(0)
    expect(clamp01(2)).toBe(1)
    expect(hexToRgb(0xffffff)).toEqual({ r: 1, g: 1, b: 1 })
  })
})

// ── hvacGeometries ────────────────────────────────────────────────────────────
describe('hvacGeometries', () => {
  it('5. component factories return named meshes at spec positions', () => {
    const c = createCompressor()
    expect(c.mesh).toBeInstanceOf(THREE.Mesh)
    expect(c.mesh.name).toBe('compressor')
    expect(c.mesh.position.x).toBe(-2)

    const cond = createCondenser()
    expect(cond.group.position.x).toBe(2)
    expect(cond.body.name).toBe('condenser')
    expect(cond.fan).toBeInstanceOf(THREE.Group)

    const e = createEvaporator()
    expect(e.group.position.y).toBe(1.0)
    expect(e.core.name).toBe('evaporator')

    const v = createExpansionValve()
    expect(v.mesh.name).toBe('valve')
    expect(v.mesh.position.y).toBe(-0.5)
  })

  it('6. health materials expose an emissive channel starting dark', () => {
    const { material } = createCompressor()
    expect(material.emissiveIntensity).toBe(0)
    expect(material.emissive).toBeInstanceOf(THREE.Color)
  })

  it('7. circuit path is a closed curve through 4 component centres', () => {
    const pos = {
      compressor: new THREE.Vector3(-2, 0, 0),
      condenser:  new THREE.Vector3(2, 0, 0),
      evaporator: new THREE.Vector3(0, 1, 0),
      valve:      new THREE.Vector3(0, -0.5, 0),
    }
    const curve = buildCircuitPath(pos)
    const start = curve.getPointAt(0)
    const end   = curve.getPointAt(1)
    expect(start.distanceTo(end)).toBeLessThan(1e-6) // closed loop
  })

  it('8. refrigerant flow allocates count*3 position & colour buffers', () => {
    const { geometry, count } = createRefrigerantFlow(140)
    expect(count).toBe(140)
    expect(geometry.attributes.position.array.length).toBe(140 * 3)
    expect(geometry.attributes.color.array.length).toBe(140 * 3)
  })
})

// ── HVACScene guard clauses (GPU-free) ────────────────────────────────────────
describe('HVACScene', () => {
  it('9. throws a clear error when constructed without a canvas', () => {
    expect(() => new HVACScene(null)).toThrow(/canvas/i)
  })
})
