/**
 * Health → colour mapping for the 3D digital-twin scene.
 *
 * A component's health (0–100 %) maps to a traffic-light scheme:
 *   > 66 %  → green   (healthy)
 *   33–66 % → amber   (degrading)
 *   < 33 %  → red     (critical)
 *
 * Colours are returned as 24-bit hex integers (0xRRGGBB) because THREE.js
 * `Color.setHex()` / material constructors expect that form, not CSS strings.
 */

export const HEALTH_GREEN = 0x22c55e
export const HEALTH_AMBER = 0xf59e0b
export const HEALTH_RED   = 0xef4444

/** Discrete traffic-light colour for a 0–100 health value. */
export function healthToHex(pct) {
  const p = clamp01(pct / 100)
  if (p > 0.66) return HEALTH_GREEN
  if (p > 0.33) return HEALTH_AMBER
  return HEALTH_RED
}

/**
 * Smoothly interpolated colour (green→amber→red) for a 0–100 health value.
 * Nicer for emissive glow than the hard discrete steps.
 * Returns { r, g, b } in 0–1 range (THREE.Color.setRGB form).
 */
export function healthToRGB(pct) {
  const p = clamp01(pct / 100)
  // 0.0 red → 0.5 amber → 1.0 green
  let from, to, t
  if (p < 0.5) {
    from = hexToRgb(HEALTH_RED)
    to   = hexToRgb(HEALTH_AMBER)
    t    = p / 0.5
  } else {
    from = hexToRgb(HEALTH_AMBER)
    to   = hexToRgb(HEALTH_GREEN)
    t    = (p - 0.5) / 0.5
  }
  return {
    r: from.r + (to.r - from.r) * t,
    g: from.g + (to.g - from.g) * t,
    b: from.b + (to.b - from.b) * t,
  }
}

/** Emissive glow intensity (0–0.8) — brighter when healthier. */
export function healthToGlow(pct) {
  return clamp01(pct / 100) * 0.8
}

/** 'healthy' | 'degrading' | 'critical' label for a 0–100 value. */
export function healthStatus(pct) {
  const p = clamp01(pct / 100)
  if (p > 0.66) return 'healthy'
  if (p > 0.33) return 'degrading'
  return 'critical'
}

// ── helpers ──────────────────────────────────────────────────────────────────
export function clamp01(v) {
  if (Number.isNaN(v) || v == null) return 0
  return Math.max(0, Math.min(1, v))
}

export function hexToRgb(hex) {
  return {
    r: ((hex >> 16) & 0xff) / 255,
    g: ((hex >> 8) & 0xff) / 255,
    b: (hex & 0xff) / 255,
  }
}
