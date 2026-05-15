/**
 * Утилиты для ограничений из GET /api/public-constraints (зеркало backend config.PublicConstraints).
 * Используются для подсказок min/max и для приведения ввода к допустимым значениям на клиенте.
 */

/**
 * @param {{ min_value?: number, max_value?: number, allowed_values?: number[] | null }} c
 * @param {number} raw
 * @returns {number}
 */
export function clampIntByNumConstraint(c, raw) {
  if (!c) return Math.round(Number(raw)) || 0;
  const n = Math.round(Number(raw)) || 0;
  const allowed = c.allowed_values;
  if (Array.isArray(allowed) && allowed.length > 0) {
    const sorted = [...allowed].sort((a, b) => a - b);
    let best = sorted[0];
    let bestD = Math.abs(best - n);
    for (const v of sorted) {
      const d = Math.abs(v - n);
      if (d < bestD) {
        best = v;
        bestD = d;
      }
    }
    return best;
  }
  const lo = c.min_value ?? 0;
  const hi = c.max_value ?? Number.MAX_SAFE_INTEGER;
  return Math.min(hi, Math.max(lo, n));
}

/**
 * @param {{ min_value?: number, max_value?: number }} c
 * @param {number} raw
 * @returns {number}
 */
export function clampFloatByFloatConstraint(c, raw) {
  if (!c) return Number(raw);
  const x = Number(raw);
  if (!Number.isFinite(x)) return c.min_value ?? 0;
  const lo = c.min_value ?? -Infinity;
  const hi = c.max_value ?? Infinity;
  return Math.min(hi, Math.max(lo, x));
}

/**
 * @param {{ min_value?: number, max_value?: number, allowed_values?: number[] | null }} c
 * @returns {{ kind: 'range', min: number, max: number } | { kind: 'allowed', values: number[] }}
 */
export function describeNumConstraint(c) {
  if (!c) return { kind: "range", min: 0, max: 100_000 };
  if (Array.isArray(c.allowed_values) && c.allowed_values.length > 0) {
    return { kind: "allowed", values: [...c.allowed_values].sort((a, b) => a - b) };
  }
  return {
    kind: "range",
    min: c.min_value ?? 0,
    max: c.max_value ?? 100_000,
  };
}

/**
 * Атрибуты для <input type="number"> по NumConstraint (если только диапазон).
 * @param {{ min_value?: number, max_value?: number, allowed_values?: number[] | null }} c
 */
export function numberInputPropsFromNumConstraint(c) {
  const d = describeNumConstraint(c);
  if (d.kind === "allowed") {
    const vals = d.values;
    return {
      min: Math.min(...vals),
      max: Math.max(...vals),
      step: 1,
    };
  }
  return { min: d.min, max: d.max, step: 1 };
}

/**
 * @param {{ min_value?: number, max_value?: number }} c
 */
export function numberInputPropsFromFloatConstraint(c) {
  if (!c) return { min: 1e-8, max: 1, step: "any" };
  return { min: c.min_value, max: c.max_value, step: "any" };
}

/**
 * @param {Record<string, unknown>} constraints
 */
export function hiddenLayersConstraintBundle(constraints) {
  const hl = constraints?.PERCEPTRON_HIDDEN_LAYERS;
  const maxLayers = hl?.max_hidden_layers ?? 12;
  const neuron = hl?.neurons_per_hidden_layer ?? {
    min_value: 1,
    max_value: 512,
    allowed_values: null,
  };
  return { maxLayers, neuronConstraint: neuron };
}
