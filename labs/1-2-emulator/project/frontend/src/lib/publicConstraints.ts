/**
 * Работа с ответом GET /api/public-constraints: типы и функции,
 * которые ограничивают ввод числовых полей в соответствии с бэкендом.
 */
export type NumConstraint = {
  min_value: number;
  max_value: number;
  allowed_values: number[] | null;
};

export type FloatConstraint = {
  min_value: number;
  max_value: number;
};

export type HiddenLayersConstraint = {
  max_hidden_layers: number;
  neurons_per_hidden_layer: NumConstraint;
};

/** Сериализованный объект из ``build_public_constraints_json`` (backend). */
export type PublicConstraints = {
  KOHONEN_INPUT_FEATURES_MAX: number;
  KOHONEN_OUTPUT_LAYER_SIZE_RANGE: NumConstraint;
  KOHONEN_INPUT_LAYER_SIZE_RANGE: NumConstraint;
  KOHONEN_LEARN_EPOCHS_RANGE: NumConstraint;
  KOHONEN_LEARN_LEARNING_RATE_RANGE: FloatConstraint;
  KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE: FloatConstraint;
  KOHONEN_GET_ANSWER_INPUT_VECTOR_MAX_LEN: number;
  PERCEPTRON_HIDDEN_LAYERS: HiddenLayersConstraint;
  PERCEPTRON_LEARN_EPOCHS_RANGE: NumConstraint;
  PERCEPTRON_LEARN_LEARNING_RATE_RANGE: FloatConstraint;
  PERCEPTRON_GET_ANSWER_INPUT_VECTOR_MAX_LEN: number;
};

export type IntFieldBinding =
  | { kind: "range"; min: number; max: number }
  | { kind: "enum"; values: number[] };

export type FloatFieldBinding = { min: number; max: number };

/** Описание, как ограничить целочисленное поле по ``NumConstraint``. */
export function intFieldBindingFromNumConstraint(c: NumConstraint): IntFieldBinding {
  const allowed = c.allowed_values;
  if (allowed && allowed.length > 0) {
    return { kind: "enum", values: [...allowed].sort((a, b) => a - b) };
  }
  return { kind: "range", min: c.min_value, max: c.max_value };
}

/** Описание для ``FloatConstraint``. */
export function floatFieldBindingFromFloatConstraint(c: FloatConstraint): FloatFieldBinding {
  return { min: c.min_value, max: c.max_value };
}

/**
 * Приводит введённое целое к допустимому значению по ограничению бэкенда.
 * Используйте в ``onChange`` числовых полей (строка из input type="number" / text).
 */
export function coerceIntWithNumConstraint(
  raw: string,
  previous: number,
  c: NumConstraint,
): number {
  const trimmed = raw.trim();
  if (trimmed === "" || trimmed === "-") return previous;
  const n = Number.parseInt(trimmed, 10);
  if (!Number.isFinite(n)) return previous;

  const binding = intFieldBindingFromNumConstraint(c);
  if (binding.kind === "range") {
    return Math.min(binding.max, Math.max(binding.min, n));
  }
  const { values } = binding;
  return values.reduce(
    (best, x) => (Math.abs(x - n) < Math.abs(best - n) ? x : best),
    values[0]!,
  );
}

/** То же для полей с ``FloatConstraint``. */
export function coerceFloatWithFloatConstraint(
  raw: string,
  previous: number,
  c: FloatConstraint,
): number {
  const trimmed = raw.trim().replace(",", ".");
  if (trimmed === "" || trimmed === "-" || trimmed === "." || trimmed === "-.") return previous;
  const n = Number.parseFloat(trimmed);
  if (!Number.isFinite(n)) return previous;
  const { min, max } = floatFieldBindingFromFloatConstraint(c);
  return Math.min(max, Math.max(min, n));
}

/** Парсинг архитектуры скрытых слоёв из строки «64, 32» с учётом ``HiddenLayersConstraint``. */
export function parseHiddenLayersFromText(
  text: string,
  constraint: HiddenLayersConstraint,
): { ok: true; layers: number[] } | { ok: false; error: string } {
  const parts = text
    .split(/[,;\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length > constraint.max_hidden_layers) {
    return {
      ok: false,
      error: `Не больше ${constraint.max_hidden_layers} скрытых слоёв`,
    };
  }
  const nc = constraint.neurons_per_hidden_layer;
  const binding = intFieldBindingFromNumConstraint(nc);
  const layers: number[] = [];
  for (let i = 0; i < parts.length; i++) {
    const n = Number.parseInt(parts[i]!, 10);
    if (!Number.isFinite(n)) {
      return { ok: false, error: `Слой ${i + 1}: ожидается целое число` };
    }
    let v = n;
    if (binding.kind === "range") {
      if (v < binding.min || v > binding.max) {
        return {
          ok: false,
          error: `Слой ${i + 1}: допустимо ${binding.min}…${binding.max}`,
        };
      }
    } else if (!binding.values.includes(v)) {
      return {
        ok: false,
        error: `Слой ${i + 1}: допустимы только значения ${binding.values.join(", ")}`,
      };
    }
    layers.push(v);
  }
  return { ok: true, layers };
}

export async function fetchPublicConstraints(apiBase: string): Promise<PublicConstraints> {
  const url = `${apiBase.replace(/\/$/, "")}/public-constraints`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Не удалось загрузить ограничения: HTTP ${res.status}`);
  }
  return res.json() as Promise<PublicConstraints>;
}
