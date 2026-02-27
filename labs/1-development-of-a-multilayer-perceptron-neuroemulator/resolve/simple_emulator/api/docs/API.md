# Multilayer Perceptron API

Base URL: `http://localhost:8000/api`

---

## Сводная таблица эндпоинтов

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/csv/upload` | Загрузить CSV-файл с обучающей выборкой. Возвращает `file_id` |
| `GET`  | `/csv/` | Список всех загруженных CSV-файлов |
| `POST` | `/actions/init` | Инициализировать перцептрон случайными весами. Возвращает `perceptrone_id` и `image_id` |
| `POST` | `/actions/learn/` | Обучить перцептрон на загруженной выборке. Возвращает `perceptrone_id` и `image_id` |
| `POST` | `/actions/get_answer` | Классифицировать входной вектор с помощью обученного перцептрона |
| `GET`  | `/actions/weights` | Список всех сохранённых файлов весов |
| `GET`  | `/images/` | Список всех сохранённых изображений визуализации |
| `GET`  | `/images/{image_id}` | Получить изображение визуализации весов по id |

---

## CSV

### POST `/csv/upload`

Загрузить CSV-датасет для обучения.

**Формат CSV:** первый столбец — id, средние — признаки, последний — метка класса.

**Body:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | CSV-файл |

**Response:**
```json
{ "file_id": "<uuid>" }
```

---

### GET `/csv/`

Список всех загруженных CSV-файлов.

**Response:**
```json
{
  "files": [
    { "id": "<uuid>", "name": "<filename>.csv", "object_type": "file_csv" }
  ]
}
```

---

## Actions

### POST `/actions/init`

Инициализировать новый перцептрон случайными весами на основе архитектуры.
Сохраняет веса и создаёт снимок визуализации.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID файла из `/csv/upload` |
| `hidden_layers_architecture` | int[] | Размеры скрытых слоёв, например `[6]` или `[8, 4]` |

**Response:**
```json
{
  "perceptrone_id": "<uuid>",
  "image_id": "<uuid>"
}
```

**Errors:**
- `404` — файл `file_id` не найден

---

### POST `/actions/learn/`

Обучить перцептрон на загруженном CSV-файле.
После обучения обновляет файл весов и снимок визуализации.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID файла из `/csv/upload` |
| `perceptrone_id` | string | ID перцептрона из `/actions/init` |
| `activation_type` | enum | `RELLU` или `SIGMOID` |
| `epochs` | int | Количество эпох обучения |
| `learning_rate` | float | Скорость обучения |

**Response:**
```json
{
  "perceptrone_id": "<uuid>",
  "image_id": "<uuid>"
}
```

**Errors:**
- `404` — файл `file_id` или `perceptrone_id` не найден

---

### POST `/actions/get_answer`

Классифицировать входной вектор с помощью обученного перцептрона.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `perceptrone_id` | string | ID перцептрона из `/actions/init` или `/actions/learn/` |
| `input_vector` | float[] | Значения признаков, например `[5.1, 3.5, 1.4, 0.2]` |
| `activation_type` | enum | `RELLU` или `SIGMOID` — должен совпадать с использованным при обучении |

**Response:**
```json
{
  "predicted": "<class_name>",
  "confidences": { "<class_name>": 0.0 },
  "output": [0.0]
}
```

**Errors:**
- `404` — `perceptrone_id` не найден

---

### GET `/actions/weights`

Список всех сохранённых файлов весов перцептрона.

**Response:**
```json
{
  "files": [
    { "id": "<uuid>", "name": "<filename>.json", "object_type": "file_json" }
  ]
}
```

---

## Images

### GET `/images/`

Список всех сохранённых изображений визуализации весов.

**Response:**
```json
{
  "images": [
    { "id": "<uuid>", "name": "<uuid>.png", "object_type": "image_png" }
  ]
}
```

---

### GET `/images/{image_id}`

Получить изображение визуализации весов перцептрона по id.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `image_id` | string | ID изображения из `/actions/init` или `/actions/learn/` |

**Response:** PNG-изображение (`image/png`)

**Errors:**
- `404` — изображение не найдено
