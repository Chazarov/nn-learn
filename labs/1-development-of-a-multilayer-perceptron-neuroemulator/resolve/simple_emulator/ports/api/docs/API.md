# Multilayer Perceptron API

Base URL: `http://localhost:8000/api`

Все эндпоинты, кроме `/auth/sign-up` и `/auth/login`, требуют авторизации через Bearer-токен:

```
Authorization: Bearer <token>
```

---

## Сводная таблица эндпоинтов

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| `POST` | `/auth/sign-up` | — | Регистрация нового пользователя |
| `POST` | `/auth/login` | — | Вход в систему, получение токена |
| `POST` | `/csv/upload` | Bearer | Загрузить CSV-файл с обучающей выборкой |
| `GET` | `/csv/` | Bearer | Список CSV-файлов текущего пользователя |
| `GET` | `/csv/{file_id}` | Bearer | Скачать CSV-файл по id |
| `DELETE` | `/csv/{file_id}` | Bearer | Удалить CSV-файл по id |
| `POST` | `/actions/init` | Bearer | Инициализировать перцептрон случайными весами |
| `POST` | `/actions/learn/` | Bearer | Обучить перцептрон |
| `POST` | `/actions/get_answer` | Bearer | Классифицировать входной вектор |
| `GET` | `/actions/projects` | Bearer | Список проектов текущего пользователя |
| `GET` | `/actions/project/{project_id}` | Bearer | Получить все данные проекта (кроме весов) |
| `DELETE` | `/actions/projects/{project_id}` | Bearer | Удалить проект по id |
| `GET` | `/images/{image_id}` | Bearer | Получить изображение визуализации весов |

---

## Auth

### POST `/auth/sign-up`

Регистрация нового пользователя. Возвращает JWT-токен.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | Email пользователя |
| `name` | string | Имя пользователя (уникальное) |
| `password` | string | Пароль |

**Response:**
```json
{ "token": "<jwt>" }
```

**Errors:**
- `403` — пользователь с таким email или именем уже существует

---

### POST `/auth/login`

Вход в систему. Возвращает JWT-токен.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | Email пользователя |
| `password` | string | Пароль |

**Response:**
```json
{ "token": "<jwt>" }
```

**Errors:**
- `401` — неверный пароль
- `404` — пользователь не найден

---

## CSV

### POST `/csv/upload`

Загрузить CSV-датасет для обучения.

**Формат CSV:** первый столбец — id, средние — признаки, последний — метка класса.

**Body:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | CSV-файл (расширение `.csv` обязательно) |

**Response:**
```json
{
  "id": "<uuid>",
  "user_id": "<uuid>",
  "name": "filename.csv",
  "created_at": 1700000000,
  "is_sample": false
}
```

**Errors:**
- `400` — файл не является CSV
- `401` — невалидный или просроченный токен

---

### GET `/csv/`

Список всех CSV-файлов текущего пользователя.

**Response:**
```json
{
  "files": [
    {
      "id": "<uuid>",
      "user_id": "<uuid>",
      "name": "filename.csv",
      "created_at": 1700000000,
      "is_sample": false
    }
  ]
}
```

---

### GET `/csv/{file_id}`

Скачать CSV-файл по id.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID файла из `GET /csv/` |

**Response:** CSV-файл (`text/csv`) с заголовком `Content-Disposition` для скачивания.

**Errors:**
- `401` — невалидный или просроченный токен
- `403` — доступ запрещён
- `404` — файл не найден или не принадлежит пользователю

---

### DELETE `/csv/{file_id}`

Удалить загруженный CSV-файл по id.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID файла из `POST /csv/upload` |

**Response:**
```json
{ "deleted": "<uuid>" }
```

**Errors:**
- `401` — невалидный или просроченный токен
- `403` — нельзя удалить образец (sample)
- `404` — файл не найден или не принадлежит пользователю

---

## Actions

### POST `/actions/init`

Инициализировать новый перцептрон случайными весами на основе архитектуры.
Создаёт проект, сохраняет веса и снимок визуализации. В ответе — все данные проекта (кроме весов) и `image_id`.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID файла из `POST /csv/upload` |
| `hidden_layers_architecture` | int[] | Размеры скрытых слоёв, например `[6]` или `[8, 4]` |

**Response:**
```json
{
  "project": {
    "id": "<uuid>",
    "user_id": "<uuid>",
    "csv_file_id": "<uuid>",
    "created_at": 1700000000,
    "nn_data": {
      "input_size": 4,
      "mins": [4.3, 2.0, 1.0, 0.1],
      "maxs": [7.9, 4.4, 6.9, 2.5],
      "classes": ["setosa", "versicolor", "virginica"]
    }
  },
  "image_id": "<uuid>"
}
```

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — файл `file_id` не найден

---

### POST `/actions/learn/`

Обучить перцептрон на CSV-файле, связанном с проектом.
После обучения обновляет веса и снимок визуализации.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | string | ID проекта из `POST /actions/init` |
| `activation_type` | enum | `RELLU` или `SIGMOID` |
| `epochs` | int | Количество эпох обучения |
| `learning_rate` | float | Скорость обучения |

**Response:**
```json
{
  "project_id": "<uuid>",
  "image_id": "<uuid>"
}
```

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — проект `project_id` не найден

---

### POST `/actions/get_answer`

Классифицировать входной вектор с помощью обученного перцептрона.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `perceptrone_id` | string | ID проекта из `POST /actions/init` |
| `input_vector` | float[] | Значения признаков, например `[5.1, 3.5, 1.4, 0.2]` |
| `activation_type` | enum | `RELLU` или `SIGMOID` — должен совпадать с использованным при обучении |

**Response:**
```json
{
  "predicted": "<class_name>",
  "confidences": { "<class_name>": 0.9312 },
  "output": [0.9312, 0.0512, 0.0176]
}
```

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — проект не найден

---

### GET `/actions/projects`

Список всех проектов текущего пользователя.

**Response:**
```json
{
  "projects": [
    {
      "id": "<uuid>",
      "user_id": "<uuid>",
      "csv_file_id": "<uuid>",
      "created_at": 1700000000
    }
  ]
}
```

---

### GET `/actions/project/{project_id}`

Получить все данные одного проекта (метаданные и параметры нейросети), кроме весов. Веса в ответ не включаются.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `project_id` | string | ID проекта из `POST /actions/init` |

**Response:**
```json
{
  "project": {
    "id": "<uuid>",
    "user_id": "<uuid>",
    "csv_file_id": "<uuid>",
    "created_at": 1700000000,
    "nn_data": {
      "input_size": 4,
      "mins": [4.3, 2.0, 1.0, 0.1],
      "maxs": [7.9, 4.4, 6.9, 2.5],
      "classes": ["setosa", "versicolor", "virginica"]
    }
  }
}
```

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — проект не найден

---

### DELETE `/actions/projects/{project_id}`

Удалить проект и связанный файл весов по id.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `project_id` | string | ID проекта из `POST /actions/init` |

**Response:**
```json
{ "deleted": "<uuid>" }
```

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — проект не найден

---

## Images

### GET `/images/{image_id}`

Получить изображение визуализации весов перцептрона по id.
`image_id` совпадает с `id` проекта.

**Path параметры:**

| Param | Type | Description |
|-------|------|-------------|
| `image_id` | string | ID проекта из `POST /actions/init` или `POST /actions/learn/` |

**Response:** PNG-изображение (`image/png`)

**Errors:**
- `401` — невалидный или просроченный токен
- `404` — изображение не найдено
