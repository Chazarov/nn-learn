# Multilayer Perceptron API

Base URL: `http://localhost:8000/api`

---

## Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/upload/csv` | Upload a CSV training file. Returns `file_id` |
| `POST` | `/learn/` | Train a perceptron on an uploaded CSV file. Returns `perceptrone_id` |
| `POST` | `/get_answer` | Run forward propagation on input vector using trained weights. Returns predicted class and confidences |
| `GET` | `/files` | List all uploaded CSV training files |
| `GET` | `/weights` | List all saved trained perceptron weights |

---

## POST `/upload/csv`

Upload a CSV dataset for training.

**CSV format:** first column — id, middle columns — features, last column — class label.

**Body:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | CSV file |

**Response:**
```json
{ "file_id": "<uuid>" }
```

---

## POST `/learn/`

Train a new perceptron on a previously uploaded CSV file.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `file_id` | string | ID returned by `/upload/csv` |
| `hidden_layers_architecture` | int[] | Sizes of hidden layers, e.g. `[6]` |
| `activation_type` | enum | `RELLU` or `SIGMOID` |
| `epochs` | int | Number of training epochs (default: 300) |
| `learning_rate` | float | Learning rate (default: 0.05) |

**Response:**
```json
{ "perceptrone_id": "<uuid>" }
```

---

## POST `/get_answer`

Classify an input vector using a trained perceptron.

**Body:** `application/json`

| Field | Type | Description |
|-------|------|-------------|
| `perceptrone_id` | string | ID returned by `/learn/` |
| `input_vector` | float[] | Input feature values, e.g. `[5.1, 3.5, 1.4, 0.2]` |
| `activation_type` | enum | `RELLU` or `SIGMOID` — must match the one used during training |

**Response:**
```json
{
  "predicted": "<class_name>",
  "confidences": { "<class_name>": 0.0 },
  "output": [0.0]
}
```

---

## GET `/files`

List all uploaded training CSV files.

**Response:**
```json
{ "files": [{ "id": "<uuid>", "name": "<filename>.csv" }] }
```

---

## GET `/weights`

List all saved trained perceptron weights.

**Response:**
```json
{ "files": [{ "id": "<uuid>", "name": "<filename>.json" }] }
```
