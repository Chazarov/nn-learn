import csv
import json
import random
from typing import List, Tuple

from nn_logic.forwrdpropagation.forward_propagation import forward_propogation
from nn_logic.training.backpropagation import BackPropagation
from nn_logic.activation import Sigmoid
from nn_logic.loss import MSE
from nn_logic.mathh.mv import normalize, apply_adjustments

WEIGHTS_PATH = "data/weights.json"
EPOCHS = 500
LEARNING_RATE = 0.05
SPECIES: List[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

Sample = Tuple[List[float], List[float]]


def load_data() -> List[Sample]:
    rows: List[Sample] = []
    with open("iris-plants/sourse/Iris.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x: List[float] = [
                float(row["SepalLengthCm"]),
                float(row["SepalWidthCm"]),
                float(row["PetalLengthCm"]),
                float(row["PetalWidthCm"]),
            ]
            label: int = SPECIES.index(str(row["Species"]))
            y: List[float] = [1.0 if i == label else 0.0 for i in range(3)]
            rows.append((x, y))
    return rows





def init_perceptron() -> List[List[List[float]]]:
    def layer(n_out: int, n_in: int) -> List[List[float]]:
        return [[random.uniform(-0.5, 0.5) for _ in range(n_in)] for _ in range(n_out)]
    return [layer(6, 4), layer(3, 6)]





raw_data: List[Sample] = load_data()
data, mins, maxs = normalize(raw_data)
random.shuffle(data)

perceptron: List[List[List[float]]] = init_perceptron()
activation = Sigmoid()
bp = BackPropagation(MSE(), LEARNING_RATE, perceptron, activation)

for epoch in range(EPOCHS):
    random.shuffle(data)
    total_loss: float = 0.0
    for x, y in data:
        outputs, weighted_sums = forward_propogation(x, perceptron, activation)
        adjustments = bp.training_iteration_calculate(x, outputs, y, weighted_sums)
        apply_adjustments(perceptron, adjustments)
        total_loss += sum((y[i] - outputs[i]) ** 2 for i in range(len(y)))
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss={total_loss / len(data):.4f}")

with open(WEIGHTS_PATH, "w") as f:
    json.dump({"weights": perceptron, "mins": mins, "maxs": maxs}, f)

print(f"Weights saved to {WEIGHTS_PATH}")
