import json
from typing import Any, List, cast

from forwrdpropagation.forward_propagation import forward_propogation
from activation import Sigmoid

WEIGHTS_PATH = "data/weights.json"
SPECIES: List[str] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
FEATURES: List[str] = ["SepalLength (cm)", "SepalWidth (cm)", "PetalLength (cm)", "PetalWidth (cm)"]

with open(WEIGHTS_PATH) as f:
    saved: Any = json.load(f)

perceptron: List[List[List[float]]] = cast(List[List[List[float]]], saved["weights"])
mins: List[float] = cast(List[float], saved["mins"])
maxs: List[float] = cast(List[float], saved["maxs"])
activation = Sigmoid()

print("Iris classifier. Enter 4 features (Ctrl+C to exit):\n")

while True:
    x: List[float] = []
    for name in FEATURES:
        x.append(float(input(f"  {name}: ")))

    xn: List[float] = [(x[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(4)]
    outputs, _ = forward_propogation(xn, perceptron, activation)

    predicted: int = outputs.index(max(outputs))
    print(f"\nPrediction: {SPECIES[predicted]}")
    print("Confidences:")
    for i in range(3):
        print(f"  {SPECIES[i]}: {outputs[i]:.3f}")
    print()
