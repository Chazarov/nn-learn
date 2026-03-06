import copy
import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from nn_logic.models.activation import ActivationType
from nn_logic.training.activation.activation import ActivationType, ACTIVATIONS, SoftMax
from nn_logic.forwrdpropagation.forward_propagation import forward_propagation
from nn_logic.loss import LossType, LOSSES
from nn_logic.mathh.models import Sample, Perceptron
from nn_logic.mathh.mv import (
    apply_adjustments,
    init_perceptron as build_perceptron,
    min_max_samples_normalaize,
    min_max_signs_normalize,
)
from nn_logic.training.backpropagation import BackPropagation
from nn_logic.visualisation.visualisation import get_visualisation as _get_visualisation, ColorTheme


class NNService:

    def init_perceptron(
        self,
        architecture: List[int],
        samples: List[Sample],
    ) -> Tuple[List[List[List[float]]], List[float], List[float]]:
        """
        Возвращает (weights, mins, maxs).
        """
        weights = build_perceptron(architecture)

        signs_count = len(samples[0].signs)
        classes_count = len(samples[0].class_marks)
        _, mins, maxs = min_max_samples_normalaize(
            samples, signs_count=signs_count, classes_count=classes_count,
        )

        return weights, mins, maxs

    def train(
        self,
        weights: List[List[List[float]]],
        samples: List[Sample],
        activation_type: ActivationType,
        loss_type: LossType,
        softmax_use: bool,
        epochs: int,
        learning_rate: float,
    ) -> List[List[List[float]]]:
        """Обучает перцептрон (мутирует weights in-place)."""
        signs_count = len(samples[0].signs)
        classes_count = len(samples[0].class_marks)

        normalized_samples, _, _ = min_max_samples_normalaize(
            samples, signs_count=signs_count, classes_count=classes_count,
        )

        layers_count = len(weights) + 1
        activations = [ACTIVATIONS[activation_type]() for _ in range(layers_count - 1)]
        loss = LOSSES[loss_type]()

        if softmax_use:
            activations[-1] = SoftMax()

        perceptron = Perceptron(
            weights=weights, activations=activations, layers_count=layers_count,
        )
        bp = BackPropagation(loss, learning_rate, perceptron)

        best_loss = float("inf")
        best_weights: List[List[List[float]]] = copy.deepcopy(perceptron.weights)

        for _ in range(epochs):
            random.shuffle(normalized_samples)
            for sample in normalized_samples:
                outputs, weighted_sums = forward_propagation(sample.signs, perceptron)
                adjustments = bp.training_iteration_calculate(
                    sample.signs, outputs, sample.class_marks, weighted_sums,
                )
                perceptron.weights = apply_adjustments(weights, adjustments)

            epoch_loss = sum(
                loss.perform(s.class_marks, forward_propagation(s.signs, perceptron)[0])
                for s in normalized_samples
            ) / len(normalized_samples)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights = copy.deepcopy(perceptron.weights)

        for q in range(len(weights)):
            for i in range(len(weights[q])):
                for j in range(len(weights[q][i])):
                    weights[q][i][j] = best_weights[q][i][j]

        return best_weights

    def predict(
        self,
        weights: List[List[List[float]]],
        input_vector: List[float],
        activation_type: ActivationType,
        softmax_use: bool,
        mins: List[float],
        maxs: List[float],
        input_size: int,
    ) -> List[float]:
        """Возвращает выходной вектор перцептрона."""
        normalized_input = min_max_signs_normalize(
            input_vector, maxs=maxs, mins=mins, signs_count=input_size,
        )

        layers_count = len(weights) + 1
        activations = [ACTIVATIONS[activation_type]() for _ in range(layers_count - 1)]
        if softmax_use:
            activations[-1] = SoftMax()

        perceptron = Perceptron(
            weights=weights, activations=activations, layers_count=layers_count,
        )
        output_vector, _ = forward_propagation(normalized_input, perceptron)
        return output_vector

    def compute_loss(
        self,
        weights: List[List[List[float]]],
        samples: List[Sample],
        activation_type: ActivationType,
        loss_type: LossType,
        softmax_use: bool,
    ) -> float:
        """Средний loss по всем samples на текущих весах."""
        signs_count = len(samples[0].signs)
        classes_count = len(samples[0].class_marks)

        normalized_samples, _, _ = min_max_samples_normalaize(
            samples, signs_count=signs_count, classes_count=classes_count,
        )

        layers_count = len(weights) + 1
        activations = [ACTIVATIONS[activation_type]() for _ in range(layers_count - 1)]
        if softmax_use:
            activations[-1] = SoftMax()

        perceptron = Perceptron(
            weights=weights, activations=activations, layers_count=layers_count,
        )

        loss_fn = LOSSES[loss_type]()
        total_loss = 0.0
        for sample in normalized_samples:
            outputs, _ = forward_propagation(sample.signs, perceptron)
            total_loss += loss_fn.perform(sample.class_marks, outputs)

        return total_loss / len(normalized_samples)

    def get_visualisation(
        self,
        weights: List[List[List[float]]],
        color_theme: ColorTheme = ColorTheme.DARK,
    ) -> npt.NDArray[np.uint8]:
        return _get_visualisation(weights, color_theme)
