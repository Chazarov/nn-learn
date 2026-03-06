from typing import Any, Dict, List

from celery_app import celery_app

from nn_logic.mathh.models import Sample
from nn_logic.training.activation import ActivationType
from nn_logic.loss import LossType

from container import project_service, csv_service, nn_service
from log import logger


def _csv_data_to_samples(data) -> List[Sample]:
    return [Sample(signs=row.signs_vector, class_marks=row.class_mark) for row in data.rows]


@celery_app.task
def train_perceptron_task(
    user_id: str,
    project_id: str,
    activation_type: str,
    softmax_use: bool,
    loss_type: str,
    epochs: int,
    learning_rate: float,
) -> Dict[str, Any]:
    p = project_service.get_project(user_id, project_id)
    samples_data = csv_service.get_data(p.csv_file_id, user_id)
    raw_samples = _csv_data_to_samples(samples_data)

    act = ActivationType(activation_type)
    lt = LossType(loss_type)

    nn_service.train(
        weights=p.nn_data.weights,
        samples=raw_samples,
        activation_type=act,
        loss_type=lt,
        softmax_use=softmax_use,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    loss = nn_service.compute_loss(
        weights=p.nn_data.weights,
        samples=raw_samples,
        activation_type=act,
        loss_type=lt,
        softmax_use=softmax_use,
    )

    img = nn_service.get_visualisation(p.nn_data.weights)
    image_id = project_service.save_image(user_id, p.id, img)
    project_service.update_weights(user_id, p.id, p.nn_data.weights)

    logger.info(f"Training completed: project={p.id}, epochs={epochs}, loss={loss:.6f}")

    return {
        "project": p.model_dump(),
        "image_id": image_id,
        "epochs": epochs,
        "loss": loss,
    }
