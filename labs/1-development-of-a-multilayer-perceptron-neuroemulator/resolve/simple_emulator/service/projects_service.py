import traceback
from typing import List
import numpy.typing as npt
import numpy as np

from models.progect_nn import Project, NNData, ProjectWithData
from repository.image_disk_repository import ImageRepository
from repository.weights_disk_repository import WeightsDiskRepository
from repository.weights_postgres_repository import ProjectsRepository
from exceptions.domain import DomainException
from exceptions.internal_server_exception import InternalServerException
from log import logger


class ProjectsService:

    def __init__(self, image_repository: ImageRepository,
                 weights_disk_repository: WeightsDiskRepository,
                 projects_repository: ProjectsRepository):
        self.image_repository = image_repository
        self.weights_disk_repository = weights_disk_repository
        self.projects_repository = projects_repository

    def create(self, user_id: str, nn_data: NNData, csv_file_id: str) -> Project:
        try:
            project: Project = self.projects_repository.create(user_id, csv_file_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while creating project record: {e}")
            traceback.print_exc()
            raise InternalServerException()

        try:
            self.weights_disk_repository.create(project.id, nn_data=nn_data)
        except DomainException:
            self.projects_repository.delete(user_id, project.id)
            raise
        except Exception as e:
            self.projects_repository.delete(user_id, project.id)
            logger.error(f"error while saving weights to disk: {e}")
            traceback.print_exc()
            raise InternalServerException()

        return project

    def get_project(self, user_id: str, id: str) -> ProjectWithData:
        try:
            project_info = self.projects_repository.get_by_id(id=id, user_id=user_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting project info: {e}")
            traceback.print_exc()
            raise InternalServerException()

        try:
            project_data = self.weights_disk_repository.get_by_id(id=project_info.id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting project weights: {e}")
            traceback.print_exc()
            raise InternalServerException()

        return ProjectWithData(
            id=project_info.id,
            user_id=user_id,
            created_at=project_info.created_at,
            csv_file_id=project_info.csv_file_id,
            nn_data=project_data,
        )

    def update_weights(self, user_id: str, id: str, weights: List[List[List[float]]]):
        try:
            project_info = self.projects_repository.get_by_id(id=id, user_id=user_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while getting project for update: {e}")
            traceback.print_exc()
            raise InternalServerException()

        try:
            existing_data = self.weights_disk_repository.get_by_id(id=project_info.id)
            updated_data = NNData(
                weights=weights,
                mins=existing_data.mins,
                maxs=existing_data.maxs,
                classes=existing_data.classes,
            )
            self.weights_disk_repository.delete(project_info.id)
            self.weights_disk_repository.create(project_info.id, nn_data=updated_data)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while updating weights: {e}")
            traceback.print_exc()
            raise InternalServerException()

    def get_projects(self, user_id: str) -> List[Project]:
        return self.projects_repository.get_all(user_id)

    def get_image(self, project_id: str, user_id: str) -> str:
        self.get_project(user_id, project_id)
        return self.image_repository.get_image(project_id)

    def save_image(self, user_id: str, project_id: str, image: npt.NDArray[np.uint8]) -> str:
        return self.image_repository.save_image(project_id, image)

    def delete_project(self, user_id: str, project_id: str):
        try:
            self.projects_repository.delete(user_id, project_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while deleting project record: {e}")
            traceback.print_exc()
            raise InternalServerException()

        try:
            self.weights_disk_repository.delete(project_id)
        except DomainException:
            raise
        except Exception as e:
            logger.error(f"error while deleting weights file: {e}")
            traceback.print_exc()
            raise InternalServerException()
