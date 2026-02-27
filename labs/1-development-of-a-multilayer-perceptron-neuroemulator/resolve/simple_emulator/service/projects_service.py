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


    def __init__(self, image_repository:ImageRepository, 
                 weights_disk_repository:WeightsDiskRepository, 
                 projects_repository: ProjectsRepository):
        self.image_repository = image_repository
        self.weights_disk_repository = weights_disk_repository
        self.projects_repository = projects_repository



    def create(self, user_id:str, nn_data:NNData, csv_file_id:str) -> Project:
        try:
            project:Project = self.projects_repository.create(user_id, csv_file_id)
        except DomainException:
            raise
        except Exception as e:
            # Обработка и логгирование возможных  ошибок (если нужно, если нет - просто оставить обертку в internal  server error и удалить этот комментарий)
            # Во всех местах проекта ошибки должны обрабатвыаться строго по этой форме!!!
            logger.error(f"error while user creating: {e}")
            traceback.print_exc()
            raise InternalServerException()
             #Оставляем пустым сообщение ошибки т.к клиенту незачем знать о причинах внутренней ошибки. А нам - НАДО!!! Поэтому нужно логгировать ошибку но только
             # один раз - в месте , где она появилась для этого и нужен такой вормат обработки. Удали этот комментарий когда закончишь
                                             
        
        try:
            self.weights_disk_repository.create(project.id, nn_data=nn_data)
        except DomainException:
            self.projects_repository.delete(user_id, project.id)
            raise
        except Exception as e:
            self.projects_repository.delete(user_id, project.id)
            # Обработка и логгирование возможных  ошибок (если нужно, если нет - просто оставить обертку в internal  server error и удалить этот комментарий)
            # Во всех местах проекта ошибки должны обрабатвыаться строго по этой форме!!! Все внутренние обработанные ошибки пробрасываем далее. А новые обрабатываем и оборачиваем во внутреннюю ошибку , чтобы пропустить ее дальше
            logger.error(f"error while user creating: {e}")
            traceback.print_exc()
            raise InternalServerException()
            #Оставляем пустым сообщение ошибки т.к клиенту незачем знать о причинах внутренней ошибки. А нам - НАДО!!! Поэтому нужно логгировать ошибку но только
             # один раз - в месте , где она появилась для этого и нужен такой вормат обработки. Удали этот комментарий когда закончишь
        


        return project

    def get_project(self, user_id:str, id:str) -> ProjectWithData:
        project_info = self.projects_repository.get_by_id(id=id, user_id=user_id)
        project_data = self.weights_disk_repository.get_by_id(id = user_id)
        return ProjectWithData(id = project_info.id, user_id=user_id, created_at= project_info.created_at, csv_file_id=project_info.csv_file_id, nn_data=project_data)
    
    def update_weights(self, user_id:str, id:str, weights:List[List[List[float]]]):
        ...
    
    
        

    def get_projects(self, user_id:str) -> List[Project]:
        return self.projects_repository.get_all(user_id)

    def get_image(self, project_id:str, user_id:str) -> str:

        self.get_project(user_id, project_id)
        return self.image_repository.get_image(project_id)
    
    def save_image(self, user_id:str, project_id:str, image:npt.NDArray[np.uint8]) -> str:
        return self.image_repository.save_image(project_id, image)
    


    