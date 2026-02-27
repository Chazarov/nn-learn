from typing import List

from models.progect_nn import Project

class ProjectsRepository:

    def create(self, user_id: str, csv_file_id:str) -> Project:
        ...
    
    def delete(self, user_id:str, id:str):
        ...

    def get_by_id(self, user_id:str, id:str) -> Project:
        ...

    def get_all(self, user_id:str) -> List[Project]:
        ...