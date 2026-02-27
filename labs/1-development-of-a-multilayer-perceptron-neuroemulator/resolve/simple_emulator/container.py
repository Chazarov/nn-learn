from service.projects_service import ProjectsService
from service.auth_service import AuthService
from service.csv_service import CsvService

from repository.image_disk_repository import ImageRepository
from repository.weights_disk_repository import WeightsDiskRepository
from repository.weights_postgres_repository import ProjectsRepository
from repository.user_postgres_repository import UserRepository
from repository.csv_disk_repository import CsvDiskRepository
from repository.csv_postgres_repository import CSVRelativeRepository

from config import config


_image_repository = ImageRepository(directory=config.IMAGES_DIRECTORY)

_weights_disk_repository = WeightsDiskRepository(directory=config.WEIGHTS_DIRECTORY)
_weights_relational_repository = ProjectsRepository()
_user_repository = UserRepository()
_csv_disk_repo = CsvDiskRepository(directory=config.CSV_DIRECTORY)
_csv_relative_repo = CSVRelativeRepository()

project_service = ProjectsService(image_repository=_image_repository, 
                                  weights_disk_repository=_weights_disk_repository,
                                  projects_repository=_weights_relational_repository)
auth_service = AuthService(user_repository=_user_repository)
csv_service = CsvService(disk_repo=_csv_disk_repo, relative_repo=_csv_relative_repo)