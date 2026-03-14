from models.project import NNData

class KohonenDiskRepo:
    # Сохранение сети кохонена на диск с использованием стандартных инструментов numpy. Файл сохраняется по пути dir именуется своим id + .npz. 
    # Все поля NNData 
    # должны быть сохранены
    def __init__(self, directory:str):
        self.dir = directory

    def create(self, id:str, nn_data:NNData):
        pass

    def delete(self, id: str) -> None:
        """ Если не найдено то производит стандартную обработку и логгирование ошибки.
         Такая обработка ошибок обязательна во всем приложении.
          1) Все ошибки должны быть обработаны и обернуты в DomainExceptions. Если ошибка уже обработана то она пробрасывается наверх, если его не
           нужно обработать , отреагировать каким то поведением.  
           если ошибка не опознана она обертывается в InternalServerException. Ошибка логгируется только один раз - при возникновении после пробрасывается или 
           обрабатывается. Пример:
           

            try:
                with self.session_factory() as session:
                    db_user = UserDB(password_hash=password_hash, email=email, name=name)
                    session.add(db_user)
                    session.commit()
                    session.refresh(db_user)
                    return User(id=db_user.id, password_hash=db_user.password_hash,
                                name=db_user.name, created_at=db_user.created_at, email=db_user.email)
            except DomainException:
                raise
            except IntegrityError as e:
                logger.error(f"user with same data already exists: {e}")
                traceback.print_exc()
                raise AlreadyExists()
            except Exception as e:
                logger.error(f"error while creating user: {e}")
                traceback.print_exc()
                raise InternalServerException()
           """
        pass
    
    def get_by_id(self, id:str) -> NNData:
        pass