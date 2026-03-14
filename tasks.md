## Реализация сервиса

Задачи:
1) реализовать сервис Kohonen_service для проведения операций с сетями кохонена. Инкапсулируя пакет nn_core от всего остального приложения
2) Реализовать алгоритмы визуализации сети кохонена в nn_core/visualisation/__init__.py
3) Реализовать repo/kohonen_disk_repo.py для хранения весов сетей кохонена по id на диске. Этот репозиторий ничего не знает про user-ов. Он работает только с id


Требования к выполнению и структуре:
1) Соблюдать структуру обработки ошибок: Все ошибки должны быть обработаны и обернуты в DomainExceptions. Если ошибка уже обработана то она пробрасывается наверх, если его не нужно обработать , отреагировать каким то поведением. Если ошибка не ожидалась она обертывается в InternalServerException. Ошибка логгируется только один раз - при возникновении после пробрасывается или 
обрабатывается. Пример:

```
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
```
3) Не писать документацию
3) Не оставлять комментариев
