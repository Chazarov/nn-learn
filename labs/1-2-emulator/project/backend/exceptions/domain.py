

class DomainException(Exception):
    is_public: bool = False

    def __init__(self, message: str, is_public: bool = False):
        super().__init__(message)
        self.is_public = is_public

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "is_public": self.is_public,
        }

