

from typing import Any, Dict


class DomainException(Exception):
    is_public: bool = False

    def __init__(self, message: str = "", is_public: bool = False):
        super().__init__(message)
        self.message=message
        self.is_public = is_public

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "is_public": self.is_public,
        }

