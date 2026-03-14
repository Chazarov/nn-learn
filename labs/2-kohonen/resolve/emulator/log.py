from typing import Any, Dict
from loguru import logger
import sys


def format_record(record: Dict[str, Any]):
    module_path = record["name"].replace(".", "\\")
    record["extra"]["custom_path"] = module_path
    return "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[custom_path]}:{line} - {message}\n{exception}"


logger.remove()
logger.add(
    sys.stderr,
    format=format_record,# type: ignore[reportCallIssue]
    level="DEBUG"
)