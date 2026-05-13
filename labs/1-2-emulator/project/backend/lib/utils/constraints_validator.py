from models.constraints import NumConstraint
from exceptions.argument_exception import ArgumentException

from pydantic import AfterValidator

def num_constraint_validator(constraint: NumConstraint) -> AfterValidator:
    allowed = constraint.allowed_values
    lo, hi = constraint.min_value, constraint.max_value
    allowed_set = frozenset(allowed) if allowed is not None else None

    def _validate(n: int) -> int:
        if allowed_set is not None:
            if n not in allowed_set:
                raise ArgumentException(f"For {constraint.name} allowed values are {sorted(allowed_set)}")
            return n
        if n < lo or n > hi:
            raise ArgumentException(f"For {constraint.name} expected value is between {lo} and {hi}")
        return n

    return AfterValidator(_validate)
 