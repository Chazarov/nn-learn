from typing import List, Tuple
from random import uniform


from exceptions.argument_exception import ArgumentException
from log import logger


def m_v_mtpc(m: List[List[float]], v: List[float]) -> List[float]:
    """
    Matrix-vector multiplication: m (n x d) × v (d x 1) = result (n x 1)
    """
    if not m or not v:
        raise ArgumentException("Matrix or vector is empty")
    
    if len(m[0]) != len(v):
        e_str = f"Columns in matrix ({len(m[0])}) must equal vector length ({len(v)})"
        logger.error(e_str)
        raise ArgumentException(e_str)
    
    result: List[float] = []
    for i in range(len(m)):
        row_result = sum(m[i][k] * v[k] for k in range(len(v)))
        result.append(row_result)
    
    return result

def v_v_elementwise(v1: List[float], v2: List[float]) -> List[float]:
    """
    Element-wise vector multiplication (Hadamard product)
    Multiplies corresponding elements of two vectors.
        
    Returns:
        Vector where each element is v1[i] * v2[i]
        
    Raises:
        ArgumentException: If vectors have different lengths
        
    Example:
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        result = [4, 10, 18]
    """
    if len(v1) != len(v2):
        e_str = f"Vectors must have equal length: {len(v1)} != {len(v2)}"
        logger.error(e_str)
        raise ArgumentException(e_str)
    
    result: List[float] = [v1[i] * v2[i] for i in range(len(v1))]
    return result

def t_mtx(matrix: List[List[float]]) -> List[List[float]]:
    """
    Matrix transpose: swaps rows with columns
    
    Converts matrix of shape (m, n) to (n, m)
    
    Args:
        matrix: Input matrix represented as list of lists
        
    Returns:
        Transposed matrix
        
    Raises:
        ArgumentException: If matrix is empty or rows have different lengths
        
    Example:
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = [[1, 4], [2, 5], [3, 6]]
    """
    if not matrix or not matrix[0]:
        e_str = "Matrix cannot be empty"
        logger.error(e_str)
        raise ArgumentException(e_str)
    
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # Check all rows have same length
    for row in matrix:
        if len(row) != num_cols:
            e_str = f"All rows must have equal length: expected {num_cols}, got {len(row)}"
            logger.error(e_str)
            raise ArgumentException(e_str)
    
    # Create transposed matrix: columns become rows
    result: List[List[float]] = []
    for col in range(num_cols):
        new_row: List[float] = [matrix[row][col] for row in range(num_rows)]
        result.append(new_row)
    
    return result



def init_perceptron(architecture: List[int]) -> List[List[List[float]]]:
    perceptron: List[List[List[float]]] = list()
    for i in range(len(architecture) - 1):
        n_in = architecture[i]
        n_out = architecture[i + 1]
        perceptron.append(get_random_matrix(n_out, n_in))
    return perceptron



def get_random_matrix(n_out: int, n_in: int) -> List[List[float]]:
    return [[uniform(-1, 1) for _ in range(n_in)] for _ in range(n_out)]


Sample = Tuple[List[float], List[float]]
# Where is x - vector of signs, y - class mark

def normalize(data: List[Sample]) -> Tuple[List[Sample], List[float], List[float]]:
    n: int = len(data[0][0])
    mins: List[float] = [min(r[0][i] for r in data) for i in range(n)]
    maxs: List[float] = [max(r[0][i] for r in data) for i in range(n)]
    normed: List[Sample] = [
        ([(x[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(n)], y)
        for x, y in data
    ]

    # normalized data in 0 - 1 range. 
    # mins - minimal values for signs.
    # If  there were 7 signs, then the lengths of the mins and maxs arrays will be 7.
    return normed, mins, maxs
    

def apply_adjustments(
    perceptron: List[List[List[float]]],
    adjustments: List[List[List[float]]],
) -> None:
    for q in range(len(perceptron)):
        for i in range(len(perceptron[q])):
            for j in range(len(perceptron[q][i])):
                perceptron[q][i][j] += adjustments[q][i][j]