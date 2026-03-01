import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)  # 🔥 convert list → array

    rows, cols = A.shape
    transpose_A = np.zeros((cols, rows), dtype=A.dtype)

    for i in range(rows):
        for j in range(cols):
            transpose_A[j, i] = A[i, j]

    return transpose_A