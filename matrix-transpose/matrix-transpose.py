import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    n,m=len(A),len(A[0])

    transpose=np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            transpose[j,i]=A[i][j]
    return transpose        