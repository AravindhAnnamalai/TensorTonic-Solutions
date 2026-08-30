import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a_norm=np.linalg.norm(a)
    b_norm=np.linalg.norm(b)
    if a_norm and b_norm:
        cos= np.array(a).dot(np.array(b))/(a_norm*b_norm)
        return float(cos)
    return 0.0
    
    