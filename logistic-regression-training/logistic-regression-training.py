import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w=np.zeros(X.shape[1])
    b=0.0

    for i in range(steps):
        dw=0
        db=0
        for j in range(len(X)):
            z= X[j]@w.T+b
            y_pred=_sigmoid(z)
            dw+=(y[j]-y_pred)*X[j]
            db+=(y[j]-y_pred)
        w+=lr*dw/len(X)
        b+=lr*db/len(X)
    return (w,b)  
            
            
    