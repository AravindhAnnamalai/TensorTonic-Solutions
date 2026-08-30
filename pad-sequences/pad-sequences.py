import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([],dtype=int).reshape(0,0)
    if max_len is None:
        max_len=max(len(seq) for seq in seqs)
    padded=[]    
    for seq in seqs:
            if len(seq)<max_len:
                seq=seq+[pad_value for _ in range(max_len-len(seq))]
                padded.append(seq)
                continue
            else:
                padded.append(seq[:max_len])
                
                
    return np.array(padded)    