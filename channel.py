import numpy as np
from constants import CODEWORDS_NBC, CODEWORDS_FBC

def apply_channel_errors(indices, p_error, code_type='FBC', ep_level=0):
    if p_error <= 0.0:
        return indices
        
    code = CODEWORDS_FBC if code_type == 'FBC' else CODEWORDS_NBC
    n_samples = len(indices)
    unprotected_bits = 3 - ep_level
    
    error_mask = np.random.rand(n_samples, unprotected_bits) < p_error
    out_indices = np.zeros_like(indices)
    
    for i in range(n_samples):
        cw = list(code[indices[i]])
        for b in range(ep_level, 3):
            if error_mask[i, b - ep_level]:
                cw[b] = 1 - cw[b]
        
        # Decode back to index
        for j, c in enumerate(code):
            if c == cw:
                out_indices[i] = j
                break
                
    return out_indices
