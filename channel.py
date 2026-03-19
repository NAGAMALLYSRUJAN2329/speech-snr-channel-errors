import numpy as np
from constants import CODEWORDS_NBC, CODEWORDS_FBC

def generate_burst_error_mask(total_bits, p_error, burst_length=20):
    if p_error <= 0.0 or p_error >= 0.5:
        return np.random.rand(total_bits) < p_error
        
    P_BG = 1.0 / burst_length
    P_B = 2.0 * p_error
    if P_B >= 1.0:
        P_B = 0.99
    P_GB = (P_B * P_BG) / (1.0 - P_B)
    
    mask = np.zeros(total_bits, dtype=bool)
    idx = 0
    state = 0 if np.random.rand() > P_B else 1
    
    while idx < total_bits:
        if state == 0:
            stay = np.random.geometric(P_GB)
            idx += stay
            state = 1
        else:
            stay = np.random.geometric(P_BG)
            end_idx = min(idx + stay, total_bits)
            mask[idx:end_idx] = np.random.rand(end_idx - idx) < 0.5
            idx += stay
            state = 0
            
    return mask

def apply_channel_errors(indices, p_error, code_type='FBC', ep_level=0, error_type='independent'):
    if p_error <= 0.0:
        return indices
        
    code = CODEWORDS_FBC if code_type == 'FBC' else CODEWORDS_NBC
    n_samples = len(indices)
    unprotected_bits = 3 - ep_level
    if unprotected_bits <= 0:
        return indices
        
    total_bits = n_samples * unprotected_bits
    
    if error_type == 'burst':
        flat_mask = generate_burst_error_mask(total_bits, p_error, burst_length=20)
        error_mask = flat_mask.reshape(n_samples, unprotected_bits)
    else:
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

