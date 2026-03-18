import numpy as np
from scipy.stats import norm
from constants import (
    V_LEVELS, U_BOUNDS_THEORETICAL, SIGMA_X_SQ, SIGMA_Q_SQ, 
    CODEWORDS_NBC, CODEWORDS_FBC
)

def get_P_v():
    p_v = np.zeros(8)
    for i in range(8):
        p_v[i] = norm.cdf(U_BOUNDS_THEORETICAL[i+1]) - norm.cdf(U_BOUNDS_THEORETICAL[i])
    return p_v

def get_transition_matrix(p_error, code_type='FBC', ep_level=0):
    """
    ep_level: 0 (No protection), 1 (EP1: protect MSB), 2 (EP2: protect 2 MSBs)
    """
    code = CODEWORDS_FBC if code_type == 'FBC' else CODEWORDS_NBC
    Pc = np.zeros((8, 8))
    m = 3
    
    for i in range(8):
        for j in range(8):
            cw_i = code[i]
            cw_j = code[j]
            
            possible = True
            for b in range(ep_level):
                if cw_i[b] != cw_j[b]:
                    possible = False
                    break
                    
            if possible:
                unprotected_bits = m - ep_level
                diff = sum(cw_i[b] != cw_j[b] for b in range(ep_level, m))
                Pc[i, j] = (p_error ** diff) * ((1 - p_error) ** (unprotected_bits - diff))
    
    return Pc

def compute_theoretical_snr(p_error, code_type='FBC', ep_level=0):
    P_v = get_P_v()
    Pc = get_transition_matrix(p_error, code_type, ep_level)
    
    eps_c_sq = 0.0
    for i in range(8):
        for j in range(8):
            delta_sq = (V_LEVELS[i] - V_LEVELS[j]) ** 2
            eps_c_sq += P_v[i] * Pc[i, j] * delta_sq
            
    total_noise = SIGMA_Q_SQ + eps_c_sq
    return 10.0 * np.log10(SIGMA_X_SQ / total_noise)

if __name__ == "__main__":
    rates = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for p in rates:
        snr_nbc = compute_theoretical_snr(p, 'NBC', 0)
        snr_fbc = compute_theoretical_snr(p, 'FBC', 0)
        snr_fbc_ep1 = compute_theoretical_snr(p, 'FBC', 1)
        snr_fbc_ep2 = compute_theoretical_snr(p, 'FBC', 2)
        print(f"P={p:.3f} | NBC: {snr_nbc:.2f} | FBC: {snr_fbc:.2f} | FBC-EP1: {snr_fbc_ep1:.2f} | FBC-EP2: {snr_fbc_ep2:.2f}")

