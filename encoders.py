import numpy as np
import scipy.linalg
from constants import V_LEVELS, U_BOUNDS_ENCODER

def quantize_gaussian(val):
    for i, u in enumerate(U_BOUNDS_ENCODER):
        if val <= u:
            return i
    return 7

def pcm_encode(signal, var=1.0):
    std = np.sqrt(var)
    indices = np.zeros(len(signal), dtype=int)
    for i, x in enumerate(signal):
        norm_x = x / std
        indices[i] = quantize_gaussian(norm_x)
    return indices

def pcm_decode(indices, var=1.0):
    std = np.sqrt(var)
    out = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        out[i] = V_LEVELS[idx] * std
    return out

def pcm_aqf_encode(signal):
    block_size = 32
    indices = np.zeros(len(signal), dtype=int)
    gains = np.zeros(len(signal) // block_size + 1)
    
    for b in range(0, len(signal), block_size):
        block = signal[b:b+block_size]
        # Max value mapped to standard deviation
        max_val = np.max(np.abs(block))
        g = max_val / 2.5 if max_val > 1e-4 else 1e-4   
        gains[b // block_size] = g
        
        for i, x in enumerate(block):
            indices[b+i] = quantize_gaussian(x / g)

    return indices, gains

def pcm_aqf_decode(indices, gains):
    block_size = 32
    out = np.zeros(len(indices))
    
    for b in range(0, len(indices), block_size):
        g = gains[b // block_size]
        for i in range(min(block_size, len(indices) - b)):
            out[b+i] = V_LEVELS[indices[b+i]] * g
            
    return out

def get_acf(x, max_lag):
    acf = np.correlate(x, x, mode='full')
    acf = acf[len(acf)//2:]
    return acf[:max_lag+1]

def solve_yule_walker(acf, order):
    if acf[0] == 0:
        return np.zeros(order)
    R = scipy.linalg.toeplitz(acf[:order])
    r = acf[1:order+1]
    try:
        h = scipy.linalg.solve(R, r)
    except scipy.linalg.LinAlgError:
        h = np.zeros(order)
    return h

def dpcm1_aqf_encode_decode(signal, h1=0.85, channel_func=None):
    block_size = 32
    indices = np.zeros(len(signal), dtype=int)
    gains = np.zeros(len(signal) // block_size + 1)
    
    for b in range(0, len(signal), block_size):
        block = signal[b:b+block_size]
        diffs = [block[0]] + [block[i] - h1 * block[i-1] for i in range(1, len(block))]
        max_val = np.max(np.abs(diffs))
        gains[b // block_size] = max_val / 2.5 if max_val > 1e-4 else 1e-4
        
    enc_state = 0.0
    for b in range(0, len(signal), block_size):
        g = gains[b // block_size]
        for i in range(min(block_size, len(signal) - b)):
            x = signal[b+i]
            pred = h1 * enc_state
            diff = x - pred
            idx = quantize_gaussian(diff / g)
            indices[b+i] = idx
            enc_state = pred + V_LEVELS[idx] * g

    rx_indices = channel_func(indices) if channel_func else indices

    out = np.zeros(len(signal))
    dec_state = 0.0
    for b in range(0, len(signal), block_size):
        g = gains[b // block_size] 
        for i in range(min(block_size, len(signal) - b)):
            idx = rx_indices[b+i]
            pred = h1 * dec_state
            rec = pred + V_LEVELS[idx] * g
            out[b+i] = rec
            dec_state = rec

    return out

def adpcm_aqf_encode_decode(signal, order=1, channel_func=None):
    block_size = 32
    indices = np.zeros(len(signal), dtype=int)
    gains = np.zeros(len(signal) // block_size + 1)
    h_coeffs = np.zeros((len(signal) // block_size + 1, order))
    
    for b in range(0, len(signal), block_size):
        block = signal[b:b+block_size]
        acf = get_acf(block, order)
        h = solve_yule_walker(acf, order)
        h_coeffs[b // block_size] = h
        
        # Calculate diffs using this optimal h
        diffs = np.zeros(len(block))
        state = np.zeros(order)
        for i, x in enumerate(block):
            pred = np.dot(h, state)
            diffs[i] = x - pred
            state = np.roll(state, 1)
            state[0] = x
            
        std_est = np.std(diffs)
        gains[b // block_size] = std_est if std_est > 1e-4 else 1e-4
        
    enc_state = np.zeros(order)
    for b in range(0, len(signal), block_size):
        g = gains[b // block_size]
        h = h_coeffs[b // block_size]
        for i in range(min(block_size, len(signal) - b)):
            x = signal[b+i]
            pred = np.dot(h, enc_state)
            diff = x - pred
            idx = quantize_gaussian(diff / g)
            indices[b+i] = idx
            rec = pred + V_LEVELS[idx] * g
            enc_state = np.roll(enc_state, 1)
            enc_state[0] = rec

    rx_indices = channel_func(indices) if channel_func else indices

    out = np.zeros(len(signal))
    dec_state = np.zeros(order)
    for b in range(0, len(signal), block_size):
        g = gains[b // block_size] 
        h = h_coeffs[b // block_size]
        for i in range(min(block_size, len(signal) - b)):
            idx = rx_indices[b+i]
            pred = np.dot(h, dec_state)
            rec = pred + V_LEVELS[idx] * g
            out[b+i] = rec
            dec_state = np.roll(dec_state, 1)
            dec_state[0] = rec

    return out

