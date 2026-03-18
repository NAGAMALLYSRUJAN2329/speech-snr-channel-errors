import numpy as np

# Lloyd-Max quantizer values for N=8 steps (Gaussian, mean 0, var 1)
# Derived from Max (1960) "Quantizing for Minimum Distortion"
V_LEVELS = np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152])

# Bounds used for encoding decisions (no infinity)
U_BOUNDS_ENCODER = np.array([-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748])

# Bounds used for theoretical numerical CDF integration
U_BOUNDS_THEORETICAL = np.array([-np.inf, -1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748, np.inf])

# Base variance characteristics
SIGMA_X_SQ = 1.0
SIGMA_Q_SQ = 0.03451  # Theoretical normalization constant from paper

# Natural Binary Code (m=3 bits)
CODEWORDS_NBC = [
    [0,0,0], [0,0,1], [0,1,0], [0,1,1],
    [1,0,0], [1,0,1], [1,1,0], [1,1,1]
]

# Folded Binary Code. Most significant bit indicates polarity
CODEWORDS_FBC = [
    [0,1,1], [0,1,0], [0,0,1], [0,0,0],
    [1,0,0], [1,0,1], [1,1,0], [1,1,1]
]

# Supported simulation schemes
SCHEMES = ['PCM', 'PCM-AQF', 'DPCM1-AQF', 'ADPCM1-AQF', 'ADPCM4-AQF']
