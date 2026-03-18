import numpy as np
import librosa
from encoders import (
    pcm_encode, pcm_decode,
    pcm_aqf_encode, pcm_aqf_decode,
    dpcm1_aqf_encode_decode,
    adpcm_aqf_encode_decode
)
from channel import apply_channel_errors

def calculate_snr(orig, recon):
    signal_power = np.mean(orig**2)
    noise_power = np.mean((orig - recon)**2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

def generate_ar1_signal(rho=0.85, length=16000):
    np.random.seed(42)
    noise = np.random.randn(length)
    signal = np.zeros(length)
    signal[0] = noise[0]
    for i in range(1, length):
        signal[i] = rho * signal[i-1] + np.sqrt(1 - rho**2) * noise[i]
    signal = signal / np.std(signal)
    return signal

def load_audio_signal(filepath, sr=8000, duration_sec=None):
    if duration_sec is not None and duration_sec > 0:
        signal, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration_sec)
    else:
        signal, _ = librosa.load(filepath, sr=sr, mono=True)
        
    remainder = len(signal) % 32
    if remainder != 0:
        signal = np.pad(signal, (0, 32 - remainder))
    
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 0:
        signal = signal / std
    return signal

def run_simulation(p_error, code_type='FBC', ep_level=0, scheme='PCM', audio_file=None, h1=0.85, duration_sec=2.0):
    if audio_file:
        signal = load_audio_signal(audio_file, sr=8000, duration_sec=duration_sec)
    else:
        length = int(duration_sec * 8000) if duration_sec and duration_sec > 0 else 16000
        signal = generate_ar1_signal(length=length)
    
    def channel_func(indices):
        return apply_channel_errors(indices, p_error, code_type, ep_level)

    if scheme == 'PCM':
        indices = pcm_encode(signal, var=1.0)
        rx_indices = channel_func(indices)
        recon = pcm_decode(rx_indices, var=1.0)
    elif scheme == 'PCM-AQF':
        indices, gains = pcm_aqf_encode(signal)
        rx_indices = channel_func(indices)
        recon = pcm_aqf_decode(rx_indices, gains)
    elif scheme == 'DPCM1-AQF':
        recon = dpcm1_aqf_encode_decode(signal, h1=h1, channel_func=channel_func)
    elif scheme == 'ADPCM1-AQF':
        recon = adpcm_aqf_encode_decode(signal, order=1, channel_func=channel_func)
    elif scheme == 'ADPCM4-AQF':
        recon = adpcm_aqf_encode_decode(signal, order=4, channel_func=channel_func)
    else:
        raise ValueError(f"Unknown scheme {scheme}")
        
    return calculate_snr(signal, recon)

