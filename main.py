import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from theoretical import compute_theoretical_snr
from simulation import run_simulation

def plot_theoretical():
    import os
    os.makedirs("plots", exist_ok=True)
    rates = np.linspace(0, 0.06, 50)
    fbc_snrs = [compute_theoretical_snr(p, 'FBC', 0) for p in rates]
    nbc_snrs = [compute_theoretical_snr(p, 'NBC', 0) for p in rates]
    fbc_ep1 = [compute_theoretical_snr(p, 'FBC', 1) for p in rates]
    fbc_ep2 = [compute_theoretical_snr(p, 'FBC', 2) for p in rates]
    
    plt.figure(figsize=(8, 6))
    plt.plot(rates, fbc_ep2, label='FBC (EP2)', color='blue')
    plt.plot(rates, fbc_ep1, label='FBC (EP1)', color='orange')
    plt.plot(rates, fbc_snrs, label='FBC (No EP)', color='green')
    plt.plot(rates, nbc_snrs, label='NBC (No EP)', color='red', linestyle='--')
    
    plt.title("Theoretical SNR vs channel bit-error rate (Fig 4)")
    plt.xlabel("Bit error rate, P")
    plt.ylabel("Overall SNR (dB)")
    plt.xlim(0, 0.06)
    plt.ylim(0, 18)
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/theoretical_snr.png")
    print("Saved plots/theoretical_snr.png")

from constants import SCHEMES

def plot_simulation(audio_file=None, duration_sec=2.0):
    import os
    os.makedirs("plots", exist_ok=True)
    rates = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'd', 'x']

    # 1. Individual plots for each of the 5 schemes showing No EP, EP1, EP2
    # So 5 figures.
    all_results = {}
    for scheme in SCHEMES:
        print(f"Running simulation for {scheme}...")
        snrs_noep = [run_simulation(p, scheme=scheme, ep_level=0, audio_file=audio_file, duration_sec=duration_sec) for p in rates]
        snrs_ep1  = [run_simulation(p, scheme=scheme, ep_level=1, audio_file=audio_file, duration_sec=duration_sec) for p in rates]
        snrs_ep2  = [run_simulation(p, scheme=scheme, ep_level=2, audio_file=audio_file, duration_sec=duration_sec) for p in rates]
        all_results[scheme] = {'noep': snrs_noep, 'ep1': snrs_ep1, 'ep2': snrs_ep2}
        
        plt.figure(figsize=(8, 6))
        plt.plot(rates, snrs_ep2, marker='s', label=f'{scheme} (EP2)', color='blue')
        plt.plot(rates, snrs_ep1, marker='^', label=f'{scheme} (EP1)', color='orange')
        plt.plot(rates, snrs_noep, marker='o', label=f'{scheme} (No EP)', color='green')
        
        plt.title(f"Simulated SNR vs Bit Error Rate - {scheme}")
        plt.xlabel("Bit Error Rate (P)")
        plt.ylabel("Signal-to-Noise Ratio (dB)")
        plt.xlim(0, 0.05)
        plt.grid(True)
        plt.legend()
        filename = f"plots/sim_{scheme.lower().replace('-', '_')}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")

    # 2. Overall Simulation SNR Plot (comparing the base No EP versions across the 5 schemes)
    print("Generating overall comparison plot...")
    plt.figure(figsize=(10, 6))
    for i, scheme in enumerate(SCHEMES):
        plt.plot(rates, all_results[scheme]['noep'], marker=markers[i], label=f'{scheme} (No EP)', color=colors[i])
        
    plt.title("Overall Simulated SNR Comparison (All schemes, No Error Protection)")
    plt.xlabel("Bit Error Rate (P)")
    plt.ylabel("Signal-to-Noise Ratio (dB)")
    plt.xlim(0, 0.05)
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sim_overall.png")
    print("Saved plots/sim_overall.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, choices=["theoretical", "simulation"], required=True)
    parser.add_argument("--audio", type=str, help="Path to input audio file to encode (e.g. elephant.mp3)", default=None)
    parser.add_argument("--duration", type=float, help="Duration in seconds. Set to 0 for whole file.", default=2.0)
    args = parser.parse_args()
    
    if args.plot == "theoretical":
        plot_theoretical()
    elif args.plot == "simulation":
        dur = None if args.duration <= 0 else args.duration
        plot_simulation(audio_file=args.audio, duration_sec=dur)

