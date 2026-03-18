# Speech-Encoding Channel Errors Simulation

This repository contains an end-to-end Python implementation and simulation of the communication systems proposed in the paper:
**"Effects of Channel Errors on the Signal-to-Noise Performance of Speech-Encoding Systems" by P. Noll (1975).**

## Overview
This project simulates the effects of digital transmission channel errors on the overall Signal-to-Noise Ratio (SNR) for various speech-encoding algorithms. It replicates both the theoretical mathematical models (using optimal Lloyd-Max 3-bit gaussian quantizers) and practical simulation pipelines tested over a pseudo-random independent error channel.

We have expanded on the original framework to natively support **real `.mp3` and `.wav` audio transmission** via `librosa`, allowing direct testing against real-world human speech variations!

5 core speech-encoding schemes are evaluated natively inside the repository codebase:
1. **PCM (Nonadaptive)**: Standard log-PCM utilizing the absolute normal distribution.
2. **PCM-AQF**: Adaptive Quantization with Forward estimation over blocks of 32 samples normalized to the absolute block peak.
3. **DPCM1-AQF**: Differential PCM utilizing an optimal 1-tap predictor state ($h_1 = \text{Variable}$).
4. **ADPCM1-AQF**: Adaptive DPCM solving a real-time short-term lag-1 autocorrelation per 32-sample block.
5. **ADPCM4-AQF**: Highly optimized ADPCM computing robust 4-tap predictor coefficients via Yule-Walker solutions per 32-sample block.

Each scheme is simulated against 3 different Error Protection (EP) levels across a transmission channel simulating bit-error rates up to $5\%$ ($P = 0.05$):
- **No EP**: No error protection provided.
- **EP1**: Perfect protection of the Most Significant Bit (MSB/Sign Bit).
- **EP2**: Perfect protection of the 2 Most Significant Bits.

## New Feature: Interactive Web Application
A completely responsive **Flask + Vanilla HTML/CSS** application is included, mapping the backend logic visually in real-time.
- **Glassmorphism UI**: Beautifully designed tracking inputs.
- **Custom Integrations**: Swap out specific variables like $P$ evaluation rates, custom snippet lengths (e.g., 2 second processing vs whole-file), and DPCM $h_1$ modifiers.
- **Alpha Parametric Evaluation**: Dynamically fits and evaluates the exact analytical transmission error estimations ($\epsilon_c^2 \approx \alpha_1 P + \alpha_2 P^2$) using Least-Squares optimization via `scipy.optimize.curve_fit`! The derived parameters neatly display underneath your generated dynamic plots!

## Project Structure
- `app.py` & `templates/`: The primary Flask Web-UI routing server.
- `constants.py`: Stores optimized quantization bounds and shared tracking metadata.
- `theoretical.py`: Computes theoretical structural formulas based on 3-bit Maximum-Lloyd quantizers, solving exact Folded Binary Code (FBC) and Natural Binary Code (NBC).
- `channel.py`: Implements the bit-transmission channel and introduces probabilistic random errors natively.
- `encoders.py`: Extensively implements the predictive mathematics solving PCM, DPCM, ADPCM1, and ADPCM4 matrices.
- `simulation.py`: Orchestrates loading audio snippets, encoding, running the channel interference arrays, and plotting identical reconstruction metrics natively into SNRs.
- `main.py`: Main CLI tool handling local shell executions.
- `docs/`: Holds exhaustive multi-step explanations and comprehensive logic derivations for each of the 5 encoding schemes natively. (e.g., `pcm_aqf.md`)
- `plots/`: Safe collection point mapping the dynamically generated `matplotlib` `.png` output graphs statically.

## Installation
Ensure you have Python 3 installed, then install the required mathematical and web libraries:
```bash
pip install numpy scipy matplotlib librosa flask flask-cors
```

## Usage

### 1. Launch Web Dashboard (Recommended)
This runs the UI displaying parameters, allowing file uploads, and presenting the $\alpha_1 / \alpha_2$ evaluation tables.
```bash
python app.py
```
*Then navigate to `http://127.0.0.1:5000` via your web browser.*

### 2. General CLI Simulation
Execute the entire sequence comparing all frameworks iteratively. If `--audio` is omitted, it will explicitly default to Noll's 1975 simulated Independent AR(1) mapping metric.
```bash
python main.py --plot simulation --audio elephant.mp3 --duration 3.0
```
> *Generates 6 files routed sequentially into `plots/`*

### 3. Generate Theoretical Model Baseline
Evaluate purely structural limits defining Folded-Binary vs Natural-Binary protections mathematically:
```bash
python main.py --plot theoretical
```
> *Generates `plots/theoretical_snr.png`*
