import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import matplotlib
import matplotlib.pyplot as plt
from simulation import run_simulation
from constants import SCHEMES
import numpy as np
import scipy.optimize

# Use non-interactive backend for server
matplotlib.use('Agg')

def fit_alpha(rates, snrs):
    E_tot = 10.0 ** (-np.array(snrs) / 10.0)
    E_q = E_tot[0]
    E_c = E_tot - E_q
    
    def model(P, a1, a2):
        return a1 * P + a2 * (P ** 2)
        
    popt, _ = scipy.optimize.curve_fit(model, rates, E_c)
    return float(popt[0]), float(popt[1])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        scheme = request.form.get('scheme', 'PCM')
        signal_type = request.form.get('signal_type', 'synthetic')
        ep_noep = request.form.get('ep_noep') == 'true'
        ep_1 = request.form.get('ep_1') == 'true'
        ep_2 = request.form.get('ep_2') == 'true'
        
        rates_str = request.form.get('rates', '0, 0.01, 0.02, 0.03, 0.04, 0.05')
        try:
            rates = [float(r.strip()) for r in rates_str.split(',')]
            rates.sort()
        except ValueError:
            rates = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
            
        try:
            h1 = float(request.form.get('h1_val', '0.85'))
        except ValueError:
            h1 = 0.85

        duration_type = request.form.get('duration_type', '2')
        if duration_type == 'whole':
            duration_sec = None
        else:
            try:
                duration_sec = float(duration_type)
            except ValueError:
                duration_sec = 2.0
        
        # Determine audio path if uploaded
        audio_path = None
        if signal_type == 'upload':
            if 'audio_file' not in request.files:
                return jsonify({'error': 'No audio file uploaded.'}), 400
            file = request.files['audio_file']
            if file.filename == '':
                return jsonify({'error': 'No audio file selected.'}), 400
            # Save temporarily
            os.makedirs('tmp', exist_ok=True)
            audio_path = os.path.join('tmp', file.filename)
            file.save(audio_path)
            
        # Run simulation
        plt.figure(figsize=(10, 6), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('#ffffff')
        
        alphas_data = []

        if ep_2:
            snrs = [run_simulation(p, scheme=scheme, ep_level=2, audio_file=audio_path, h1=h1, duration_sec=duration_sec) for p in rates]
            plt.plot(rates, snrs, marker='s', label=f'{scheme} (EP2)', color='#3b82f6', linewidth=2, markersize=8)
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': 'EP2', 'a1': a1, 'a2': a2})
        if ep_1:
            snrs = [run_simulation(p, scheme=scheme, ep_level=1, audio_file=audio_path, h1=h1, duration_sec=duration_sec) for p in rates]
            plt.plot(rates, snrs, marker='^', label=f'{scheme} (EP1)', color='#f59e0b', linewidth=2, markersize=8)
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': 'EP1', 'a1': a1, 'a2': a2})
        if ep_noep:
            snrs = [run_simulation(p, scheme=scheme, ep_level=0, audio_file=audio_path, h1=h1, duration_sec=duration_sec) for p in rates]
            plt.plot(rates, snrs, marker='o', label=f'{scheme} (No EP)', color='#10b981', linewidth=2, markersize=8)
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': 'No EP', 'a1': a1, 'a2': a2})
            
        plt.title(f"Simulated SNR vs Bit Error Rate - {scheme}", fontsize=14, pad=15)
        plt.xlabel("Bit Error Rate (P)", fontsize=12)
        plt.ylabel("Signal-to-Noise Ratio (dB)", fontsize=12)
        
        if len(rates) > 1:
            plt.xlim(min(rates), max(rates))
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(framealpha=0.9, fancybox=True, edgecolor='#d1d5db')
        plt.tight_layout()
        
        # Save to base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            
        return jsonify({
            'plot_url': f"data:image/png;base64,{img_str}",
            'alphas': alphas_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

