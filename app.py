import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import matplotlib
import matplotlib.pyplot as plt
from simulation import run_simulation
from theoretical import compute_theoretical_snr
from constants import SCHEMES
import numpy as np
import scipy.optimize
import json

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
        code_type = request.form.get('code_type', 'FBC')
        error_type = request.form.get('error_type', 'independent')
        signal_type = request.form.get('signal_type', 'synthetic')
        ep_noep = request.form.get('ep_noep') == 'true'
        ep_1 = request.form.get('ep_1') == 'true'
        ep_2 = request.form.get('ep_2') == 'true'
        plot_theo = request.form.get('plot_theo') == 'true'
        
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
        
        err_lbl = 'B' if error_type == 'burst' else 'I'

        run_idx = int(request.form.get('history_count', '0'))
        
        # Styles for different runs
        STYLES = [
            {'colors': ['#3b82f6', '#f59e0b', '#10b981'], 'markers': ['s', '^', 'o']},
            {'colors': ['#ef4444', '#8b5cf6', '#14b8a6'], 'markers': ['D', 'v', 'p']},
            {'colors': ['#ec4899', '#eab308', '#0ea5e9'], 'markers': ['X', '<', 'h']},
            {'colors': ['#f97316', '#6366f1', '#84cc16'], 'markers': ['P', '>', 'd']},
            {'colors': ['#a855f7', '#f43f5e', '#06b6d4'], 'markers': ['*', 'H', '8']}
        ]
        style = STYLES[run_idx % len(STYLES)]
        c_ep2, c_ep1, c_noep = style['colors']
        m_ep2, m_ep1, m_noep = style['markers']

        prev_lines_str = request.form.get('previous_lines', '[]')
        prev_alphas_str = request.form.get('previous_alphas', '[]')
        try:
            prev_lines = json.loads(prev_lines_str)
        except:
            prev_lines = []
        try:
            prev_alphas = json.loads(prev_alphas_str)
        except:
            prev_alphas = []

        lines_data = list(prev_lines)
        alphas_data = list(prev_alphas)

        # Plot previous lines first
        for line in prev_lines:
            plt.plot(line['x'], line['y'], marker=line.get('marker', ''), linestyle=line.get('linestyle', '-'), color=line.get('color', 'blue'), linewidth=2, markersize=8, label=line['label'])

        if ep_2:
            snrs = [run_simulation(p, scheme=scheme, code_type=code_type, ep_level=2, audio_file=audio_path, h1=h1, duration_sec=duration_sec, error_type=error_type) for p in rates]
            lbl = f'{scheme} {code_type} {err_lbl} (EP2)'
            plt.plot(rates, snrs, marker=m_ep2, label=lbl, color=c_ep2, linewidth=2, markersize=8)
            lines_data.append({'x': rates, 'y': snrs, 'label': lbl, 'color': c_ep2, 'marker': m_ep2, 'linestyle': '-'})
            if plot_theo:
                theo = [compute_theoretical_snr(p, code_type=code_type, ep_level=2) for p in rates]
                theo_lbl = f'Theo {code_type} (EP2)'
                plt.plot(rates, theo, linestyle='--', color=c_ep2, linewidth=2, label=theo_lbl)
                lines_data.append({'x': rates, 'y': theo, 'label': theo_lbl, 'color': c_ep2, 'marker': '', 'linestyle': '--'})
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': lbl, 'a1': a1, 'a2': a2})
            
        if ep_1:
            snrs = [run_simulation(p, scheme=scheme, code_type=code_type, ep_level=1, audio_file=audio_path, h1=h1, duration_sec=duration_sec, error_type=error_type) for p in rates]
            lbl = f'{scheme} {code_type} {err_lbl} (EP1)'
            plt.plot(rates, snrs, marker=m_ep1, label=lbl, color=c_ep1, linewidth=2, markersize=8)
            lines_data.append({'x': rates, 'y': snrs, 'label': lbl, 'color': c_ep1, 'marker': m_ep1, 'linestyle': '-'})
            if plot_theo:
                theo = [compute_theoretical_snr(p, code_type=code_type, ep_level=1) for p in rates]
                theo_lbl = f'Theo {code_type} (EP1)'
                plt.plot(rates, theo, linestyle='--', color=c_ep1, linewidth=2, label=theo_lbl)
                lines_data.append({'x': rates, 'y': theo, 'label': theo_lbl, 'color': c_ep1, 'marker': '', 'linestyle': '--'})
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': lbl, 'a1': a1, 'a2': a2})
            
        if ep_noep:
            snrs = [run_simulation(p, scheme=scheme, code_type=code_type, ep_level=0, audio_file=audio_path, h1=h1, duration_sec=duration_sec, error_type=error_type) for p in rates]
            lbl = f'{scheme} {code_type} {err_lbl} (No EP)'
            plt.plot(rates, snrs, marker=m_noep, label=lbl, color=c_noep, linewidth=2, markersize=8)
            lines_data.append({'x': rates, 'y': snrs, 'label': lbl, 'color': c_noep, 'marker': m_noep, 'linestyle': '-'})
            if plot_theo:
                theo = [compute_theoretical_snr(p, code_type=code_type, ep_level=0) for p in rates]
                theo_lbl = f'Theo {code_type} (No EP)'
                plt.plot(rates, theo, linestyle='--', color=c_noep, linewidth=2, label=theo_lbl)
                lines_data.append({'x': rates, 'y': theo, 'label': theo_lbl, 'color': c_noep, 'marker': '', 'linestyle': '--'})
            a1, a2 = fit_alpha(rates, snrs)
            alphas_data.append({'variant': lbl, 'a1': a1, 'a2': a2})
            
        is_multiple = len(prev_lines) > 0
        title_text = "Comparison of Simulated SNR vs Bit Error Rate" if is_multiple else f"Simulated SNR vs Bit Error Rate - {scheme} ({code_type})"
        plt.title(title_text, fontsize=14, pad=15)
        plt.xlabel("Bit Error Rate (P)", fontsize=12)
        plt.ylabel("Signal-to-Noise Ratio (dB)", fontsize=12)
        
        if len(rates) > 1:
            plt.xlim(min(rates), max(rates))
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(framealpha=0.9, fancybox=True, edgecolor='#d1d5db')
        plt.tight_layout()
        
        # Save to base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', transparent=False)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            
        # Generate Alpha Plot
        alpha_img_str = None
        if len(alphas_data) > 0:
            x_order = SCHEMES 
            series_data = {} 
            
            for item in alphas_data:
                variant = item['variant']
                a1 = item['a1']
                a2 = item['a2']
                
                found_sch = None
                for sch in sorted(SCHEMES, key=len, reverse=True):
                    if variant.startswith(sch):
                        found_sch = sch
                        break
                
                if found_sch:
                    remainder = variant[len(found_sch):].strip()
                    if remainder not in series_data:
                        series_data[remainder] = {}
                    series_data[remainder][found_sch] = (a1, a2)

            if series_data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                fig.patch.set_facecolor('#ffffff')
                
                lines_styles = ['-', '--', '-.', ':']
                markers = ['o', 's', '^', 'D', 'v', 'p', 'X']
                
                for i, (remainder, sch_dict) in enumerate(series_data.items()):
                    style = lines_styles[i % len(lines_styles)]
                    marker = markers[i % len(markers)]
                    
                    x_vals = []
                    y1_vals = []
                    y2_vals = []
                    for sch in x_order:
                        if sch in sch_dict:
                            x_vals.append(sch)
                            y1_vals.append(sch_dict[sch][0])
                            y2_vals.append(sch_dict[sch][1])
                            
                    if x_vals:
                        ax1.plot(x_vals, y1_vals, marker=marker, linestyle=style, linewidth=2, markersize=8, label=remainder)
                        ax2.plot(x_vals, y2_vals, marker=marker, linestyle=style, linewidth=2, markersize=8, label=remainder)
                        
                ax1.set_ylabel(r'$\alpha_1$ (Linear)', fontsize=12)
                ax1.set_title('Alpha Parameters vs Encoder Scheme', fontsize=14, pad=15)
                ax1.grid(True, linestyle='--', alpha=0.6)
                if len(series_data) > 0:
                    ax1.legend(framealpha=0.9, fancybox=True, edgecolor='#d1d5db', fontsize=9, loc='best')
                
                ax2.set_ylabel(r'$\alpha_2$ (Quadratic)', fontsize=12)
                ax2.set_xlabel('Encoder Scheme', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                
                buf_alpha = BytesIO()
                plt.savefig(buf_alpha, format='png', dpi=150, facecolor='white')
                buf_alpha.seek(0)
                alpha_img_str = base64.b64encode(buf_alpha.read()).decode('utf-8')
                plt.close(fig)

        return jsonify({
            'plot_url': f"data:image/png;base64,{img_str}",
            'alphas': alphas_data,
            'lines': lines_data,
            'alpha_plot_url': f"data:image/png;base64,{alpha_img_str}" if alpha_img_str else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

