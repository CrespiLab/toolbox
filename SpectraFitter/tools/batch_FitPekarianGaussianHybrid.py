# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:11:55 2025
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QTextEdit
)
from PyQt5.QtCore import Qt
import math
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# --- Pekarian and helper functions ---

def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def pekaria_auto(v, S, v0, Omega, sigma0, delta, k_max=10):
    v = np.asarray(v, dtype=float)
    spec = np.zeros_like(v)
    exp_neg_S = np.exp(-S)
    for k in range(k_max + 1):
        weight = S ** k / math.factorial(k)
        mu = v0 + k * Omega
        sigma = sigma0 + k * delta
        if sigma <= 0:
            continue
        spec += weight * gaussian(v, mu, sigma)
    return exp_neg_S * spec

def pekaria_scaled_k(v, A, S, v0, Omega, sigma0, delta, k_max):
    return A * pekaria_auto(v, S, v0, Omega, sigma0, delta, k_max)

def pure_gaussian(v, A, v0, sigma):
    return A * gaussian(v, v0, sigma)

def iterative_pekaria_fit(v, a, v_centers, k_max=20):
    n_pf = len(v_centers)
    p0 = []
    bounds_lower = []
    bounds_upper = []
    for vc in v_centers:
        A = max(a) / (n_pf * 1.5)
        S = 0.7
        Omega = 1000
        sigma0 = 600
        delta = 60
        p0.extend([A, S, vc, Omega, sigma0, delta])
        bounds_lower.extend([0, 0.01, vc - 500, 200, 100, 0])
        bounds_upper.extend([1e8, 5.0, vc + 500, 3000, 3000, 300])
    # Gaussian tail params
    A_tail = max(a) / 2
    v_tail = max(v) + 1000
    sigma_tail = 1000
    p0.extend([A_tail, v_tail, sigma_tail])
    bounds_lower.extend([0, max(v), 100])
    bounds_upper.extend([1e8, max(v) + 5000, 3000])
    
    def hybrid_model_dynamic(v, *params):
        comps = []
        idx = 0
        for i in range(n_pf):
            comp = pekaria_scaled_k(v, *params[idx:idx+6], k_max=k_max)
            comps.append(comp)
            idx += 6
        tail = pure_gaussian(v, *params[idx:idx+3])
        comps.append(tail)
        return sum(comps), comps
    
    def model_to_fit(v, *params):
        s, _ = hybrid_model_dynamic(v, *params)
        return s

    popt, pcov = curve_fit(
        model_to_fit, v, a, p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=100000
    )
    
    v_fit = np.array(v)
    a_fit, components = hybrid_model_dynamic(v_fit, *popt)
    
    return popt, pcov, v_fit, a_fit, components

def residual_driven_peak_finder(v, a, max_peaks, residual_threshold, k_max,
                                center_deviation_threshold,
                                cancel_flag=None, output_console=None):
    centers = []
    popt = None

    for iteration in range(max_peaks):
        if cancel_flag and cancel_flag():
            if output_console:
                output_console.append("❌ Fitting cancelled by user.")
                QApplication.processEvents()
            break

        if len(centers) == 0:
            initial_centers = [v[np.argmax(a)]]
        else:
            initial_centers = centers

        try:
            popt, pcov, v_fit, a_fit, comps = iterative_pekaria_fit(v, a, initial_centers, k_max=k_max)
        except Exception as e:
            if output_console:
                output_console.append(f"⚠️ Fitting error: {e}")
                QApplication.processEvents()
            break

        interp_fit = interp1d(v_fit, a_fit, kind='linear', bounds_error=False, fill_value="extrapolate")
        residuals = a - interp_fit(v)
        max_residual = np.max(residuals)

        if output_console:
            output_console.append(f"Cycle {iteration+1}: max residual = {max_residual:.4f}")
            QApplication.processEvents()

        if max_residual < residual_threshold:
            if output_console:
                output_console.append("✅ Residual threshold met. Stopping.")
                QApplication.processEvents()
            break

        peaks, _ = find_peaks(residuals, height=max_residual * 0.5, distance=20)
        if len(peaks) == 0:
            if output_console:
                output_console.append("No new peaks found in residuals. Stopping.")
                QApplication.processEvents()
            break

        new_center = v[peaks[0]]
        if any(abs(new_center - c) < center_deviation_threshold for c in centers):
            if output_console:
                output_console.append(f"🔁 New center {new_center:.1f} too close to existing peaks. Stopping.")
                QApplication.processEvents()
            break

        centers.append(new_center)
        if output_console:
            output_console.append(f"➕ Added new center at {new_center:.1f} cm⁻¹")
            QApplication.processEvents()

    return popt, centers, v_fit, a_fit, residuals

# --- PyQt5 Application ---

# class PekariaFitApp(QWidget):
#     def __init__(self, v_data, a_data):
#         super().__init__()
#         self.v = v_data
#         self.a = a_data
#         self.manual_mode = True
#         self.initUI()
#         self.threshold = 0.001
#         self.max_peaks = 10
#         self.k_max = 10
#         self.center_deviation_threshold = 100

    
#     def initUI(self):
#         self.setWindowTitle("Interactive Pekarian Fit")
#         layout = QVBoxLayout()
        
#         # Mode toggle button
#         self.mode_button = QPushButton("Switch to Auto Mode")
#         self.mode_button.clicked.connect(self.toggle_mode)
#         layout.addWidget(self.mode_button)
        
#         # Slider and label for number of PFs
#         h_layout = QHBoxLayout()
#         label = QLabel("Number of Pekarian bands:")
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setMinimum(1)
#         self.slider.setMaximum(10)
#         self.slider.setValue(7)
#         self.slider.valueChanged.connect(self.update_centers_placeholder)
#         self.slider.valueChanged.connect(self.update_param_count_label)
#         h_layout.addWidget(label)
#         h_layout.addWidget(self.slider)
#         layout.addLayout(h_layout)
        
#         # Parameter count label
#         self.param_count_label = QLabel()
#         layout.addWidget(self.param_count_label)
#         self.update_param_count_label()
        
#         # Centers input
#         layout.addWidget(QLabel("Enter centers (cm⁻¹), comma separated:"))
#         self.centers_text = QTextEdit()
#         self.centers_text.setPlaceholderText("23000, 28000, 32000")
#         self.centers_text.setFixedHeight(50)
#         layout.addWidget(self.centers_text)
#         self.update_centers_placeholder()
        
#         # Fit button
#         self.fit_button = QPushButton("Fit")
#         self.fit_button.clicked.connect(self.run_fit)
#         layout.addWidget(self.fit_button)
        
#         # Fit plot
#         self.fig_fit, self.ax_fit = plt.subplots(figsize=(8, 4))
#         self.canvas_fit = FigureCanvas(self.fig_fit)
#         layout.addWidget(self.canvas_fit)
        
#         # Residual plot
#         self.fig_res, self.ax_res = plt.subplots(figsize=(8, 2))
#         self.canvas_res = FigureCanvas(self.fig_res)
#         layout.addWidget(self.canvas_res)
        
        
#         self.save_button = QPushButton("Save Fit as CSV")
#         self.save_button.clicked.connect(self.save_fit_to_csv)
#         layout.addWidget(self.save_button)
#         self.setLayout(layout)
#         self.update_widgets_state()
    
#     def toggle_mode(self):
#         self.manual_mode = not self.manual_mode
#         self.update_widgets_state()
    
#     def update_widgets_state(self):
#         if self.manual_mode:
#             self.mode_button.setText("Switch to Auto Mode")
#             self.slider.setEnabled(True)
#             self.centers_text.setEnabled(True)
#             self.param_count_label.show()
#             # self.threshold_text.hide()
#             # self.max_peaks_text.setEnabled(True)
#             # self.k_max_text.hide()
#         else:
#             self.mode_button.setText("Switch to Manual Mode")
#             self.slider.setEnabled(False)
#             self.centers_text.setEnabled(False)
#             self.param_count_label.hide()
#             # self.threshold_text.show()
#             # self.max_peaks_text.setEnabled(True)
#             # self.k_max_text.show()
#             self.centers_text.clear()
#             self.centers_text.setPlaceholderText("Auto mode will detect peaks automatically.")
    
#     def update_centers_placeholder(self):
#         n = self.slider.value()
#         example_centers = ', '.join(str(23000 + i*5000) for i in range(n))
#         print(f"example_centers: {example_centers}")
#         self.centers_text.setPlaceholderText(example_centers)
    
#     def update_param_count_label(self):
#         n_pf = self.slider.value()
#         total_params = n_pf
#         self.param_count_label.setText(f"Total parameters: {total_params}")
    
#     def run_fit(self):
#         if self.manual_mode:
#             centers_str = self.centers_text.toPlainText()
#             if not centers_str.strip():
#                 centers_str = self.centers_text.placeholderText()
#             try:
#                 centers = [float(c.strip()) for c in centers_str.split(',')]
#             except:
#                 self.show_error("Invalid centers input!")
#                 return
            
#             if len(centers) != self.slider.value():
#                 self.show_error("Number of centers does not match slider value!")
#                 return
            
#             try:
#                 popt, pcov, v_fit, a_fit, comps = iterative_pekaria_fit(self.v, self.a, centers, k_max=self.k_max)
#             except Exception as e:
#                 self.show_error(f"Fit error: {e}")
#                 return
            
#             residuals = self.a - np.interp(self.v, v_fit, a_fit)
#             self.plot_results(v_fit, a_fit, comps, centers, residuals)
        
#         else:
#             try:
#                 popt, centers, v_fit, a_fit, residuals = residual_driven_peak_finder(self.v, self.a, 
#                                                                                      max_peaks=self.max_peaks, 
#                                                                                      residual_threshold=self.threshold, 
#                                                                                      k_max=self.k_max,
#                                                                                      center_deviation_threshold=self.center_deviation_threshold)
#             except Exception as e:
#                 self.show_error(f"Auto fit error: {e}")
#                 return
#             self.plot_results(v_fit, a_fit, None, centers, residuals, auto_mode=True)
    
#     def plot_results(self, v_fit, a_fit, comps, centers, residuals, auto_mode=False):
#         self.ax_fit.clear()
#         self.ax_fit.plot(self.v, self.a, label="Experimental", color='black', linewidth=1.2)
        
#         if auto_mode:
#             # Plot PFs from parameters if possible (skipped here for brevity)
#             for i, c in enumerate(centers, 1):
#                 self.ax_fit.axvline(c, color='blue', linestyle='--', label=f"Auto Peak {i} ({c:.0f})")
#         else:
#             for i, comp in enumerate(comps[:-1], 1):
#                 self.ax_fit.plot(v_fit, comp, '--', label=f"PF{i} (~{centers[i-1]})")
#             self.ax_fit.plot(v_fit, comps[-1], '--', label="Gaussian Tail")
        
#         self.ax_fit.plot(v_fit, a_fit, label="Total Fit", color='orange', linewidth=1.5)
#         self.ax_fit.set_xlabel("Wavenumber (cm⁻¹)")
#         self.ax_fit.set_ylabel("Epsilon")
#         self.ax_fit.set_title("Interactive Pekarian Fit" + (" (Auto Mode)" if auto_mode else ""))
#         self.ax_fit.legend()
#         self.ax_fit.invert_xaxis()
#         self.canvas_fit.draw()
        
#         self.ax_res.clear()
#         self.ax_res.plot(self.v, residuals, color='red')
#         self.ax_res.axhline(0, color='black', linestyle='--')
#         self.ax_res.set_xlabel("Wavenumber (cm⁻¹)")
#         self.ax_res.set_ylabel("Residuals")
#         self.ax_res.set_title("Residuals (Experimental - Fit)")
#         self.last_v_fit = v_fit
#         self.last_a_fit = a_fit
#         self.ax_res.invert_xaxis()
#         self.canvas_res.draw()
    
#     def show_error(self, message):
#         self.ax_fit.clear()
#         self.ax_fit.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
#         self.canvas_fit.draw()


#     def save_fit_to_csv(self):
#         if not hasattr(self, 'last_v_fit') or not hasattr(self, 'last_a_fit'):
#             self.show_error("❌ No fit data to save.")
#             return
#         path, _ = QFileDialog.getSaveFileName(self, "Save Fit", "", "CSV Files (*.csv)")
#         if path:
#             df = pd.DataFrame({'Wavenumber': self.last_v_fit, 'Fitted Absorbance': self.last_a_fit})
#             df.to_csv(path, index=False)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
    
#     filename = sys.argv[1]
#     df = pd.read_csv(filename, sep='\t')
#     v_trans = df.iloc[:, 1].values
#     a_trans = df.iloc[:, 2].values
#     # print(f"v_trans: {v_trans}")
#     # print(f"a_trans: {a_trans}")
    
#     ex = PekariaFitApp(v_trans, a_trans)
#     ex.show()
#     sys.exit(app.exec_())