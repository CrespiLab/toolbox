# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:30:26 2025

"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt
from SpectraFitter.tools.FitPekarianGaussianHybrid import iterative_pekaria_fit, residual_driven_peak_finder

from PyQt5.QtCore import QThread, pyqtSignal
from SpectraFitter.tools.FitWorker_Auto_SingleSpectrum import FitWorker

def main():
    class PekariaFitApp(QWidget):
        def __init__(self, v_data, a_data):
            super().__init__()
            self.v = v_data
            self.a = a_data
            self.manual_mode = True
            self.threshold_percent = 0.1
            self.max_peaks = 10
            self.k_max = 10
            self.center_deviation_threshold = 100
            self.worker = None
            self.initUI()
    
        def initUI(self):
            self.setWindowTitle("Interactive Pekarian Fit")
            layout = QVBoxLayout()
    
            self.mode_button = QPushButton("Switch to Auto Mode")
            self.mode_button.clicked.connect(self.toggle_mode)
            layout.addWidget(self.mode_button)
    
            self.param_count_label = QLabel("Manual mode: enter peak centers")
            layout.addWidget(self.param_count_label)
    
            layout.addWidget(QLabel("Centers (cm⁻¹), comma separated:"))
            self.centers_text = QTextEdit()
            self.centers_text.setPlaceholderText("23000, 28000, 32000")
            self.centers_text.setFixedHeight(50)
            layout.addWidget(self.centers_text)
    
            self.max_peaks_input = QLineEdit(str(self.max_peaks))
            self.threshold_input = QLineEdit(str(self.threshold_percent))
            self.kmax_input = QLineEdit(str(self.k_max))
            self.devthresh_input = QLineEdit(str(self.center_deviation_threshold))
    
            for label_text, widget in [
                ("Max PFs (Auto):", self.max_peaks_input),
                ("Residual Threshold (% of max):", self.threshold_input),
                ("k_max:", self.kmax_input),
                ("Center Deviation Threshold:", self.devthresh_input)
            ]:
                hbox = QHBoxLayout()
                hbox.addWidget(QLabel(label_text))
                hbox.addWidget(widget)
                layout.addLayout(hbox)
    
            self.fit_button = QPushButton("Fit")
            self.fit_button.clicked.connect(self.run_fit)
            layout.addWidget(self.fit_button)
    
            self.cancel_button = QPushButton("Cancel Fit")
            self.cancel_button.clicked.connect(self.cancel_fit)
            layout.addWidget(self.cancel_button)
    
            self.output_console = QTextEdit()
            self.output_console.setReadOnly(True)
            self.output_console.setFixedHeight(100)
            layout.addWidget(QLabel("Fitting Progress:"))
            layout.addWidget(self.output_console)
    
            self.save_button = QPushButton("Save Plotted Data")
            self.save_button.clicked.connect(self.save_data)
            layout.addWidget(self.save_button)
    
            self.fig_fit, self.ax_fit = plt.subplots(figsize=(8, 4))
            self.canvas_fit = FigureCanvas(self.fig_fit)
            layout.addWidget(self.canvas_fit)
    
            self.fig_res, self.ax_res = plt.subplots(figsize=(8, 2))
            self.canvas_res = FigureCanvas(self.fig_res)
            layout.addWidget(self.canvas_res)
    
            self.setLayout(layout)
            self.update_widgets_state()
    
        def toggle_mode(self):
            self.manual_mode = not self.manual_mode
            self.update_widgets_state()
    
        def update_widgets_state(self):
            if self.manual_mode:
                self.mode_button.setText("Switch to Auto Mode")
                self.centers_text.setEnabled(True)
                self.param_count_label.show()
            else:
                self.mode_button.setText("Switch to Manual Mode")
                self.centers_text.setEnabled(False)
                self.param_count_label.hide()
                self.centers_text.clear()
                self.centers_text.setPlaceholderText("Auto mode will detect peaks automatically.")
    
        def run_fit(self):
            self.max_peaks = int(self.max_peaks_input.text())
            self.threshold_percent = float(self.threshold_input.text())/100
            self.k_max = int(self.kmax_input.text())
            self.center_deviation_threshold = float(self.devthresh_input.text())
    
            threshold_absolute = self.threshold_percent * np.max(self.a)
    
            self.ax_fit.clear()
            self.canvas_fit.draw()
            self.ax_res.clear()
            self.canvas_res.draw()
            self.output_console.clear()
    
            if self.manual_mode:
                centers_str = self.centers_text.toPlainText()
                if not centers_str.strip():
                    centers_str = self.centers_text.placeholderText()
                try:
                    centers = [float(c.strip()) for c in centers_str.split(',')]
                except:
                    self.show_error("Invalid centers input!")
                    return
    
                try:
                    popt, pcov, v_fit, a_fit, comps = iterative_pekaria_fit(self.v, self.a, centers, k_max=self.k_max)
                except Exception as e:
                    self.show_error(f"Fit error: {e}")
                    return
                residuals = self.a - np.interp(self.v, v_fit, a_fit)
                self.plot_results(v_fit, a_fit, comps, centers, residuals)
            else:
                self.worker = FitWorker(
                    self.v, self.a, self.max_peaks, threshold_absolute,
                    self.k_max, self.center_deviation_threshold
                )
                self.worker.progress.connect(self.output_console.append)
                self.worker.finished.connect(self.on_fit_finished)
                self.worker.interrupted.connect(self.output_console.append)
                self.worker.start()
    
        def cancel_fit(self):
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.output_console.append("❌ Cancel requested...")
    
        def on_fit_finished(self, result):
            popt, centers, v_fit, a_fit, residuals, comps = result
            self.plot_results(v_fit, a_fit, comps, centers, residuals)
            interp_residuals = np.interp(v_fit, self.v, residuals)
            self.last_fit_data = pd.DataFrame({
                'Wavenumber': v_fit,
                'Fit': a_fit,
                **{f'PF{i+1}': comps[i] for i in range(len(comps)-1)},
                'GaussianTail': comps[-1],
                'Residuals': interp_residuals
            })
    
        def plot_results(self, v_fit, a_fit, comps, centers, residuals):
            self.ax_fit.clear()
            self.ax_fit.plot(self.v, self.a, label="Experimental", color='black', linewidth=1.2)
            for i, comp in enumerate(comps[:-1], 1):
                self.ax_fit.plot(v_fit, comp, '--', label=f"PF{i} (~{centers[i-1]:.0f})")
            self.ax_fit.plot(v_fit, comps[-1], '--', label="Gaussian Tail")
            self.ax_fit.plot(v_fit, a_fit, label="Total Fit", color='orange', linewidth=1.5)
            self.ax_fit.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_fit.set_ylabel("Epsilon")
            self.ax_fit.set_title("Pekarian Fit" + (" (Auto)" if not self.manual_mode else ""))
            self.ax_fit.legend()
            self.ax_fit.invert_xaxis()
            self.canvas_fit.draw()
    
            self.ax_res.clear()
            self.ax_res.plot(self.v, residuals, color='red')
            self.ax_res.axhline(0, color='black', linestyle='--')
            self.ax_res.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_res.set_ylabel("Residuals")
            self.ax_res.set_title("Residuals (Experimental - Fit)")
            self.ax_res.invert_xaxis()
            self.canvas_res.draw()
    
        def save_data(self):
            if hasattr(self, 'last_fit_data'):
                path, _ = QFileDialog.getSaveFileName(self, "Save Fitted Data + Residuals", "fit_data.txt", "Text Files (*.txt)")
                if path:
                    with open(path, 'w', newline='') as f:
                        self.last_fit_data.to_csv(f, sep='\t', index=False)
    
        def show_error(self, message):
            self.ax_fit.clear()
            self.ax_fit.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
            self.canvas_fit.draw()
    ####################################
    app = QApplication(sys.argv)
    filename = sys.argv[1]
    df = pd.read_csv(filename, sep='\t')
    v_trans = df.iloc[:, 1].values
    a_trans = df.iloc[:, 2].values
    ex = PekariaFitApp(v_trans, a_trans)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # filename = sys.argv[1]
    # df = pd.read_csv(filename, sep='\t')
    # v_trans = df.iloc[:, 1].values
    # a_trans = df.iloc[:, 2].values
    # ex = PekariaFitApp(v_trans, a_trans)
    # ex.show()
    # sys.exit(app.exec_())
    main()
