# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:35:11 2025

@author: jorst136
"""
from PyQt5.QtCore import QThread, pyqtSignal

from SpectraFitter.tools.FitPekarianGaussianHybrid import residual_driven_peak_finder, iterative_pekaria_fit

class FitWorker(QThread):
    finished = pyqtSignal(object)  # returns (popt, centers, v_fit, a_fit, residuals, comps)
    progress = pyqtSignal(str)
    interrupted = pyqtSignal(str)

    def __init__(self, v, a, max_peaks, threshold_absolute, k_max, center_deviation_threshold):
        super().__init__()
        self.v = v
        self.a = a
        self.max_peaks = max_peaks
        self.threshold_absolute = threshold_absolute
        self.k_max = k_max
        self.center_deviation_threshold = center_deviation_threshold
        self._is_running = True

    def run(self):
        def cancel_flag():
            return not self._is_running

        try:
            result = residual_driven_peak_finder(
                self.v, self.a, self.max_peaks, self.threshold_absolute,
                self.k_max, self.center_deviation_threshold,
                cancel_flag=cancel_flag, output_console=self  # redirect messages to self.progress.emit
            )
            if self._is_running:
                popt, centers, v_fit, a_fit, residuals = result
                _, _, _, _, comps = iterative_pekaria_fit(self.v, self.a, centers, k_max=self.k_max)
                self.finished.emit((popt, centers, v_fit, a_fit, residuals, comps))
            else:
                self.interrupted.emit("Fitting was cancelled.")
        except Exception as e:
            self.interrupted.emit(f"Error: {str(e)}")

    def stop(self):
        self._is_running = False

    def append(self, text):
        # Allows passing output_console=self and calling output_console.append in fitting
        self.progress.emit(text)
