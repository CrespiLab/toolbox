# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QFileDialog,    
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from SpectraFitter.tools.batch_FitPekarianGaussianHybrid import iterative_pekaria_fit
    ##!!! FIX importing of data in general module such that the same one can be used for both single and batch versions

import SpectraFitter.tools.load_data as LoadData

def main():
    class PekariaFitApp(QWidget):
        def __init__(self):
            super().__init__()
            self.k_max = 10
            self.initUI()
        
        def initUI(self):
                self.setWindowTitle("Manual Endpoint Fitting GUI")
                layout = QVBoxLayout()

                self.load_data_button = QPushButton("Load Spectra")
                self.load_data_button.clicked.connect(self.load_full_dataset)
                layout.addWidget(self.load_data_button)
    
                layout.addWidget(QLabel("Centers (cm⁻¹), comma separated:"))
                self.centers_text = QTextEdit()
                self.centers_text.setPlaceholderText("23000, 28000, 32000")
                self.centers_text.setFixedHeight(50)
                layout.addWidget(self.centers_text)
    
               # self.max_peaks_input = QLineEdit("5")
               # self.threshold_input = QLineEdit("2.0")
               # self.devthresh_input = QLineEdit("100")
    
                # for label_text, widget in [
                #     ("Max PFs (Auto):", self.max_peaks_input),
                #     ("Residual Threshold (% of max):", self.threshold_input),
                #     ("Center Deviation Threshold:", self.devthresh_input)
                # ]:
                #     hbox = QHBoxLayout()
                #     hbox.addWidget(QLabel(label_text))
                #     hbox.addWidget(widget)
                #     layout.addLayout(hbox)
    
                self.kmax_input = QLineEdit(str(self.k_max))
                hbox_k = QHBoxLayout()
                hbox_k.addWidget(QLabel("k_max:"))
                hbox_k.addWidget(self.kmax_input)
                layout.addLayout(hbox_k)
    
                self.fit_first_button = QPushButton("Fit First Spectrum")
                self.fit_first_button.clicked.connect(self.fit_first_spectrum)
                layout.addWidget(self.fit_first_button)
    
                self.store_first_button = QPushButton("Store Fit for First Spectrum")
                self.store_first_button.clicked.connect(self.store_first_fit)
                layout.addWidget(self.store_first_button)
    
                self.fit_last_button = QPushButton("Fit Last Spectrum")
                self.fit_last_button.clicked.connect(self.fit_last_spectrum)
                layout.addWidget(self.fit_last_button)
    
                self.store_last_button = QPushButton("Store Fit for Last Spectrum")
                self.store_last_button.clicked.connect(self.store_last_fit)
                layout.addWidget(self.store_last_button)
    
                self.fig_fit, self.ax_fit = plt.subplots(figsize=(8, 4))
                self.canvas_fit = FigureCanvas(self.fig_fit)
                layout.addWidget(self.canvas_fit)
    
                self.fig_res, self.ax_res = plt.subplots(figsize=(8, 2))
                self.canvas_res = FigureCanvas(self.fig_res)
                layout.addWidget(self.canvas_res)
    
                self.output_console = QTextEdit()
                self.output_console.setReadOnly(True)
                self.output_console.setFixedHeight(100)
                layout.addWidget(self.output_console)
        
                self.save_both_fits_button = QPushButton("Save First and Last Fits to CSV")
                self.save_both_fits_button.clicked.connect(self.save_both_fits_to_csv)
                layout.addWidget(self.save_both_fits_button)
                
                self.fit_all_weights_button = QPushButton("Fit All Spectra with First/Last Weights")
                self.fit_all_weights_button.clicked.connect(self.fit_all_spectra_with_weights)
                layout.addWidget(self.fit_all_weights_button)
                self.setLayout(layout)

        def load_full_dataset(self):
            self.load_file("Full Dataset")

        def load_file(self, file_desc):
            """Load a file based on the file description."""
            try:
                options = QFileDialog.Options()
                ## File dialog for selecting files
                file_name, _ = QFileDialog.getOpenFileName(self, f"Load {file_desc} File", "",
                                                           "CSV, DAT Files (*.csv *dat);;DAT Files (*.dat);;All Files (*)", 
                                                           options=options)
                if not file_name:
                    QMessageBox.warning(self, "Error", f"No {file_desc} file selected")
                    return
    
                ##################################################
                ## Store the file path in the appropriate attribute based on the file type/extension
                file_ext = os.path.splitext(file_name)[1].lower()
                
                ## Load data dependent on file_ext
                LoadData.loaded_data = LoadData.load_spectra(file_name, file_ext, file_desc)
                
                if file_desc == "Full Dataset":
                    (LoadData.loaded_wavelengths, 
                     LoadData.loaded_wavenumbers, 
                     LoadData.loaded_spectra,
                     LoadData.loaded_first_spectrum,
                     LoadData.loaded_last_spectrum,
                     LoadData.loaded_number_of_spectra) = LoadData.load_full(LoadData.loaded_data)
                else:
                    pass

                QMessageBox.information(self, "Success", f"{file_desc} file loaded successfully!")
            except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load {file_desc} file: {e}")
        
        #####################################################################
        #####################################################################

        def fit_first_spectrum(self):
            if LoadData.loaded_spectra is None:
                self.output_console.append("⚠️ No transient data loaded.")
                return
            wavenumbers = LoadData.loaded_wavenumbers
            a_first = LoadData.loaded_first_spectrum
            number_of_spectra = LoadData.loaded_number_of_spectra
            
            centers = self.get_centers()
            k_max = int(self.kmax_input.text())
            try:
                popt, _, v_fit, a_fit, comps = iterative_pekaria_fit(wavenumbers, a_first, centers, k_max=k_max)
                fitted_centers = [popt[i * 6 + 2] for i in range(len(centers))]
                # residuals = a_first - np.interp(self.v, v_fit, a_fit)
                residuals = a_first - a_fit
                self.plot_fit_step(v_fit, a_fit, comps, centers, fitted_centers, residuals, 0, number_of_spectra, a_first)
                self.temp_fit_first = a_fit
                self.output_console.append("✅ First spectrum fit complete.")
            except Exception as e:
                self.output_console.append(f"❌ First spectrum fit error: {e}")
    
        def fit_last_spectrum(self):
            if LoadData.loaded_spectra is None:
                self.output_console.append("⚠️ No transient data loaded.")
                return
            wavenumbers = LoadData.loaded_wavenumbers
            a_last = LoadData.loaded_last_spectrum
            number_of_spectra = LoadData.loaded_number_of_spectra

            centers = self.get_centers()
            k_max = int(self.kmax_input.text())
            try:
                popt, _, v_fit, a_fit, comps = iterative_pekaria_fit(wavenumbers, a_last, centers, k_max=k_max)
                fitted_centers = [popt[i * 6 + 2] for i in range(len(centers))]
                #residuals = a_last - np.interp(self.v, v_fit, a_fit)
                residuals = a_last - a_fit
                self.plot_fit_step(v_fit, a_fit, comps, centers, fitted_centers, residuals, number_of_spectra-1, number_of_spectra, a_last)
                self.temp_fit_last = a_fit
                self.output_console.append("✅ Last spectrum fit complete.")
            except Exception as e:
                self.output_console.append(f"❌ Last spectrum fit error: {e}")
    
        def store_first_fit(self):
            if hasattr(self, 'temp_fit_first'):
                self.fit_first_stored = self.temp_fit_first
                self.output_console.append("✅ First fit stored.")
            else:
                self.output_console.append("⚠️ No first fit available to store.")
    
        def store_last_fit(self):
            if hasattr(self, 'temp_fit_last'):
                self.fit_last_stored = self.temp_fit_last
                self.output_console.append("✅ Last fit stored.")
            else:
                self.output_console.append("⚠️ No last fit available to store.")
    
        def plot_fit_step(self, v_fit, a_fit, comps, centers, fitted_centers, residuals, index, total, a_exp):
            wavenumbers = LoadData.loaded_wavenumbers

            self.ax_fit.clear()
            self.ax_fit.plot(wavenumbers, a_exp, label="Experimental", color='black')
            for i, comp in enumerate(comps[:-1]):
                label = f"PF{i+1}: guess={centers[i]:.0f}, fit={fitted_centers[i]:.0f}"
                self.ax_fit.plot(v_fit, comp, '--', label=label)
            self.ax_fit.plot(v_fit, comps[-1], '--', label="Tail")
            self.ax_fit.plot(v_fit, a_fit, label="Fit", color='orange')
            self.ax_fit.set_title(f"Fit Result (Spectrum {index+1}/{total})")
            self.ax_fit.legend()
            self.ax_fit.invert_xaxis()
            self.canvas_fit.draw()
    
            self.ax_res.clear()
            self.ax_res.plot(wavenumbers, residuals, color='red')
            self.ax_res.axhline(0, color='black', linestyle='--')
            self.ax_res.set_title("Residuals")
            self.ax_res.invert_xaxis()
            self.canvas_res.draw()
        
        def fit_all_spectra_with_weights(self):
            ''' 
            The previously obtained fits of the first and last spectra are used to obtain "fits" of all spectra by
            combining the first and last spectra in a certain ratio (or weight).
            '''
            if not hasattr(self, 'temp_fit_first') or not hasattr(self, 'temp_fit_last'):
                self.output_console.append("❌ First and last fits must be available.")
                return
            if LoadData.loaded_spectra is None:
                self.output_console.append("❌ No full transient dataset loaded.")
                return
    
            # spectra = LoadData.loaded_spectra.iloc[:, 2:].values.T  # shape: [n_spectra, n_points]
            spectra = LoadData.loaded_spectra.T  # shape: [n_spectra, n_points]
            wavenumbers = LoadData.loaded_wavenumbers
    
            fit_first = self.temp_fit_first
            fit_last = self.temp_fit_last
    
            from scipy.optimize import minimize_scalar
    
            fitted_matrix = []
            residual_matrix = []
    
            for i, a_exp in enumerate(spectra):
                def model(w):
                    return w * fit_last + (1 - w) * fit_first
    
                def residual(w):
                    return np.sum((a_exp - model(w)) ** 2)
    
                try:
                    res = minimize_scalar(residual, bounds=(0, 1), method='bounded')
                    w_opt = res.x
                    fit_result = model(w_opt)
                    residuals = a_exp - fit_result
                    fitted_matrix.append(fit_result)
                    residual_matrix.append(residuals)
                    self.output_console.append(f"✅ Spectrum {i+1}: w = {w_opt:.3f}")
                except Exception as e:
                    self.output_console.append(f"❌ Error on spectrum {i+1}: {e}")
                    fitted_matrix.append(np.full_like(wavenumbers, np.nan))
                    residual_matrix.append(np.full_like(wavenumbers, np.nan))
    
            fitted_matrix = np.array(fitted_matrix).T  # [n_points, n_spectra]
            residual_matrix = np.array(residual_matrix).T  # [n_points, n_spectra]
    
            df_fit = pd.DataFrame({'Wavenumber': wavenumbers})
            df_resid = pd.DataFrame({'Wavenumber': wavenumbers})
    
            for i in range(fitted_matrix.shape[1]):
                df_fit[f'Fit_{i+1}'] = fitted_matrix[:, i]
                df_resid[f'Residual_{i+1}'] = residual_matrix[:, i]
    
            path_fit, _ = QFileDialog.getSaveFileName(self, "Save Fitted Spectra", "", "CSV Files (*.csv)")
            if path_fit:
                df_fit.to_csv(path_fit, index=False)
                self.output_console.append(f"💾 Fitted spectra saved to {path_fit}")
    
            path_resid, _ = QFileDialog.getSaveFileName(self, "Save Residuals", "", "CSV Files (*.csv)")
            if path_resid:
                df_resid.to_csv(path_resid, index=False)
                self.output_console.append(f"💾 Residuals saved to {path_resid}")
    
    
        def get_centers(self):
            centers_str = self.centers_text.toPlainText()
            if not centers_str.strip():
                centers_str = self.centers_text.placeholderText()
            return [float(c.strip()) for c in centers_str.split(',')]
    
    
        def save_both_fits_to_csv(self):
            if not hasattr(self, 'temp_fit_first') or not hasattr(self, 'temp_fit_last'):
                self.output_console.append("❌ Both first and last fits are required to save.")
                return
            
            wavenumbers = LoadData.loaded_wavenumbers

            path, _ = QFileDialog.getSaveFileName(self, "Save Fits", "", "CSV Files (*.csv)")
            if path:
                df = pd.DataFrame({
                    'Wavenumber': wavenumbers,
                    'Fit_First': self.temp_fit_first,
                    'Fit_Last': self.temp_fit_last
                })
                df.to_csv(path, index=False)
                self.output_console.append(f"💾 First and last fits saved to {path}")            
                return
            if LoadData.loaded_spectra is None:
                self.output_console.append("❌ No full transient dataset loaded.")
                return
    
            # spectra = LoadData.loaded_spectra.iloc[:, 2:].values
            spectra = LoadData.loaded_spectra
            v = wavenumbers
            fit_first = np.interp(v, v, self.temp_fit_first)
            fit_last = np.interp(v, v, self.temp_fit_last)
    
            from scipy.optimize import minimize_scalar
    
            fitted_spectra = []
            weights = []
    
            for i, a_exp in enumerate(spectra.T):
                def model(w):
                    return w * fit_last + (1 - w) * fit_first
    
                def residual(w):
                    return np.sum((a_exp - model(w)) ** 2)
    
                try:
                    res = minimize_scalar(residual, bounds=(0, 1), method='bounded')
                    w_opt = res.x
                    weights.append(w_opt)
                    fitted_spectra.append(model(w_opt))
                    self.output_console.append(f"✅ Spectrum {i+1}: w = {w_opt:.3f}")
                except Exception as e:
                    self.output_console.append(f"❌ Error on spectrum {i+1}: {e}")
                    weights.append(np.nan)
                    fitted_spectra.append(np.zeros_like(v))
    
            df_out = pd.DataFrame({'Wavenumber': v})
            for i, spec in enumerate(fitted_spectra):
                df_out[f'Fit_{i+1}'] = spec
            df_out['Weights'] = weights
    
            path, _ = QFileDialog.getSaveFileName(self, "Save Weighted Fits", "", "CSV Files (*.csv)")
            if path:
                df_out.to_csv(path, index=False)
                self.output_console.append(f"💾 Weighted fits saved to {path}")
    #########################################################
    app = QApplication.instance() or QApplication(sys.argv)
    ex = PekariaFitApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
