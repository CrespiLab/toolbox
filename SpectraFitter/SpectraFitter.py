# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QFileDialog,    
    # QMessageBox,
    QApplication, QMainWindow, 
)
from SpectraFitter.tools.FitPekarianGaussianHybrid import iterative_pekaria_fit
from SpectraFitter.tools.FitWorker_Auto_SingleSpectrum import FitWorker
import SpectraFitter.tools.load_data as LoadData
from SpectraFitter.UIs.MainWindow import Ui_MainWindow

def main():
    class PekariaFitApp(QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(PekariaFitApp, self).__init__()
            self.setupUi(self)
            
            self.max_peaks = 10
            self.threshold_percent = 0.1
            self.k_max = 10
            self.center_deviation_threshold = 100
            self.manual_mode = True
            self.worker = None
            self.initUI()
        
        def initUI(self):
            self.radioButton_Single.toggled.connect(self.handle_radio_selection_DataType)
            self.radioButton_FirstLast.toggled.connect(self.handle_radio_selection_DataType)
            self.radioButton_Batch.toggled.connect(self.handle_radio_selection_DataType)

            self.LoadButton_Single.clicked.connect(self.load_single_spectrum) ##!!! ADD
            self.LoadButton_FirstLast.clicked.connect(self.load_full_dataset)
            self.LoadButton_Batch.clicked.connect(self.load_batch) ##!!! ADD SEPARATE FUNCTION
            
            self.Button_FitMode.clicked.connect(self.toggle_mode)

            self.lineEdit_MaxPFs.textChanged.connect(self.update_MaxPFs)
            self.lineEdit_Threshold.textChanged.connect(self.update_Threshold)
            self.lineEdit_kmax.textChanged.connect(self.update_kmax)
            self.lineEdit_CenterDeviation.textChanged.connect(self.update_CenterDeviationThreshold)
             
            # self.Button_FitSingle ##!!! ADD

            self.Button_Fit_First_fit.clicked.connect(self.fit_first_spectrum)
            self.Button_Fit_First_store.clicked.connect(self.store_first_fit)
            self.Button_Fit_Last_fit.clicked.connect(self.fit_last_spectrum)
            self.Button_Fit_Last_store.clicked.connect(self.store_last_fit)
            self.SaveButton_FirstLast_Fits.clicked.connect(self.save_both_fits_to_csv)
            self.Button_Fit_FirstLastWeights.clicked.connect(self.fit_all_spectra_with_weights)

            ############################################################
            ### Plot Areas ###
            ##!!! change to using class MplCanvas in separate .py
            self.fig_fit, self.ax_fit = plt.subplots(figsize=(8, 4))
            self.canvas_fit = FigureCanvas(self.fig_fit)
            self.verticalLayout_Plot.addWidget(self.canvas_fit)

            self.fig_res, self.ax_res = plt.subplots(figsize=(8, 2))
            self.canvas_res = FigureCanvas(self.fig_res)
            self.verticalLayout_Plot.addWidget(self.canvas_res)
            ############################################################
            
            self.Button_CancelFit.clicked.connect(self.cancel_fit)
            
            self.output_console = self.textEdit_OutputConsole
            
            #### INITIALISATION ####
            self.SetTextfields() # Set text fields defaults
            self.handle_radio_selections()

        ############################################################
        ############################################################

        def handle_radio_selections(self):
            self.handle_radio_selection_DataType()

        def handle_radio_selection_DataType(self):
            if self.radioButton_Single.isChecked(): # Single
                self.LoadButton_Single.setEnabled(True)
                self.LoadButton_FirstLast.setEnabled(False)
                self.LoadButton_Batch.setEnabled(False)
                
                self.Button_FitSingle.setEnabled(True)
                self.labelDescr_Fit_First.setEnabled(False)
                self.Button_Fit_First_fit.setEnabled(False)
                self.Button_Fit_First_store.setEnabled(False)
                self.labelDescr_Fit_Last.setEnabled(False)
                self.Button_Fit_Last_fit.setEnabled(False)
                self.Button_Fit_Last_store.setEnabled(False)
                self.SaveButton_FirstLast_Fits.setEnabled(False)
                self.Button_Fit_FirstLastWeights.setEnabled(False)
                self.Button_FitBatch.setEnabled(False)

            if self.radioButton_FirstLast.isChecked(): # First-Last-Weighted
                self.LoadButton_Single.setEnabled(False)
                self.LoadButton_FirstLast.setEnabled(True)
                self.LoadButton_Batch.setEnabled(False)
                
                self.Button_FitSingle.setEnabled(False)
                self.labelDescr_Fit_First.setEnabled(True)
                self.Button_Fit_First_fit.setEnabled(True)
                self.Button_Fit_First_store.setEnabled(True)
                self.labelDescr_Fit_Last.setEnabled(True)
                self.Button_Fit_Last_fit.setEnabled(True)
                self.Button_Fit_Last_store.setEnabled(True)
                self.SaveButton_FirstLast_Fits.setEnabled(True)
                self.Button_Fit_FirstLastWeights.setEnabled(True)
                self.Button_FitBatch.setEnabled(False)
                
            if self.radioButton_Batch.isChecked(): # Batch
                self.LoadButton_Single.setEnabled(False)
                self.LoadButton_FirstLast.setEnabled(False)
                self.LoadButton_Batch.setEnabled(True)
                
                self.Button_FitSingle.setEnabled(False)
                self.labelDescr_Fit_First.setEnabled(False)
                self.Button_Fit_First_fit.setEnabled(False)
                self.Button_Fit_First_store.setEnabled(False)
                self.labelDescr_Fit_Last.setEnabled(False)
                self.Button_Fit_Last_fit.setEnabled(False)
                self.Button_Fit_Last_store.setEnabled(False)
                self.SaveButton_FirstLast_Fits.setEnabled(False)
                self.Button_Fit_FirstLastWeights.setEnabled(False)
                self.Button_FitBatch.setEnabled(True)

        def SetTextfields(self):
            """ Display experimental parameters in text fields """
            # self.lineEdit_MaxPFs.setText(str(self.max_peaks))
            self.lineEdit_Threshold.setText(str(self.threshold_percent))
            self.lineEdit_kmax.setText(str(self.k_max))
            self.lineEdit_CenterDeviation.setText(str(self.center_deviation_threshold))
            self.update_widgets_mode()
            
        def update_widgets_mode(self):
            if self.manual_mode:
                self.Button_FitMode.setText("Switch to Auto Mode")
                self.textEdit_centers.setEnabled(True)
                # self.textEdit_centers.setPlaceholderText("23000, 28000, 32000")
                self.textEdit_centers.setText("23000, 28000, 32000") ##!!! define with default
                self.labelDescr_MaxPFs.setEnabled(False)
                self.lineEdit_MaxPFs.clear()
                self.lineEdit_MaxPFs.setEnabled(False)
            else:
                self.Button_FitMode.setText("Switch to Manual Mode")
                self.textEdit_centers.setEnabled(False)
                self.textEdit_centers.clear()
                self.textEdit_centers.setPlaceholderText("Auto mode will detect peaks automatically.")
                self.labelDescr_MaxPFs.setEnabled(True)
                self.lineEdit_MaxPFs.setEnabled(True)
                self.lineEdit_MaxPFs.setText(str(self.max_peaks)) 

        def toggle_mode(self):
            self.manual_mode = not self.manual_mode
            self.update_widgets_mode()

        ######### Update methods for the parameters ########################
        def update_MaxPFs(self):
            try:
                self.max_peaks = int(self.lineEdit_MaxPFs.text())  # Convert the input to a float
            except ValueError:
                pass  # Handle the case where the input is not a valid number

        def update_Threshold(self):
            try:
                self.threshold_percent = float(self.lineEdit_Threshold.text())  # Convert the input to a float
            except ValueError:
                pass  # Handle the case where the input is not a valid number

        def update_kmax(self):
            try:
                self.k_max = int(self.lineEdit_kmax.text())  # Convert the input to a float
            except ValueError:
                pass  # Handle the case where the input is not a valid number

        def update_CenterDeviationThreshold(self):
            try:
                self.center_deviation_threshold = float(self.lineEdit_CenterDeviation.text())  # Convert the input to a float
            except ValueError:
                pass  # Handle the case where the input is not a valid number

        ###########################################################################
        def load_single_spectrum(self):
            self.load_file("Single Spectrum")

        def load_full_dataset(self):
            self.load_file("Full Dataset")

        def load_batch(self):
            self.load_file("Batch")

        def load_file(self, file_desc):
            """Load a file based on the file description."""
            try:
                options = QFileDialog.Options()
                ## File dialog for selecting files
                file_name, _ = QFileDialog.getOpenFileName(self, f"Load {file_desc} File", "",
                                                           "CSV, DAT Files (*.csv *dat);;DAT Files (*.dat);;All Files (*)", 
                                                           options=options)
                if not file_name:
                    # QMessageBox.warning(self, "Error", f"No {file_desc} file selected")
                    self.output_console.append(f"No {file_desc} file selected")

                    return
    
                ##################################################
                ## Store the file path in the appropriate attribute based on the file type/extension
                file_ext = os.path.splitext(file_name)[1].lower()
                
                ## Load data dependent on file_ext
                LoadData.loaded_data = LoadData.load_spectra(file_name, file_ext, file_desc)
                
                if file_desc == "Single Spectrum":
                    ##!!! ADD CODE
                    self.output_console.append(f"File Type {file_desc} not available yet")
                    print("code for loading a single spectrum")
                
                elif file_desc == "Full Dataset":
                    (LoadData.loaded_wavelengths, 
                     LoadData.loaded_wavenumbers, 
                     LoadData.loaded_spectra,
                     LoadData.loaded_first_spectrum,
                     LoadData.loaded_last_spectrum,
                     LoadData.loaded_number_of_spectra) = LoadData.load_full(LoadData.loaded_data)
                elif file_desc == "Batch":
                    self.output_console.append(f"File Type {file_desc} not available yet")
                else:
                    pass
                self.output_console.append(f"{file_desc} file {file_name} loaded successfully!")

            except Exception as e:
                self.output_console.append(f"Failed to load {file_desc} file {file_name}: {e}")
        
        #####################################################################
        #####################################################################

        def set_fit_parameters(self):
            ''' Calculate necessary variables for the fit. '''
            self.threshold_fraction = self.threshold_percent/100
            self.threshold_absolute = self.threshold_fraction * np.max(self.abs)

        def fit_first_spectrum(self):
            if LoadData.loaded_spectra is None:
                self.output_console.append("⚠️ No data loaded.")
                return
            self.type_of_spectrum = "weights_first"
            
            self.wavenumbers = LoadData.loaded_wavenumbers
            self.abs = LoadData.loaded_first_spectrum
            self.number_of_spectra = LoadData.loaded_number_of_spectra
            self.index = 0
            self.set_fit_parameters()
            
            if self.manual_mode:
                centers = self.get_centers()
                try:
                    popt, _, v_fit, a_fit, comps = iterative_pekaria_fit(self.wavenumbers, self.abs,
                                                                         centers, k_max=self.k_max)
                    fitted_centers = [popt[i * 6 + 2] for i in range(len(centers))]
                    # residuals = a_first - np.interp(self.v, v_fit, a_fit)
                    residuals = self.abs - a_fit
                    self.plot_fit_step(v_fit, a_fit, comps, centers, fitted_centers, residuals,
                                       self.index, self.number_of_spectra, self.abs)
                    self.temp_fit_first = a_fit
                    self.output_console.append("✅ First spectrum fit complete.")
                except Exception as e:
                    self.output_console.append(f"❌ First spectrum fit error: {e}")
            else:
                self.worker = FitWorker(
                    self.wavenumbers, self.abs, self.max_peaks, self.threshold_absolute,
                    self.k_max, self.center_deviation_threshold
                )
                self.worker.progress.connect(self.output_console.append)
                self.worker.finished.connect(self.on_fit_finished)
                self.worker.interrupted.connect(self.output_console.append)
                self.worker.start()
    
        def fit_last_spectrum(self):
            if LoadData.loaded_spectra is None:
                self.output_console.append("⚠️ No data loaded.")
                return
            self.type_of_spectrum = "weights_last"
            
            self.wavenumbers = LoadData.loaded_wavenumbers
            self.abs = LoadData.loaded_last_spectrum
            self.number_of_spectra = LoadData.loaded_number_of_spectra
            self.index = self.number_of_spectra-1
            self.set_fit_parameters()
            
            if self.manual_mode:
                centers = self.get_centers()
                try:
                    popt, _, v_fit, a_fit, comps = iterative_pekaria_fit(self.wavenumbers, self.abs,
                                                                         centers, k_max=self.k_max)
                    fitted_centers = [popt[i * 6 + 2] for i in range(len(centers))]
                    #residuals = a_last - np.interp(self.v, v_fit, a_fit)
                    residuals = self.abs - a_fit
                    self.plot_fit_step(v_fit, a_fit, comps, centers, fitted_centers, residuals, 
                                       self.index, self.number_of_spectra, self.abs)
                    self.temp_fit_last = a_fit
                    self.output_console.append("✅ Last spectrum fit complete.")
                except Exception as e:
                    self.output_console.append(f"❌ Last spectrum fit error: {e}")
            else:
                self.worker = FitWorker(
                    self.wavenumbers, self.abs, self.max_peaks, self.threshold_absolute,
                    self.k_max, self.center_deviation_threshold,
                )
                self.worker.progress.connect(self.output_console.append)
                self.worker.finished.connect(self.on_fit_finished)
                self.worker.interrupted.connect(self.output_console.append)
                self.worker.start()
    
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
            self.ax_fit.clear()
            self.ax_fit.plot(self.wavenumbers, a_exp, label="Experimental", color='black')
            for i, comp in enumerate(comps[:-1]):
                label = f"PF{i+1}: guess={centers[i]:.0f}, fit={fitted_centers[i]:.0f}"
                self.ax_fit.plot(v_fit, comp, '--', label=label)
            self.ax_fit.plot(v_fit, comps[-1], '--', label="Tail")
            self.ax_fit.plot(v_fit, a_fit, label="Fit", color='orange')
            self.ax_fit.set_title(f"Fit Result (Spectrum {index+1}/{total}) (Manual)")
            self.ax_fit.legend()
            self.ax_fit.invert_xaxis()
            self.canvas_fit.draw()
    
            self.ax_res.clear()
            self.ax_res.plot(self.wavenumbers, residuals, color='red')
            self.ax_res.axhline(0, color='black', linestyle='--')
            self.ax_res.set_title("Residuals")
            self.ax_res.invert_xaxis()
            self.canvas_res.draw()
        
        def plot_results(self, v_fit, a_fit, comps, centers, residuals):
            ##!!! MERGE THIS PLOT FUNCTION WITH plot_fit_step
            index = self.index
            total = self.number_of_spectra
            
            self.ax_fit.clear()
            self.ax_fit.plot(self.wavenumbers, self.abs, label="Experimental", color='black', linewidth=1.2)
            for i, comp in enumerate(comps[:-1], 1):
                self.ax_fit.plot(v_fit, comp, '--', label=f"PF{i} (~{centers[i-1]:.0f})")
            self.ax_fit.plot(v_fit, comps[-1], '--', label="Gaussian Tail")
            self.ax_fit.plot(v_fit, a_fit, label="Total Fit", color='orange', linewidth=1.5)
            self.ax_fit.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_fit.set_ylabel("Epsilon")
            
            if self.type_of_spectrum in ["weights_first", "weights_last"]:
                self.ax_fit.set_title(f"Fit Result (Spectrum {index+1}/{total})" + (" (Auto)" if not self.manual_mode else " (Manual)"))
            elif self.type_of_spectrum == "single":
                self.ax_fit.set_title("Fit Result (Single)" + (" (Auto)" if not self.manual_mode else " (Manual)"))
            self.ax_fit.legend()
            self.ax_fit.invert_xaxis()
            self.canvas_fit.draw()
    
            self.ax_res.clear()
            self.ax_res.plot(self.wavenumbers, residuals, color='red')
            self.ax_res.axhline(0, color='black', linestyle='--')
            self.ax_res.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_res.set_ylabel("Residuals")
            self.ax_res.set_title("Residuals (Experimental - Fit)")
            self.ax_res.invert_xaxis()
            self.canvas_res.draw()
        
        def cancel_fit(self):
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.output_console.append("❌ Cancel requested...")
    
        def on_fit_finished(self, result):
            popt, centers, v_fit, a_fit, residuals, comps = result
            self.assign_temp_fit(a_fit)
            self.plot_results(v_fit, a_fit, comps, centers, residuals)
            interp_residuals = np.interp(v_fit, self.wavenumbers, residuals)
            self.last_fit_data = pd.DataFrame({
                'Wavenumber': v_fit,
                'Fit': a_fit,
                **{f'PF{i+1}': comps[i] for i in range(len(comps)-1)},
                'GaussianTail': comps[-1],
                'Residuals': interp_residuals,
                'Spectrum': self.type_of_spectrum
            })
        
        def assign_temp_fit(self, a_fit):
            if self.type_of_spectrum == "weights_first":
                self.temp_fit_first = a_fit
            elif self.type_of_spectrum == "weights_last":
                self.temp_fit_last = a_fit
        
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
            centers_str = self.textEdit_centers.toPlainText()
            if not centers_str.strip():
                centers_str = self.textEdit_centers.placeholderText()
            try:
                centers = [float(c.strip()) for c in centers_str.split(',')]
                return centers
            except:
                self.show_error("Invalid centers input!")
                return
    
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

        def show_error(self, message):
            self.ax_fit.clear()
            self.ax_fit.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
            self.canvas_fit.draw()

    #########################################################
    app = QApplication.instance() or QApplication(sys.argv)
    ex = PekariaFitApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
