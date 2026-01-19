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
import PSS_Calculator.tools.load_data as LoadData
from PSS_Calculator.UIs.MainWindow import Ui_MainWindow

def main():
    class PSS_Calculator(QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(PSS_Calculator, self).__init__()
            self.setupUi(self)
            
            (self.loaded_wavelengths_Stable, 
             self.loaded_wavenumbers_Stable, 
             self.loaded_spectrum_Stable) = (None, None, None)
            
            (self.loaded_wavelengths_PSS, 
             self.loaded_wavenumbers_PSS, 
             self.loaded_spectrum_PSS) = (None, None, None)
            
            self.processed_wavelengths = None
            self.processed_wavenumbers = None
            self.processed_spectra = {} # keys: 'Stable', 'PSS', 'Calculated_Metastable', 'Calculated_Metastable_before_rescaling'
            
            self.stable_at_PSS_percent = None
            self.stable_at_PSS_fraction = None
            self.metastable_at_PSS_percent = None
            self.metastable_at_PSS_fraction = None
            self.worker = None
            self.initUI()
        
        def initUI(self):
            self.LoadButton_Stable.clicked.connect(self.load_stable_spectrum)
            self.LoadButton_PSS.clicked.connect(self.load_PSS_spectrum)
            self.checkBox_WavenumbersOnly.stateChanged.connect(self.handle_check_buttons)

            self.lineEdit_StableatPSS.textChanged.connect(self.update_ratio_at_PSS)
            
            self.Button_PlotLoaded.clicked.connect(self.plot_loaded_spectra)

            self.Button_CalculateMetastable.clicked.connect(self.calculate_metastable)
            self.SaveButton_CalculatedMetastable.clicked.connect(self.save_calculated_metastable)

            ############################################################
            ### Plot Areas ###
            ##!!! change to using class MplCanvas in separate .py
            self.fig_main, self.ax_main = plt.subplots(figsize=(8, 4))
            self.canvas_main = FigureCanvas(self.fig_main)
            self.verticalLayout_Plot.addWidget(self.canvas_main)

            # self.fig_res, self.ax_res = plt.subplots(figsize=(8, 2))
            # self.canvas_res = FigureCanvas(self.fig_res)
            # self.verticalLayout_Plot.addWidget(self.canvas_res)
            ############################################################
            self.output_console = self.textEdit_OutputConsole
            
            #### INITIALISATION ####
            self.handle_check_buttons()

        ############################################################
        ############################################################
        
        def handle_check_buttons(self):
            if self.checkBox_WavenumbersOnly.isChecked():
                self.wavenumbers_only = True
                self.output_console.append("Option checked for: Wavenumbers only (first column)")
            else:
                self.wavenumbers_only = False
                self.output_console.append("Option not checked for: Wavenumbers only (first column)")

        ######### Update methods for the parameters ########################
        def update_ratio_at_PSS(self):
            try:
                self.stable_at_PSS_percent = float(self.lineEdit_StableatPSS.text())  # Convert the input to a float
                if float(0) <= self.stable_at_PSS_percent <= float(100):
                    self.stable_at_PSS_fraction = self.stable_at_PSS_percent/100
                else:
                    self.stable_at_PSS_fraction = None
            except ValueError as e:
                self.stable_at_PSS_fraction = None
                self.output_console.append(f"Incorrect input: {e}")
            self.check_PSS_fractions()
            
        def check_PSS_fractions(self):
            if self.stable_at_PSS_fraction != None:
                self.metastable_at_PSS_percent = 100-self.stable_at_PSS_percent
                self.lineEdit_MetastableatPSS.setText(str(self.metastable_at_PSS_percent)) ## display calculated value for Metastable at PSS (%)
                self.metastable_at_PSS_fraction = self.metastable_at_PSS_percent/100
                self.sum_of_fractions = self.stable_at_PSS_fraction+self.metastable_at_PSS_fraction
                
                self.output_console.append(f"{self.stable_at_PSS_fraction*100}% of Stable and {self.metastable_at_PSS_fraction*100}% of Metastable at PSS")
                message = "fine"
            else:
                self.lineEdit_MetastableatPSS.setText("Error") ## display calculated value for Metastable at PSS (%)
                self.metastable_at_PSS_fraction = None
                self.sum_of_fractions = None

                self.output_console.append("ERROR: Incorrect input of Stable fraction at PSS")
                message = "issue"
            return message

        ###########################################################################        
        ###########################################################################
        
        def load_stable_spectrum(self):
            self.load_file("Stable Spectrum")
            self.plot_spectra("loaded")

        def load_PSS_spectrum(self):
            self.load_file("PSS Spectrum")
            self.plot_spectra("loaded")

        def load_file(self, file_desc):
            """Load a file based on the file description."""
            try:
                options = QFileDialog.Options()
                file_name, _ = QFileDialog.getOpenFileName(self, f"Load File of {file_desc}", "",
                                                           "CSV, DAT, TXT Files (*.csv *dat *txt);;DAT Files (*.dat);;All Files (*)", 
                                                           options=options)
                if not file_name:
                    self.output_console.append(f"No {file_desc} file selected")

                    return
    
                ##################################################
                ## Store the file path in the appropriate attribute based on the file type/extension
                file_ext = os.path.splitext(file_name)[1].lower()
                
                ## Load data dependent on file_ext
                self.loaded_data = LoadData.load_spectra(file_name, file_ext, file_desc)
                
                if file_desc == "Stable Spectrum":
                    (self.loaded_wavelengths_Stable, 
                     self.loaded_wavenumbers_Stable, 
                     self.loaded_spectrum_Stable) = LoadData.load_spectrum(self.loaded_data, self.wavenumbers_only)
                elif file_desc == "PSS Spectrum":
                    (self.loaded_wavelengths_PSS, 
                     self.loaded_wavenumbers_PSS, 
                     self.loaded_spectrum_PSS) = LoadData.load_spectrum(self.loaded_data, self.wavenumbers_only)

                else:
                    pass
                self.output_console.append(f"{file_desc} file {file_name} loaded successfully!")

            except Exception as e:
                self.output_console.append(f"Failed to load {file_desc} file {file_name}: {e}")
        
        #####################################################################
        #####################################################################
        
        def plot_loaded_spectra(self):
            self.plot_spectra("loaded")

        def interpolate_spectra(self):
            ''' Interpolate spectra over wavelengths of Stable data '''
            self.processed_wavelengths = self.loaded_wavelengths_Stable
            self.processed_wavenumbers = self.loaded_wavenumbers_Stable

            self.processed_spectra['Stable'] = np.interp(self.processed_wavenumbers, 
                                                         self.loaded_wavenumbers_Stable,
                                                         self.loaded_spectrum_Stable)

            self.processed_spectra['PSS'] = np.interp(self.processed_wavenumbers,
                                                      self.loaded_wavenumbers_PSS,
                                                      self.loaded_spectrum_PSS)
            
        def process_spectra(self):
            ## Add any processing before interpolation
            
            ## Interpolation
            self.interpolate_spectra()

        def calculate_metastable(self):
            if self.loaded_spectrum_Stable is None or self.loaded_spectrum_PSS is None:
                self.output_console.append("ERROR: Load Stable and PSS spectra before calculation.")
                return

            message = self.check_PSS_fractions()
            if message == "issue":
                return
            
            if self.sum_of_fractions != float(1):
                self.output_console.append("ERROR: Ratios do not add up to 1")
                return

            try:
                self.process_spectra()
            except Exception as e:
                self.output_console.append(f"ERROR: Processing of spectra failed: {e}.")
                return
            
            try:
                PSS_spectrum = self.processed_spectra['PSS']
                stable_fraction = self.stable_at_PSS_fraction
                stable_spectrum = self.processed_spectra['Stable']
                metastable_before_rescaling = PSS_spectrum - (stable_fraction * stable_spectrum)
                self.processed_spectra['Calculated_Metastable_before_rescaling'] = metastable_before_rescaling
                
                metastable_fraction = self.metastable_at_PSS_fraction
                metastable_rescaled = metastable_before_rescaling / metastable_fraction
                self.processed_spectra['Calculated_Metastable'] = metastable_rescaled
                self.output_console.append("Calculation of Metastable spectrum successful")
                
                self.plot_spectra("processed")
                
                ##!!! ADD WARNING when calculated Metastable spectrum has (significant) negative values, indicating an error in the used ratio
                
            except Exception as e:
                self.output_console.append(f"Calculation failed: {e}.")
                return

        def plot_spectra(self, mode):
            self.ax_main.clear()
            
            if mode == "loaded":
                if self.loaded_spectrum_Stable is not None:
                    (x, y) = (self.loaded_wavenumbers_Stable, self.loaded_spectrum_Stable)
                    self.ax_main.plot(x, y, label='Stable')
                if self.loaded_spectrum_PSS is not None:
                    (x, y) = (self.loaded_wavenumbers_PSS, self.loaded_spectrum_PSS)
                    self.ax_main.plot(x, y, label='PSS')
            
            elif mode == "processed":
                x = self.processed_wavenumbers
                if 'Stable' in self.processed_spectra:
                    y = self.processed_spectra['Stable']
                    self.ax_main.plot(x, y, label='Stable')
                if 'PSS' in self.processed_spectra:
                    y = self.processed_spectra['PSS']
                    self.ax_main.plot(x, y, label='PSS')
                if 'Calculated_Metastable_before_rescaling' in self.processed_spectra:
                    y = self.processed_spectra['Calculated_Metastable_before_rescaling']
                    self.ax_main.plot(x, y, '--', color = 'green',
                                      label=f'Calculated Metastable ({self.metastable_at_PSS_percent:.1f}%) (before rescaling)')
                if 'Calculated_Metastable' in self.processed_spectra:
                    y = self.processed_spectra['Calculated_Metastable']
                    self.ax_main.plot(x, y, '-', color = 'green',
                                      label='Calculated Metastable')

            self.ax_main.invert_xaxis()
            self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_main.set_ylabel("Epsilon")
            self.ax_main.set_title("Spectra")
            self.ax_main.legend()
                ##!!! move legend next to plot area (need gridspec)
            self.canvas_main.draw()
        
        def save_calculated_metastable(self):
            if not 'Calculated_Metastable' in self.processed_spectra:
                self.output_console.append("ERROR: First calculate the spectrum please.")
                return
            
            wavelengths = self.processed_wavelengths
            wavenumbers = self.processed_wavenumbers
            stable = self.processed_spectra['Stable']
            pss = self.processed_spectra['PSS']
            metastable_before_rescaling = self.processed_spectra['Calculated_Metastable_before_rescaling']
            metastable = self.processed_spectra['Calculated_Metastable']

            fullpath, _ = QFileDialog.getSaveFileName(self, "Save Calculated Metastable Spectrum", "",
                                                      "CSV Files (*.csv);;DAT Files (*.dat)")
            path = os.path.splitext(fullpath)[0]
            if path:
                df = pd.DataFrame({
                    'Wavelength (nm)': wavelengths,
                    'Wavenumbers (cm-1)': wavenumbers,
                    'Stable': stable,
                    'PSS': pss,
                    f'Calculated Metastable (before re-scaling ({self.metastable_at_PSS_percent:.1f} pct))': metastable_before_rescaling,
                    'Calculated Metastable': metastable
                })
                df_inv = df[::-1] ## invert dataframe so that wavelength increases downward
                df_inv.drop('Wavenumbers (cm-1)', axis=1).to_csv(path+'.csv', sep=',', index=False) ## csv file (without Wavenumbers)
                df_inv.to_csv(path+'.dat', sep='\t', index=False) ## dat file (including Wavenumbers)
                self.output_console.append(f"Calculated Metastable spectrum saved to {path}.csv and .dat")
            else:
                self.output_console.append("Nothing was saved")
                return

        def show_error(self, message):
            self.ax_main.clear()
            self.ax_main.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
            self.canvas_main.draw()

    #########################################################
    app = QApplication.instance() or QApplication(sys.argv)
    ex = PSS_Calculator()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
