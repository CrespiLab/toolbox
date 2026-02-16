# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QFileDialog,    
    # QMessageBox,
    QApplication, QMainWindow, 
)
import PSS_Calculator.tools.load_data as LoadData
import PSS_Calculator.tools.process_spectra as ProcessSpectra

from PSS_Calculator.UIs.MainWindow import Ui_MainWindow

def main():
    class PSS_Calculator(QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(PSS_Calculator, self).__init__()
            self.setupUi(self)
            
            (self.loaded_wavelengths_Stable, 
             self.loaded_wavenumbers_Stable, 
             self.loaded_spectrum_Stable) = ([], [], [])
            
            (self.loaded_wavelengths_PSS, 
             self.loaded_wavenumbers_PSS, 
             self.loaded_spectrum_PSS) = ([], [], [])
            
            self.wavelengthlimit_low = None
            self.wavelengthlimit_high = None
            
            self.processed_wavelengths = []
            self.processed_wavenumbers = []
            self.processed_spectra = {} # keys: 'Stable', 'PSS'
            self.calculated_spectra = {} # keys: 'Calculated_Metastable', 'Calculated_Metastable_before_rescaling'
            
            self.stable_at_PSS_percent = None
            self.stable_at_PSS_fraction = None
            self.metastable_at_PSS_percent = None
            self.metastable_at_PSS_fraction = None
            
            self.initUI()
        
        def initUI(self):
            self.LoadButton_Stable.clicked.connect(lambda: self.load_file("Stable Spectrum"))
            self.LoadButton_PSS.clicked.connect(lambda: self.load_file("PSS Spectrum"))
            self.checkBox_WavenumbersOnly.stateChanged.connect(self.state_changed_WavenumbersOnly)

            self.lineEdit_StableatPSS.textChanged.connect(self.update_ratio_at_PSS)
            
            self.Button_PlotLoaded.clicked.connect(self.plot_loaded_spectra)
            
            self.lineEdit_CutSpectra_min.textChanged.connect(self.update_wavelength_limits)
            self.lineEdit_CutSpectra_max.textChanged.connect(self.update_wavelength_limits)
            self.Button_Reset_WavelengthLimits.clicked.connect(self.reset_wavelength_limits)
            self.Button_Process_Spectra.clicked.connect(self.process_spectra)

            self.Button_CalculateMetastable.clicked.connect(self.calculate_metastable)
            self.SaveButton_CalculatedMetastable.clicked.connect(self.save_calculated_metastable)

            ############################################################
            ### Plot Areas ###
            ##!!! change to using class MplCanvas in separate .py
            self.fig_main, self.ax_main = plt.subplots(figsize=(8, 4))
            self.canvas_main = FigureCanvas(self.fig_main)
            self.navigation = NavigationToolbar(self.canvas_main, self)
            self.verticalLayout_Plot.addWidget(self.navigation)  # Add the toolbar at the top
            self.verticalLayout_Plot.addWidget(self.canvas_main)
            ############################################################
            self.output_console = self.textEdit_OutputConsole
            
            #### INITIALISATION ####
            self.handle_check_buttons()
            self.handle_buttons()
            ##!!! ADD DEFAULT WAVELENGTH LIMITS TO FIELDS

        ############################################################
        ############################################################

        def handle_check_buttons(self):
            self.wavenumbers_only = self.checkBox_WavenumbersOnly.isChecked()
            self.output_console.append(f"Wavenumbers only (first column): {self.wavenumbers_only}")

        def state_changed_WavenumbersOnly(self):
            self.handle_check_buttons()
            self.reset_loaded_data()
            self.output_console.append("Please re-load your Stable and PSS spectra")

        def handle_buttons(self):
            self.Button_CalculateMetastable.setEnabled(False)
            self.SaveButton_CalculatedMetastable.setEnabled(False)

        ######### Update methods for the parameters ########################
        def update_wavelength_limits(self):
            try:
                self.wavelengthlimit_low = float(self.lineEdit_CutSpectra_min.text())
                self.wavelengthlimit_high = float(self.lineEdit_CutSpectra_max.text())
            except ValueError:
                pass

        def check_input_wavelength_limits(self):
            ''' Check if wavelengthlimit variables have the correct format '''
            if type(self.wavelengthlimit_low) is float and type(self.wavelengthlimit_high) is float:
                message = None
            if self.wavelengthlimit_low > self.wavelengthlimit_high:
                message = "ERROR: lower limit is larger than the higher limit"
            
            ##!!! CHECK IF wavelength limits are outside range of data

            return message
        
        def update_fields_wavelength_limits(self):
            self.lineEdit_CutSpectra_min.setText(str(self.wavelengthlimit_low))
            self.lineEdit_CutSpectra_max.setText(str(self.wavelengthlimit_high))
        
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
                
                self.loaded_data = LoadData.load_spectra(file_name, file_ext, file_desc) ## Load data dependent on file_ext
                self.load_spectra(file_desc)
                self.plot_loaded_spectra()
                self.retrieve_loaded_wavelength_limits()
                self.output_console.append(f"{file_desc} file {file_name} loaded successfully!")
            except Exception as e:
                self.output_console.append(f"Failed to load {file_desc} file {file_name}: {e}")
        
        def load_spectra(self, file_desc):
            try:
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
            except Exception as e:
                self.output_console.append(f"Failed to load {file_desc} file: {e}")

        #####################################################################
        #####################################################################
        
        def plot_loaded_spectra(self):
            try:
                if LoadData.check_not_empty(self.loaded_spectrum_Stable) or LoadData.check_not_empty(self.loaded_spectrum_PSS):
                    self.plot_spectra("loaded")
                else:
                    self.output_console.append("No loaded spectra available.")
            except Exception as e:
                self.output_console.append(f"Failed to plot loaded spectra: {e}")

        def interpolate_spectra(self):
            ''' Interpolate spectra over wavelengths of Stable data '''
            ##!!! IMPROVE: select smallest wavelength range of Stable and PSS data in case they are different
            
            self.processed_wavelengths = self.loaded_wavelengths_Stable
            self.processed_wavenumbers = self.loaded_wavenumbers_Stable

            self.processed_spectra['Stable'] = np.interp(self.processed_wavelengths, 
                                                         self.loaded_wavelengths_Stable,
                                                         self.loaded_spectrum_Stable)

            self.processed_spectra['PSS'] = np.interp(self.processed_wavelengths,
                                                      self.loaded_wavelengths_PSS,
                                                      self.loaded_spectrum_PSS)

        def reset_loaded_data(self):
            try:
                self.loaded_spectrum_Stable = []
                self.loaded_spectrum_PSS = []
            except Exception as e:
                self.output_console.append(f"Failed to reset loaded data: {e}")

        def check_if_loaded_data(self):
            if not (LoadData.check_not_empty(self.loaded_spectrum_Stable) or
                    LoadData.check_not_empty(self.loaded_spectrum_PSS)):
                message = "ERROR: please load both spectra first"
            else:
                message = None
            return message
        
        def reset_wavelength_limits(self):
            message = self.check_if_loaded_data()
            if message is None:
                self.retrieve_loaded_wavelength_limits()
                self.output_console.append("Reset wavelength limits")
            else:
                self.output_console.append(f"{message}")
                return
        
        def retrieve_loaded_wavelength_limits(self):
            if self.loaded_spectrum_Stable is not None and self.loaded_spectrum_PSS is not None:
                self.wavelengthlimit_low = float(max([self.loaded_wavelengths_Stable[0],
                                                self.loaded_wavelengths_PSS[0]]))
                self.wavelengthlimit_high = float(min([self.loaded_wavelengths_Stable[-1],
                                                self.loaded_wavelengths_PSS[-1]]))
                self.update_fields_wavelength_limits()
        
        def retrieve_indices_wavelength_limits(self):
            ''' Find indices that belong to wavelength range '''
            wl_low = self.wavelengthlimit_low
            wl_high = self.wavelengthlimit_high
            low = np.argmin(np.abs(self.processed_wavelengths.astype(float) - wl_low))
            high = np.argmin(np.abs(self.processed_wavelengths.astype(float) - wl_high))
            return low, high
        
        def cut_spectra(self):
            low, high = self.retrieve_indices_wavelength_limits()

            self.processed_wavelengths = self.processed_wavelengths[low:high] ## cut to range
            self.processed_wavenumbers = LoadData.wavenumbers_from_wavelengths(self.processed_wavelengths) ## cut to range

            self.processed_spectra['Stable'] = self.processed_spectra['Stable'][low:high] ## cut to wavelength range
            self.processed_spectra['PSS'] = self.processed_spectra['PSS'][low:high] ## cut to wavelength range

        def process_spectra(self):
            ''' Processing '''
            message = self.check_if_loaded_data()
            
            if message is None:
                message = self.check_input_wavelength_limits()
            else:
                self.output_console.append(f"{message}")
                return

            if message is None:
                try:
                    self.interpolate_spectra() ## Interpolation
                    self.cut_spectra()
    
                    if 'Stable' in self.processed_spectra and 'PSS' in self.processed_spectra:
                        self.output_console.append(f"Processed spectra from {self.wavelengthlimit_low}-{self.wavelengthlimit_high} nm")
                        self.plot_spectra("processed")
    
                        self.Button_CalculateMetastable.setEnabled(True) ## Activate Calculate button
                        self.SaveButton_CalculatedMetastable.setEnabled(False) ## Deactivate Save button
                    
                except Exception as e:
                    self.Button_CalculateMetastable.setEnabled(False) ## Deactivate Calculate button
                    self.SaveButton_CalculatedMetastable.setEnabled(False) ## Deactivate Save button
                    self.output_console.append(f"ERROR: Processing of spectra failed: {e}.")
                    return
            else:
                self.output_console.append(f"{message}")

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
                stable_spectrum = self.processed_spectra['Stable']
                PSS_spectrum = self.processed_spectra['PSS']
                stable_fraction = self.stable_at_PSS_fraction
                metastable_fraction = self.metastable_at_PSS_fraction

                (metastable_before_rescaling,
                 metastable_rescaled)= ProcessSpectra.calculate_metastable_spectrum(stable_spectrum,
                                                                                    PSS_spectrum,
                                                                                    stable_fraction,
                                                                                    metastable_fraction)

                self.calculated_spectra['Calculated_Metastable_before_rescaling'] = metastable_before_rescaling
                self.calculated_spectra['Calculated_Metastable'] = metastable_rescaled

                self.output_console.append("Calculation of Metastable spectrum successful")
                self.plot_spectra("calculated")
                self.SaveButton_CalculatedMetastable.setEnabled(True)
                
                ##!!! ADD WARNING when calculated Metastable spectrum has (significant) negative values, indicating an error in the used ratio
                
            except Exception as e:
                self.Button_CalculateMetastable.setEnabled(True)
                self.SaveButton_CalculatedMetastable.setEnabled(False)
                self.output_console.append(f"Calculation failed: {e}.")
                return

        def plot_spectra(self, mode):
            self.ax_main.clear()
            
            x_mode = "wavelengths"
            
            if mode == "loaded":
                if self.loaded_spectrum_Stable is not None:
                    x = self.loaded_wavelengths_Stable
                    y = self.loaded_spectrum_Stable
                    self.ax_main.plot(x, y, label='Stable')
                if self.loaded_spectrum_PSS is not None:
                    x = self.loaded_wavelengths_PSS
                    y = self.loaded_spectrum_PSS
                    self.ax_main.plot(x, y, label='PSS')
                self.ax_main.set_title("Loaded Spectra")
                self.output_console.append("Plot loaded spectra.")
            
            elif mode == "processed":
                x = self.processed_wavelengths
                if 'Stable' in self.processed_spectra:
                    y = self.processed_spectra['Stable']
                    self.ax_main.plot(x, y, label='Stable')
                if 'PSS' in self.processed_spectra:
                    y = self.processed_spectra['PSS']
                    self.ax_main.plot(x, y, label='PSS')
                self.ax_main.set_title("Processed Spectra")
                self.output_console.append("Plot processed spectra.")

            elif mode == "calculated":
                x = self.processed_wavelengths
                if 'Stable' in self.processed_spectra:
                    y = self.processed_spectra['Stable']
                    self.ax_main.plot(x, y, label='Stable')
                if 'PSS' in self.processed_spectra:
                    y = self.processed_spectra['PSS']
                    self.ax_main.plot(x, y, label='PSS')
                if 'Calculated_Metastable_before_rescaling' in self.calculated_spectra:
                    y = self.calculated_spectra['Calculated_Metastable_before_rescaling']
                    self.ax_main.plot(x, y, '--', color = 'green',
                                      label=f'Calculated Metastable ({self.metastable_at_PSS_percent:.1f}%) (before rescaling)')
                if 'Calculated_Metastable' in self.calculated_spectra:
                    y = self.calculated_spectra['Calculated_Metastable']
                    self.ax_main.plot(x, y, '-', color = 'green',
                                      label='Calculated Metastable')
                self.ax_main.set_title("Calculated Spectra")
                self.output_console.append("Plot calculated spectra.")

            if x_mode == "wavenumbers":
                self.ax_main.invert_xaxis()
                self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
            else:
                self.ax_main.set_xlabel("Wavelength (nm)")

            self.ax_main.set_ylabel("Epsilon")
            self.ax_main.legend()

            self.canvas_main.draw()
        
        def save_calculated_metastable(self):
            if not 'Calculated_Metastable' in self.calculated_spectra:
                self.output_console.append("ERROR: First calculate the spectrum please.")
                return
            
            wavelengths = self.processed_wavelengths
            wavenumbers = self.processed_wavenumbers
            stable = self.processed_spectra['Stable']
            pss = self.processed_spectra['PSS']
            metastable_before_rescaling = self.calculated_spectra['Calculated_Metastable_before_rescaling']
            metastable = self.calculated_spectra['Calculated_Metastable']

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
                df.drop('Wavenumbers (cm-1)', axis=1).to_csv(path+'.csv', sep=',', index=False) ## csv file (without Wavenumbers)
                df.to_csv(path+'.dat', sep='\t', index=False) ## dat file (including Wavenumbers)

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
