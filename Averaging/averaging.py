# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QFileDialog,    
    # QMessageBox,
    QApplication, QMainWindow, 
)

from Averaging.UIs.MainWindow import Ui_MainWindow
import Averaging.error_propagation as ErrorProp

def main():
    class Averaging(QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(Averaging, self).__init__()
            self.setupUi(self)
            
            self.values = {}
            self.errors = {}
            
            self.results = {'scaled': {},
                            'internal': {},
                            'simple': {}} ## dictionary of results
            
            self.lineEdits_Values = {1: self.lineEdit_Value_1,
                                       2: self.lineEdit_Value_2,
                                       3: self.lineEdit_Value_3}
            self.lineEdits_Errors = {1: self.lineEdit_Error_1,
                                       2: self.lineEdit_Error_2,
                                       3: self.lineEdit_Error_3}
            
            self.initUI()
        
        def initUI(self):
            self.loadButton_Values_Errors.clicked.connect(self.load_file)

            ##!!! add code to allow addition of more fields in case of >3 values
            
            self.DeleteButton_1.clicked.connect(lambda: self.delete_values_errors(1))
            self.DeleteButton_2.clicked.connect(lambda: self.delete_values_errors(2))
            self.DeleteButton_3.clicked.connect(lambda: self.delete_values_errors(3))
            
            self.Button_ClearAll.clicked.connect(self.clear_data_all)
            
            self.Button_CalculateAverage.clicked.connect(self.calculate_average)
            
            ##!!! ADD results fields
            ##!!! ADD button to round results to a certain number of significant figures
            
            self.Button_SaveResults.clicked.connect(self.save_results)

            self.output_console = self.textEdit_OutputConsole
            
            #### INITIALISATION ####
            self.handle_buttons_init()
            self.set_init_values()

        ############################################################
        ############################################################

        def handle_buttons_init(self):
            self.Button_CalculateAverage.setEnabled(True)
            self.Button_SaveResults.setEnabled(False)
            
        def set_init_values(self):
            for i in self.lineEdits_Values:
                self.lineEdits_Values[i].setText("")
            for i in self.lineEdits_Errors:
                self.lineEdits_Errors[i].setText("")

        def delete_values_errors(self, count):
            """"""
            try:
                print("===delete_values_errors===")
                print(f"self.values: {self.values}")
                print(f"count: {count}")
                if count not in self.values or self.errors:
                    self.output_console.append("Cannot remove: data does not exist.")
                    return
        
                self.values.pop(count) # remove value from dictionary
                self.errors.pop(count) # remove error from dictionary
        
                self.lineEdits_Values[count].setText("")
                self.lineEdits_Errors[count].setText("")
            except Exception as e:
                self.output_console.append(f"FAILED to delete data: {e}")

        def clear_data_all(self):
            print("TBA")

        ######### Checks, processing, updates ########################
        def obtain_values_errors(self):
            ''' Check format of Values and Errors input '''
            try:
                for i in self.lineEdits_Values:
                    value_string = self.lineEdits_Values[i].text()
                    if value_string != '':
                        self.values[i] = float(value_string)
                for i in self.lineEdits_Errors:
                    error_string = self.lineEdits_Errors[i].text()
                    if error_string != '':
                        self.errors[i] = float(error_string)
            except ValueError:
                pass
            
        def process_values_errors(self):
            ''' Convert input Values and Errors to correct format '''
            try:
                for i in self.lineEdits_Values:
                    self.values[i] = self.values[i]
                for i in self.lineEdits_Errors:
                    self.errors[i] = self.errors[i]
            except Exception as e:
                self.output_console.append(f"Failed to process values and errors: {e}")

        def update_fields_values_errors(self):
            ''' Upon loading a file of Values and Errors '''
            try:
                for i in self.lineEdits_Values:
                    self.lineEdits_Values[i].setText(str(self.values[i]))
                for i in self.lineEdits_Errors:
                    self.lineEdits_Errors[i].setText(str(self.errors[i]))
            except Exception as e:
                self.output_console.append(f"Failed to update fields: {e}")
        
        ###########################################################################        
        ###########################################################################

        def load_file(self, file_desc):
        #     """Load a file based on the file description."""
            self.output_console.append(f"loading feature TBA")
        
        #     try:
        #         options = QFileDialog.Options()
        #         file_name, _ = QFileDialog.getOpenFileName(self, f"Load File of {file_desc}", "",
        #                                                    "CSV, DAT, TXT Files (*.csv *dat *txt);;DAT Files (*.dat);;All Files (*)", 
        #                                                    options=options)
        #         if not file_name:
        #             self.output_console.append(f"No {file_desc} file selected")
        #             return
        #         ##################################################
        #         ## Store the file path in the appropriate attribute based on the file type/extension
        #         file_ext = os.path.splitext(file_name)[1].lower()
                
        #         self.loaded_data = LoadData.load_spectra(file_name, file_ext, file_desc) ## Load data dependent on file_ext

        #         self.output_console.append(f"{file_desc} file {file_name} loaded successfully!")
        #     except Exception as e:
        #         self.output_console.append(f"Failed to load {file_desc} file {file_name}: {e}")
        
        #####################################################################

        def calculate_average(self):
            ''' Calculate the average and final uncertainty in several ways '''
            self.Button_SaveResults.setEnabled(False)

            self.obtain_values_errors()
            
            ##!!! HOW TO put this into a check function?
            
            if not self.values or not self.errors:
                self.output_console.append("FAILED to calculate the average: dictionary is empty")
                return            
            
            if not len(self.values) == len(self.errors):
                self.output_console.append("FAILED to calculate the average: lengths of values and errors are not the same")
                return            
            
            for (i,j) in zip(self.values, self.errors):
                if self.values[i] == '' or self.errors[j] == '':
                    self.output_console.append("FAILED to calculate the average: one or more inputs is an empty string")
                    return
            
            self.process_values_errors()
            
            ##!!! HOW TO put this into a check function?
            for (i,j) in zip(self.values, self.errors):
                if type(self.values[i]) != float or type(self.errors[j]) != float:
                    self.output_console.append("FAILED to calculate the average: one or more inputs is not a float")
                    return
            
            try:
                
                for i in self.results:
                    (self.results[i]['Mean'],
                     self.results[i]['Uncertainty'],
                     self.results[i]['Reduced Chi-Squared']) = ErrorProp.calc_final_mean(self.values, 
                                                                                          self.errors,
                                                                                          calc_type=i,
                                                                                          verbose=False)
    
                self.output_console.append("=======================================================")
                self.output_console.append("Calculation of final average and uncertainty successful")
                self.output_results()
                self.Button_SaveResults.setEnabled(True)
            except Exception as e:
                self.Button_CalculateAverage.setEnabled(True)
                self.Button_SaveResults.setEnabled(True)
                self.output_console.append(f"Calculation failed: {e}.")
                return

        def output_results(self):
            for calc_type in self.results:
                results = self.results[calc_type]
                if not self.results[calc_type]['Reduced Chi-Squared'] is None:
                    self.output_console.append(f"Average with {calc_type} uncertainty: {results['Mean']:.4f} ± {results['Uncertainty']:.4f} with a χ²ᵥ of {results['Reduced Chi-Squared']:.4f}")
                    # self.output_console.append(f"χ²ᵥ of {results['Reduced Chi-Squared']}")
                else:
                    self.output_console.append(f"Average with {calc_type} uncertainty: {results['Mean']:.4f} ± {results['Uncertainty']:.4f}")

        def save_results(self):
            fullpath, _ = QFileDialog.getSaveFileName(self, "Save Calculated Averages and Uncertainties", "",
                                                      "TXT Files (*.txt)")
            path = os.path.splitext(fullpath)[0]
            if path:
                df = pd.DataFrame(self.results).T ## create dataframe from dictionary: type of averaging as index
                df.to_csv(path+'.txt', sep='\t', index=True) ## tab-separated textfile
                self.output_console.append(f"Calculated Averages and Uncertainties saved to {path}.txt")
            else:
                self.output_console.append("Nothing was saved")
                return


    #########################################################
    app = QApplication.instance() or QApplication(sys.argv)
    ex = Averaging()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
