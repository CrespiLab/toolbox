# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

loaded_data = None
loaded_spectra = None

def load_spectra(file_name, file_ext, file_desc):
    if file_ext == ".csv":
        separator = ','
    elif file_ext in (".txt",".dat"):
        separator = '\t'
    
    df = pd.read_csv(file_name, sep=separator)
    return df

def load_spectrum(loaded_data, wavenumbers_only=False):
    if wavenumbers_only:
        wavenumbers = loaded_data.iloc[:, 0].values ## numpy array
        wavelengths = wavelengths_from_wavenumbers(wavenumbers) ## convert wavenumbers to wavelength ## numpy array
        spectrum = loaded_data.iloc[:, 1].values ## numpy array
        if wavenumbers[0] < wavenumbers[1]: ## if wavenumber values increase downward (i.e. the wrong way around)
            wavenumbers = wavenumbers[::-1]
            wavelengths = wavelengths[::-1]
            spectrum = spectrum[::-1]
    else:
        if len(loaded_data.columns) == 2:
            wavelengths = loaded_data.iloc[:, 0].values ## numpy array
            wavenumbers = wavenumbers_from_wavelengths(wavelengths) ## convert wavelength to wavenumbers ## numpy array
            spectrum = loaded_data.iloc[:, 1].values ## numpy array
        elif len(loaded_data.columns) == 3: ## Spectragryph format (wavelength, wavenumbers, Abs)
            wavelengths = loaded_data.iloc[:, 0].values ## numpy array
            wavenumbers = loaded_data.iloc[:, 1].values ## obtained from file ## numpy array
            spectrum = loaded_data.iloc[:, 2].values ## numpy array
        else:
            print("something wrong with loaded data (number of columns is not 2 or 3)")
            ##!!! ADD TO output_console (use message)
    
    return wavelengths, wavenumbers, spectrum

def wavenumbers_from_wavelengths(wavelengths):
    wavenumbers = 1e7/wavelengths
    return wavenumbers

def wavelengths_from_wavenumbers(wavenumbers):
    wavelengths = 1e7/wavenumbers
    return wavelengths

def check_not_empty(thing):
    ''' Check if a list or numpy array exists (i.e. is not empty) '''
    try:
        if type(thing) == list:
            if not thing:
                thing_exists = False
            else:
                thing_exists = True
        elif type(thing) == np.ndarray:
            if thing.size == 0:
                thing_exists = False
            else:
                thing_exists = True
        else:
            print("this thing is something else")
        return thing_exists
    except Exception as e:
        print(f"FAILED to check whether list or numpy array exists: {e}") ##!!! send this to message console somehow (need to make this module a class?)
