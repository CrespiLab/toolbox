# -*- coding: utf-8 -*-
import numpy as np

def Interpolate_Spectra(reference_wavelengths,
                        other_wavelengths, other_values):
    '''
    Interpolate all spectral data according to wavelengths of reference data,
        so that all datasets use the same wavelengths data.

    Parameters
    ----------
    reference_wavelengths : numpy array
        DESCRIPTION.
    other_wavelengths : numpy array
        DESCRIPTION.
    other_values : numpy array
        DESCRIPTION.

    Returns
    -------
    other_interp : numpy array
        interpolated y-data according to wavelengths of reference data.
    '''
    other_interp = np.interp(reference_wavelengths, 
                             other_wavelengths, 
                             other_values)
    return other_interp
