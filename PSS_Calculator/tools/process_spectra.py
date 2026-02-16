# -*- coding: utf-8 -*-

def calculate_metastable_spectrum(stable_spectrum, PSS_spectrum,
                                  stable_fraction, metastable_fraction):
    '''
    Calculate spectrum of metastable isomer by scaled subtraction.
    
    Parameters
    ----------
    stable_spectrum : numpy array
        DESCRIPTION.
    PSS_spectrum : numpy array
        DESCRIPTION.
    stable_fraction : float
        DESCRIPTION.
    metastable_fraction : float
        DESCRIPTION.

    Returns
    -------
    metastable_before_rescaling : numpy array
        spectrum of metastable isomer * metastable_fraction.
    metastable_rescaled : numpy array
        spectrum of metastable isomer
    '''
    metastable_before_rescaling = PSS_spectrum - (stable_fraction * stable_spectrum)
    metastable_rescaled = metastable_before_rescaling / metastable_fraction
    
    return metastable_before_rescaling, metastable_rescaled
