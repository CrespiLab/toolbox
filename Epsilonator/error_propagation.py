# -*- coding: utf-8 -*-
''' 
Sources:
- Bevington & Robinson “Data Reduction and Error Analysis for the Physical Sciences”
- John R. Taylor – “An Introduction to Error Analysis”
- Particle Data Group (PDG) Review of Particle Physics. Section: “Statistics”
- Skoog, Holler, Crouch – Principles of Instrumental Analysis
- Harris – Quantitative Chemical Analysis
'''
import numpy as np

def calc_weighted_mean(values, sigmas):
    ''' 
    Calculate weighted mean and its internal uncertainty (error the weighted mean),
    as well as the scaled uncertainty that takes into account the reduced chi-squared.
    Best if σᵢ are trustworthy
    Assumes σᵢ are correct
    
    Parameters
    ----------
    values : numpy array
        DESCRIPTION.
    sigmas : numpy array
        DESCRIPTION.
    
    Returns
    -------
    weighted_mean : float
        DESCRIPTION.
    sigma_mean_internal : float
        DESCRIPTION.
    chi2_red : float
        DESCRIPTION.
    sigma_mean_scaled : float
        DESCRIPTION.
    '''
    n = len(values)

    weights = 1 / sigmas**2
    
    weighted_mean = np.sum((weights * values), axis=0) / np.sum(weights, axis=0)
    sigma_mean_internal = np.sqrt(1 / np.sum(weights, axis=0))
    
    chi2 = np.sum((((values - weighted_mean)**2) / sigmas**2), axis=0)
    chi2_red = chi2 / (n - 1)
    
    sigma_mean_scaled = sigma_mean_internal * np.sqrt(chi2_red)
    
    return weighted_mean, sigma_mean_internal, chi2_red, sigma_mean_scaled

def calc_simple_mean(values, sigmas):
    ''' 
    Calculate simple mean and standard deviation of values. 
    Good if σᵢ are uncertain or unavailable.
    sigma_simple (standard deviation / sqrt(n)) gives the overall scatter of the values
    
    Parameters
    ----------
    values : numpy array
        DESCRIPTION.
    sigmas : numpy array
        DESCRIPTION.
    
    Returns
    -------
    simple_mean : float
        DESCRIPTION.
    sigma_simple : float
        DESCRIPTION.
    '''
    n = len(values)
    simple_mean = np.mean(values, axis=0)

    sample_std = np.std(values, ddof=1, axis=0)  # sample standard deviation
    sigma_simple = sample_std / np.sqrt(n)

    return simple_mean, sigma_simple

def calc_final_mean(values, errors, calc_type='scaled', verbose=False):
    '''
    Parameters
    ----------
    values : numpy array
        DESCRIPTION.
    sigmas : numpy array
        DESCRIPTION.
    
    '''
    slopes_array = np.array(list(values.values()))
    std_err_slopes_array = np.array(list(errors.values()))
    
    (weighted_mean, sigma_mean_internal,
     chi2_red, sigma_mean_scaled) = calc_weighted_mean(slopes_array, std_err_slopes_array)

    (simple_mean, sigma_simple) = calc_simple_mean(slopes_array, std_err_slopes_array)
    
    #####################################################################
    ################### Which type of mean and average ##################
    
    #################################
    ## dependent on chi2_red: works for single mean and error values ##
    # if chi2_red > 1:
        # mean = weighted_mean
        # error = sigma_mean_scaled
    # else:
        # mean = weighted_mean
        # error = sigma_mean_internal
    #################################
    
    ## user chooses (scaled is recommended)
    
    if calc_type=='scaled':
        mean = weighted_mean
        error = sigma_mean_scaled
    elif calc_type=='internal':
        mean = weighted_mean
        error = sigma_mean_internal
    elif calc_type=='simple':
        mean = simple_mean
        error = sigma_simple
    
    #####################################################################
    
    if verbose==True:
        print("---- Weighted Mean ----")
        print(f"Mean = {weighted_mean:.4f}")
        print(f'Internal uncertainty = {sigma_mean_internal:.4f} ("error of weighted mean")')
        print(f"Reduced chi-square = {chi2_red:.3f}")
        print(f"Scaled uncertainty = {sigma_mean_scaled:.4f}")
        
        print("\n---- Simple Mean ----")
        print(f"Mean = {simple_mean:.4f}")
        print(f"Std dev / sqrt(n) = {sigma_simple:.4f}")
        
        print("\n---- Final Result ----")
        print(f"χ²ᵥ = {chi2_red:.3f} (an indication of the overlap of values ± uncertainty --"+
              " the higher χ²ᵥ, the less overlap (data scatter is larger than suggested by individual error bars))")
        print(f"- If χ²ᵥ ≈ 1: x̄ _weighted ± internal uncertainty\n {weighted_mean:.4f} ± {sigma_mean_internal:.4f}")
        print(f"- If χ²ᵥ > 1: x̄ _weighted ± scaled uncertainty\n {weighted_mean:.4f} ± {sigma_mean_scaled:.4f}")
        
        print(f"- If sigmas are uncertain: x̄ _normal ± Std dev / sqrt(n)\n {simple_mean:.4f} ± {sigma_simple:.4f}")
    elif verbose=="Abs_max":
        print(f"At Abs_max\nFinal Mean: {mean}\nFinal Error: {error}")
        
    return mean, error
