# -*- coding: utf-8 -*-
"""
Obtain uncertainties in epsilon values over all wavelengths

METHOD 1 (used here):
- Calculate epsilon (and uncertainty) at Abs_max for each dataset
- Convert Abs to epsilon for each dataset (10 mm spectrum)
    - Also take into account the uncertainty as a relative uncertainty?
- Obtain average and uncertainty in epsilon at every wavelength from the three epsilons spectra

METHOD 2 (to be done):
- Calculate epsilon (and uncertainty) at each wavelength; for each dataset
    - *NEED* baseline-corrected spectra
    - *NEED* solvent-corrected spectra (i.e. a blank needs to have been recorded beforehand in that same cuvette)
        otherwise the short-pathlength spectra show significant deviation towards the UV region.
- Obtain weighted mean and appropriate uncertainty from the three epsilons (± uncertainty) spectra
"""
import pandas as pd
import numpy as np
from scipy import stats
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import error_propagation as ErrorPropagation

# data_folder = r"FULLPATH\ExampleData" ## full path to ExampleData folder

#### CHOOSE ####
PROCESSING="unprocessed"
# PROCESSING="baseline-corrected"
################

if PROCESSING=="unprocessed":
    data_files = {'1': r"absorptionspectra_1.csv",
                  '2': r"absorptionspectra_2.csv",
                  '3': r"absorptionspectra_3.csv"}
elif PROCESSING=="baseline-corrected":
    data_files = {'1': r"absorptionspectra_1_blcorr.csv",
                  '2': r"absorptionspectra_2_blcorr.csv",
                  '3': r"absorptionspectra_3_blcorr.csv"}

conc_sample = 4.90376e-5 ## mol L-1
pathlengths = np.array([0.1, 0.2, 1]) ## cm
conc_times_pathlength = conc_sample*pathlengths ## mol L-1 cm

def import_data(folder, file):
    df = pd.read_csv(rf"{folder}/{file}")
    if len(df.columns) > 4: ## if there is a 5th column (e.g. spectragryph adds a comma at the end of a .csv file and therefore pandas reads a final empty column)
        df = df.drop(df.columns[-1], axis=1)
    return df

datasets_orig = {}
for i in data_files:
    datasets_orig[i] = import_data(data_folder, data_files[i])

#%%
def cut_spectrum(low, high, df):
    ''' slice between low nm and high nm'''
    df_cut = df.loc[(df['Wavelength [nm]'] > low) & (df['Wavelength [nm]'] < high)] 
    return df_cut

datasets = {}
for df in datasets_orig:
    datasets[df] = cut_spectrum(320, 600, datasets_orig[df])

# print(datasets)

#%%
fig = plt.figure(figsize=(12, 8), dpi=600, constrained_layout=True)
gs = gridspec.GridSpec(6, 6, figure=fig)

fig.suptitle(f'Absorption spectra: {PROCESSING}')

axSpectra_1 = fig.add_subplot(gs[0:2, 0:2])
axSpectra_2 = fig.add_subplot(gs[0:2, 2:4])
axSpectra_3 = fig.add_subplot(gs[0:2, 4:6])

axPoints_b_1 = fig.add_subplot(gs[2:4, 0:2])
axPoints_b_2 = fig.add_subplot(gs[2:4, 2:4])
axPoints_b_3 = fig.add_subplot(gs[2:4, 4:6])

axesSpectra = [axSpectra_1, axSpectra_2, axSpectra_3]
axesPoints_b = [axPoints_b_1, axPoints_b_2, axPoints_b_3]

def plot_spectra(df, ax):
    for y in df.iloc[:,1:]:
        ax.plot(df.iloc[:,0], df[y], 
                label=y
                )
    ax.legend()
######################################

maxes_1cm = {} ## dict of Abs maxima in 1 cm for each dataset
wls_max1cm = {} ## dict of wavelengths of Abs maxima in 1 cm for each dataset
maxesAbs = {} ## dict of arrays of Abs maxima at each pathlength for each dataset
for (df, ax) in zip(datasets, axesSpectra):
    plot_spectra(datasets[df], ax)

    maxes_1cm[df] = datasets[df]['Absorbance_10mm'].max()
    wls_max1cm[df] = datasets[df][datasets[df]['Absorbance_10mm'] == maxes_1cm[df]]['Wavelength [nm]'].values[0]
    maxesAbs[df] = datasets[df][datasets[df]['Wavelength [nm]'] == wls_max1cm[df]].iloc[:,1:].values

    ax.set_xlabel('Wavelength (nm)')
    ax.axvline(wls_max1cm[df], color='k', linestyle='--')
    ax.set_title(f"Dataset {df}")
    
    if df == '1':
        ax.set_ylabel('Absorbance')
    ############################

def lin_reg(x, y):
    '''
    Parameters
    ----------
    x : numpy array
        concentration*pathlength (3 elements).
    y : numpy array
        Absorbance values (3 elements).

    Returns
    -------
    
    '''
    (slope, intercept, 
     r_value, p_value, std_err) = stats.linregress(x, y)
    regression_line = slope * x + intercept
    return slope, intercept, r_value, p_value, std_err, regression_line
     
slopes = {} ## slopes of regression line (i.e. epsilon) of Abs_max of each pathlength for each dataset
std_err_slopes = {} ## standard deviation in slopes (i.e. error in epsilon) for each dataset

################################################################
#################### LINEAR REGRESSION PLOTS ###################
################################################################
for (df, ax) in zip(datasets, axesPoints_b):
    (slopes[df], intercept, 
     r_value, p_value, 
     std_err_slopes[df], regression_line) = lin_reg(conc_times_pathlength,
                                                              maxesAbs[df])

    ax.set_title("y = ax + b")
    ax.scatter(conc_times_pathlength, maxesAbs[df])
    ax.plot(conc_times_pathlength, regression_line, 
             color='darkorange', label = f"slope: {slopes[df]:.0f}"+r"$\pm$"f"{std_err_slopes[df]:.0f}"
             +"\n"+
             f"intercept: {intercept:.2f}\n"+
             r"R${^2}$"+f": {r_value**2:.2f}")

    ax.set_xlabel(r'c$\cdot$l (mol$\cdot$L$^{-1}\cdot$cm)')
    
    if df == '1':
        ax.set_ylabel(r'Absorbance at $\lambda_{max}$')
    
    ax.legend()

plt.show()

#%% Epsilons spectra
fig = plt.figure(figsize=(10, 8), dpi=600, constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)

fig.suptitle("Molar Absorptivity Spectra Calculated using Epsilon at " r"$\lambda_{max}$" f" from {PROCESSING} Absorption Spectra")

axEpsilons_all = fig.add_subplot(gs[0, 0])
axEpsilons_all_errors = fig.add_subplot(gs[1, 0])
axEpsilons_avg = fig.add_subplot(gs[0, 1])
axEpsilons_avg_errors = fig.add_subplot(gs[1,1])

def plot_epsilons(df, ax, column_headers,
                  title, ylabel):
    ''' 
    Input: dataframe 
    Plot selected columns in a dataframe versus Wavelength [nm] column
    '''
    
    for y in column_headers:
        ax.plot(df['Wavelength [nm]'], df[y],
                    label=y)
    ax.legend()
    
    ax.set_title(f'{title}')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f'{ylabel}')

def plot_epsilons_with_errorbars(dict_epsilons, key, ax, y, y_error,
                  title):
    df = dict_epsilons[key]
    
    ax.errorbar(df['Wavelength [nm]'], df[y],
                df[y_error],
                label=key)
    ax.legend()
    
    ax.set_title(f'{title}')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel('Epsilon')

def plot_epsilons_errors(ax, y,
                         title, ylabel,
                         dict_of_dfs=None, df_name=None):
    if dict_of_dfs is not None:
        df = dict_of_dfs[df_name]
    else:
        df = df_name
        df_name = ''
        
    ax.scatter(df['Wavelength [nm]'], df[y],
                label=df_name)
    if df_name != '':
        ax.legend()
    
    ax.set_title(f"{title}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f'{ylabel}')

###################################################
epsilons_spectra_1cm = {} ## dictionary of dataframes for epsilons of 10mm spectra

def convert_to_epsilon(df):
    ''' 
    Convert Abs spectra to Epsilons spectra using:
    - slopes: calculated Epsilon at Abs_max
    - maxes_1cm: Abs_max of 1 cm pathlength
    
    Calculate associated uncertainty in Epsilon at each wavelength similarly:
    - std_err_slopes: standard error in calculated Epsilon
    - maxes_1cm
    Result: an error relative to the magnitude of Abs
    
    ##!!! IS THIS THE BEST WAY TO CALCULATE THE ERROR ON THE FULL EPSILON SPECTRUM?
    
    '''
    epsilons_spectra_1cm[df] = pd.DataFrame()
    epsilons_spectra_1cm[df]['Wavelength [nm]'] = datasets[df]['Wavelength [nm]']
    
    #### convert Abs to Epsilon for all 10mm spectra ####
    epsilons_spectra_1cm[df]['Epsilon_10mm'] = (slopes[df] / maxes_1cm[df]) * datasets[df]['Absorbance_10mm']
    
    #### propagate error ####
    epsilons_spectra_1cm[df]['Epsilon_10mm_error'] = abs((std_err_slopes[df] / maxes_1cm[df]) * datasets[df]['Absorbance_10mm']) ## absolute value
    
    plot_epsilons_with_errorbars(epsilons_spectra_1cm, df, axEpsilons_all, 'Epsilon_10mm', 'Epsilon_10mm_error',
                  'Epsilons, all datasets (with error bars)')
    axEpsilons_all.axvline(wls_max1cm[df], color='k', linestyle='--')
    plot_epsilons_errors(axEpsilons_all_errors, 'Epsilon_10mm_error',
                         title='Errors, all datasets', ylabel=r'$\Delta$ Epsilon',
                         dict_of_dfs=epsilons_spectra_1cm, df_name=df)
#####################################################################
for df in datasets:
    convert_to_epsilon(df)
#####################################################################
#### Calculate final epsilons with uncertainties ####

epsilons_dict_values = {}
epsilons_dict_errors = {}

for df in epsilons_spectra_1cm:
    epsilons_dict_values[df] = epsilons_spectra_1cm[df]['Epsilon_10mm'].values
    epsilons_dict_errors[df] = epsilons_spectra_1cm[df]['Epsilon_10mm_error'].values

epsilons_mean = pd.DataFrame()
epsilons_mean['Wavelength [nm]'] = datasets['1']['Wavelength [nm]']

##################
calc_type='scaled' ## recommended: takes into account chi2_red
# calc_type='internal'
# calc_type='simple'
##################

(epsilons_mean['Epsilons_averaged'], 
 epsilons_mean['Error_averaged']) = ErrorPropagation.calc_final_mean(epsilons_dict_values, 
                                                                      epsilons_dict_errors,
                                                                      calc_type=calc_type,
                                                                      verbose=False)

epsilons_mean['Epsilons_averaged_plus_error'] = epsilons_mean['Epsilons_averaged'] + epsilons_mean['Error_averaged']
epsilons_mean['Epsilons_averaged_minus_error'] = epsilons_mean['Epsilons_averaged'] - epsilons_mean['Error_averaged']
                                                                     
plot_epsilons(epsilons_mean, axEpsilons_avg,
              ['Epsilons_averaged', 'Epsilons_averaged_plus_error', 'Epsilons_averaged_minus_error'],
              title='Epsilons, averaged (plus/minus errors)', ylabel='Epsilon')
plot_epsilons_errors(axEpsilons_avg_errors, 'Error_averaged',
                     title=f'Errors, averaged ({calc_type})', ylabel=r'$\Delta$ Epsilon',
                     df_name=epsilons_mean)

plt.show()

#%%
##!!! NEXT STEPS
## - implement into autoQY: run QY optimisation with average, average+pluserror, and average+minuserror
## - write down how epsilon average and error are currently calculated
## - offer alternative if possible (include whole spectra instead of Abs_max)


def save_epsilons_averaged(folder, filename, 
                           dataframe, column_headers):
    df = pd.DataFrame()
    
    for y in column_headers:
        df[y] = dataframe[y]
        
    # path_fit = DataHandling.modify_filename(path, "FirstLastFittedSpectra", ext='.csv')
    
    # df.to_csv(path_fit, index=False)
    ext=".csv"
    df.to_csv(rf"{folder}/{filename}{ext}", index=False)
    
    
save_epsilons_averaged(data_folder, "Epsilons_and_Error",
                       epsilons_mean, 
                       ['Wavelength [nm]','Epsilons_averaged', 'Error_averaged'])

save_epsilons_averaged(data_folder, "Epsilons_plus_minus_Error",
                       epsilons_mean, 
                       ['Wavelength [nm]', 'Epsilons_averaged', 'Epsilons_averaged_plus_error', 'Epsilons_averaged_minus_error'])

#%%
##!!! HOW TO OBTAIN average+uncertainty epsilons for Metastable spectrum?
'''
- use calculated average+error of Stable epsilons
- calculate Metastable spectrum using PSS_Calculator
- calculate Metastable error in the same way: relative to magnitude of Abs

'''


#%% average epsilon at Abs_max obtained from slopes of three datasets

# slopes_array = np.array(list(slopes.values()))
# std_err_slopes_array = np.array(list(std_err_slopes.values()))
Epsilon_max_mean, Epsilon_max_error = ErrorPropagation.calc_final_mean(slopes, std_err_slopes, 
                                               verbose="Abs_max"
                                               )

#%%