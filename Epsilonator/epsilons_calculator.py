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
#%% Import data
import pandas as pd
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import Averaging.error_propagation as ErrorPropagation

import src.process_spectra as ProcessSpectra
import PSS_Calculator.tools.process_spectra as PSS_ProcessSpectra

data_folder = r"ExampleData" ## data folder
results_folder = r"ExampleResults" ## results folder

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

def import_data(folder, file, ext='.csv'):
    if ext=='.dat':
        df = pd.read_csv(rf"{folder}/{file}", sep='\t',
                         usecols=lambda x: x not in ["Wavenumbers [1/cm]"])
    else:
        df = pd.read_csv(rf"{folder}/{file}")

    if len(df.columns) > 4: ## if there is a 5th column (e.g. spectragryph adds a comma at the end of a .csv file and therefore pandas reads a final empty column)
        df = df.drop(df.columns[-1], axis=1)
    return df

datasets_orig = {}
for i in data_files:
    datasets_orig[i] = import_data(data_folder, data_files[i])

def cut_spectrum(low, high, df):
    ''' slice between low nm and high nm'''
    df_cut = df.loc[(df['Wavelength [nm]'] > low) & (df['Wavelength [nm]'] < high)] 
    return df_cut

datasets = {}
for df in datasets_orig:
    datasets[df] = cut_spectrum(320, 600, datasets_orig[df])

############
pss_filename = "stable_PSS_No1.dat"
df_Stable_PSS = import_data(data_folder, pss_filename, ext='.dat')
df_Stable_PSS.columns = ["Wavelength [nm]",
              "Absorbance Stable",
              "Absorbance PSS"]
PSS_ratio = [39,61] ## stable/metastable at PSS
############
#%% Linear Regression to obtain epsilon of each dataset
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
    y = y[0] # turn into one-dimensional array
    
    (slope, intercept, 
     r_value, p_value, std_err) = stats.linregress(x, y)
    regression_line = slope * x + intercept
    slope = int(np.round(slope)) # round to integer
    std_err = int(np.round(std_err)) # round to integer
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
             color='darkorange', label = f"slope: {slopes[df]}"+r"$\pm$"f"{std_err_slopes[df]}"
             +"\n"+
             f"intercept: {intercept:.2f}\n"+
             r"R${^2}$"+f": {r_value**2:.2f}")

    ax.set_xlabel(r'c$\cdot$l (mol$\cdot$L$^{-1}\cdot$cm)')
    
    if df == '1':
        ax.set_ylabel(r'Absorbance at $\lambda_{max}$')
    
    ax.legend()

savefile_path = fr"{results_folder}/lin_reg"
savefile_svg = savefile_path+".svg"
savefile_png = savefile_path+".png"
fig.savefig(savefile_svg,bbox_inches="tight")
fig.savefig(savefile_png,bbox_inches="tight")

# plt.show()

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
 epsilons_mean['Error_averaged'],
 epsilons_mean['Reduced Chi-Squared']) = ErrorPropagation.calc_final_mean(epsilons_dict_values, ## dict of values from each dataset
                                                                      epsilons_dict_errors, ## dict of errors from each dataset
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

savefile_path = fr"{results_folder}/epsilons"
savefile_svg = savefile_path+".svg"
savefile_png = savefile_path+".png"
fig.savefig(savefile_svg,bbox_inches="tight")
fig.savefig(savefile_png,bbox_inches="tight")

###############################################################################
###### average epsilon at Abs_max obtained from slopes of three datasets ######
# Epsilon_max_mean, Epsilon_max_error = ErrorPropagation.calc_final_mean(slopes, std_err_slopes, 
#                                                verbose="Abs_max"
#                                                )
###############################################################################
#################################### SAVE #####################################
###############################################################################
def save_df_epsilons(folder, filename, 
                           dataframe, column_headers = 'all'):
    df = pd.DataFrame()
    
    if column_headers == 'all':
        column_headers = dataframe.columns
    
    for y in column_headers:
        df[y] = dataframe[y]
        
    # ext=".csv"
    df.to_csv(rf"{folder}/{filename}.csv", index=False) ## comma-separated .csv file
    df.to_csv(rf"{folder}/{filename}.dat", sep='\t',index=False) ## tab-separated .dat file
    
save_df_epsilons(results_folder, "Epsilons_and_Error",
                       epsilons_mean, 
                       ['Wavelength [nm]','Epsilons_averaged', 'Error_averaged'])

save_df_epsilons(results_folder, "Epsilons_plus_minus_Error",
                       epsilons_mean, 
                       ['Wavelength [nm]', 'Epsilons_averaged', 'Epsilons_averaged_plus_error', 'Epsilons_averaged_minus_error'])
###############################################################################
#%% Calculate molar absorptivity spectrum of Metastable isomer
'''
- use calculated average+error of Stable epsilons
- calculate Metastable spectrum using PSS_Calculator
    - to include uncertainty: calculate total concentration using Stable epsilons +/- error

TO DO?
- [ ] Calculate metastable spectrum from more than 1 PSS dataset?

'''
fig = plt.figure(figsize=(10, 6), dpi=600, constrained_layout=True)
gs = gridspec.GridSpec(2, 2, figure=fig)

fig.suptitle("Molar Absorptivity Spectrum of Metastable Isomer Calculated using Averaged Epsilon of Stable")

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

def obtain_concentrations(wavelengths, absorbance_Stable, absorbance_PSS,
                          epsilons_Stable, epsilons_plus_error_S, epsilons_minus_error_S,
                          PSS_percentage_Stable):
    ''' 
    Calculate total concentration and fractions at PSS of Stable and Metastable isomers 
    Use uncertainty on epsilons to obtain three values of total_conc
    '''
    list_plusmin_error = [epsilons_Stable, epsilons_plus_error_S, epsilons_minus_error_S]
    init_conc_S = [] ## avg, avg+error, avg-error
    for eps_spectrum in list_plusmin_error:
        init_conc_S += [trapezoid(absorbance_Stable, x=wavelengths) / trapezoid(eps_spectrum, x=wavelengths)] ## append list
    
    total_conc = init_conc_S ## avg, avg+error, avg-error
    
    PSS_fraction_S = float(PSS_percentage_Stable/100)
    PSS_fraction_MS = 1-PSS_fraction_S
    # PSS_conc_S = PSS_fraction_S * total_conc ## list of avg, avg+error, avg-error
    # PSS_conc_MS = PSS_fraction_MS * total_conc ## list of avg, avg+error, avg-error
    
    print(f"init_conc_Stable: {init_conc_S}")
    print(f"total_conc: {total_conc}")
    print(f"PSS fractions S/MS: {PSS_fraction_S}/{PSS_fraction_MS}")
    # print(f"PSS_conc_MS: {PSS_conc_MS}")
    
    ## PSS_conc_S and PSS_conc_MS not needed
    return PSS_fraction_S, PSS_fraction_MS, total_conc#, PSS_conc_S, PSS_conc_MS

def obtain_metastable_epsilons(wavelengths, absorbance_Stable, absorbance_PSS,
                          epsilons_S, epsilons_plus_error_S, epsilons_minus_error_S,
                          PSS_percentage_Stable):
    ''' 
    Calculate molar absorptivity spectrum of metastable isomer including uncertainties:
        coming from total_conc calculated from the epsilons of the Stable isomer that include uncertainties
    '''
    
    (stable_fraction, metastable_fraction, 
     total_conc) = obtain_concentrations(wavelengths, absorbance_Stable, absorbance_PSS,
                                         epsilons_S, epsilons_plus_error_S, epsilons_minus_error_S,
                                         PSS_percentage_Stable)
    
    (metastable_before_rescaling,
     metastable_rescaled)= PSS_ProcessSpectra.calculate_metastable_spectrum(absorbance_Stable,
                                                                        absorbance_PSS,
                                                                        stable_fraction,
                                                                        metastable_fraction)
    
    epsilons_MS = [metastable_rescaled / i for i in total_conc] ## divide by [avg, avg+error, avg-error] of total concentration

    return total_conc, metastable_before_rescaling, metastable_rescaled, epsilons_MS[0], epsilons_MS[1], epsilons_MS[2]

###############################################################################
###############################################################################
df_to_metastable = pd.DataFrame() ##!!! CHANGE NAME TO DF

## interpolate
df_to_metastable['Wavelength [nm]'] = epsilons_mean["Wavelength [nm]"] ## wavelengths Stable spectral data from epsilons calculation
df_to_metastable['Abs_Stable'] = ProcessSpectra.Interpolate_Spectra(df_to_metastable['Wavelength [nm]'],
                                          df_Stable_PSS["Wavelength [nm]"].values,
                                          df_Stable_PSS["Absorbance Stable"].values) 
df_to_metastable['Abs_PSS'] = ProcessSpectra.Interpolate_Spectra(df_to_metastable['Wavelength [nm]'],
                                       df_Stable_PSS["Wavelength [nm]"].values,
                                       df_Stable_PSS["Absorbance PSS"].values)
df_to_metastable['EpsilonAvg_Stable'] = epsilons_mean["Epsilons_averaged"]
df_to_metastable['EpsilonPlusError_Stable'] = epsilons_mean["Epsilons_averaged_plus_error"]
df_to_metastable['EpsilonMinusError_Stable'] = epsilons_mean["Epsilons_averaged_minus_error"]

## calculate metastable spectrum
(total_conc,
 df_to_metastable['Abs_Metastable_before_rescaling'],
 df_to_metastable['Abs_Metastable_rescaled'],
 df_to_metastable['EpsilonAvg_Metastable'],
 df_to_metastable['EpsilonPlusError_Metastable'],
 df_to_metastable['EpsilonMinusError_Metastable']) = obtain_metastable_epsilons(df_to_metastable['Wavelength [nm]'],
                                       df_to_metastable['Abs_Stable'], df_to_metastable['Abs_PSS'],
                                       df_to_metastable['EpsilonAvg_Stable'], 
                                       df_to_metastable['EpsilonPlusError_Stable'], df_to_metastable['EpsilonMinusError_Stable'],
                                       PSS_percentage_Stable=PSS_ratio[0]) 
total_conc_avg = total_conc[0]
df_to_metastable["EpsilonAvg_Stable * total_conc_avg"] = df_to_metastable['EpsilonAvg_Stable']*total_conc_avg
df_to_metastable["EpsilonAvg_Metastable * total_conc_avg"] = df_to_metastable['EpsilonAvg_Metastable']*total_conc_avg
                               
#######################################
################ PLOT #################
#######################################
list_prop = ['colour', 'linestyle']
df_colours_linestyles = pd.DataFrame({'Abs_Stable': ['black', '-'], 'Abs_PSS': ['orange', '-'],
                                      'Abs_Metastable_before_rescaling': ['green', '-'], 'Abs_Metastable_rescaled': ['red', '-'],
                                      'EpsilonAvg_Stable * total_conc_avg': ['lightgrey', '--'], 'EpsilonAvg_Metastable * total_conc_avg': ['lightsalmon', '--'],
                                      'EpsilonAvg_Stable': ['black', '-'], 'EpsilonPlusError_Stable': ['black','--'],
                                      'EpsilonMinusError_Stable': ['black',':'], 'EpsilonAvg_Metastable': ['red','-'],
                                      'EpsilonPlusError_Metastable': ['red','--'], 'EpsilonMinusError_Metastable': ['red',':']},
                                     index=list_prop)

for y in ['Abs_Stable','Abs_PSS','Abs_Metastable_before_rescaling','Abs_Metastable_rescaled',
          'EpsilonAvg_Stable * total_conc_avg', 'EpsilonAvg_Metastable * total_conc_avg']:
    ax1.plot(df_to_metastable['Wavelength [nm]'], df_to_metastable[y], label=y,
             color=df_colours_linestyles.loc['colour'][y], linestyle=df_colours_linestyles.loc['linestyle'][y])
ax1.legend()

for y in ['EpsilonAvg_Stable', 'EpsilonAvg_Metastable']:
    ax2.plot(df_to_metastable['Wavelength [nm]'], df_to_metastable[y], label=y,
             color=df_colours_linestyles.loc['colour'][y], linestyle=df_colours_linestyles.loc['linestyle'][y])
ax2.legend()
#########################################
for y in ['EpsilonAvg_Stable', 'EpsilonPlusError_Stable', 'EpsilonMinusError_Stable',
          'EpsilonAvg_Metastable','EpsilonPlusError_Metastable','EpsilonMinusError_Metastable']:
    ax3.plot(df_to_metastable['Wavelength [nm]'], df_to_metastable[y], label=y,
             color=df_colours_linestyles.loc['colour'][y], linestyle=df_colours_linestyles.loc['linestyle'][y])
ax3.legend()

#######################################
################ SAVE #################
#######################################
save_df_epsilons(results_folder, "Metastable_Stable_epsilons_and_error",
                       df_to_metastable)
################################
savefile_path = fr"{results_folder}/metastable"
savefile_svg = savefile_path+".svg"
savefile_png = savefile_path+".png"
fig.savefig(savefile_svg,bbox_inches="tight")
fig.savefig(savefile_png,bbox_inches="tight")
#######################################

#%% ##! FINAL STEPS
## - implement into autoQY: run QY optimisation with average, average+pluserror, and average+minuserror
## - write down how epsilon average and error are currently calculated
    ## - any alternatives/improvements?
## - offer alternative if possible (include whole spectra instead of Abs_max)

#%%