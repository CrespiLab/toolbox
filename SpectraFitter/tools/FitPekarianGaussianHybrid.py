# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:11:55 2025
"""
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (QApplication)

# --- Pekarian and helper functions ---

def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def pekaria_auto(v, S, v0, Omega, sigma0, delta, k_max=10):
    v = np.asarray(v, dtype=float)
    spec = np.zeros_like(v)
    exp_neg_S = np.exp(-S)
    for k in range(k_max + 1):
        weight = S ** k / math.factorial(k)
        mu = v0 + k * Omega
        sigma = sigma0 + k * delta
        if sigma <= 0:
            continue
        spec += weight * gaussian(v, mu, sigma)
    return exp_neg_S * spec

def pekaria_scaled_k(v, A, S, v0, Omega, sigma0, delta, k_max):
    return A * pekaria_auto(v, S, v0, Omega, sigma0, delta, k_max)

def pure_gaussian(v, A, v0, sigma):
    return A * gaussian(v, v0, sigma)

def iterative_pekaria_fit(v, a, v_centers, k_max=20):
    n_pf = len(v_centers)
    p0 = []
    bounds_lower = []
    bounds_upper = []
    for vc in v_centers:
        A = max(a) / (n_pf * 1.5)
        S = 0.7
        Omega = 1000
        sigma0 = 600
        delta = 60
        p0.extend([A, S, vc, Omega, sigma0, delta])
        bounds_lower.extend([0, 0.01, vc - 500, 200, 100, 0])
        bounds_upper.extend([1e8, 5.0, vc + 500, 3000, 3000, 300])
    # Gaussian tail params
    A_tail = max(a) / 2
    v_tail = max(v) + 1000
    sigma_tail = 1000
    p0.extend([A_tail, v_tail, sigma_tail])
    bounds_lower.extend([0, max(v), 100])
    bounds_upper.extend([1e8, max(v) + 5000, 3000])
    
    def hybrid_model_dynamic(v, *params):
        comps = []
        idx = 0
        for i in range(n_pf):
            comp = pekaria_scaled_k(v, *params[idx:idx+6], k_max=k_max)
            comps.append(comp)
            idx += 6
        tail = pure_gaussian(v, *params[idx:idx+3])
        comps.append(tail)
        return sum(comps), comps
    
    def model_to_fit(v, *params):
        s, _ = hybrid_model_dynamic(v, *params)
        return s

    popt, pcov = curve_fit(
        model_to_fit, v, a, p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=100000
    )
    
    # v_fit = np.linspace(min(v), max(v), 2000)
    v_fit = np.array(v)
    a_fit, components = hybrid_model_dynamic(v_fit, *popt)
    
    return popt, pcov, v_fit, a_fit, components

def list_to_string(my_list):
    my_list_stringelements = [str(i) for i in my_list] ## list of strings

    my_string = ''
    for i in range(0,len(my_list_stringelements)):
        my_string = my_string+my_list_stringelements[i]
        if i is not len(my_list_stringelements)-1:
            my_string = my_string+", "
    
    return my_string

def residual_driven_peak_finder(v, a, max_peaks, residual_threshold, k_max,
                                center_deviation_threshold,
                                cancel_flag=None, output_console=None):
    centers = []
    popt = None

    for iteration in range(max_peaks):
        if cancel_flag and cancel_flag():
            if output_console:
                output_console.append("❌ Fitting cancelled by user.")
                QApplication.processEvents()
            break

        if len(centers) == 0:
            initial_centers = [v[np.argmax(a)]]
            ##!!! FIND PEAKS FIRST (SCIPY) AND START WITH CA. 5
        else:
            initial_centers = centers

        try:
            popt, pcov, v_fit, a_fit, comps = iterative_pekaria_fit(v, a, initial_centers, k_max=k_max)
        except Exception as e:
            if output_console:
                output_console.append(f"⚠️ Fitting error: {e}")
                QApplication.processEvents()
            break

        interp_fit = interp1d(v_fit, a_fit, kind='linear', bounds_error=False, fill_value="extrapolate")
        residuals_absolute = a - interp_fit(v) ## difference between experimental Abs/epsilon and value from fit
        max_absorbance = np.max(np.abs(a))
        residuals = (residuals_absolute) / max_absorbance ## residuals normalised to maximum Abs
        max_residual = np.max(np.abs(residuals))

        if output_console:
            output_console.append(f"Cycle {iteration+1}: max residual = {max_residual:.4f}")
            QApplication.processEvents()

        if max_residual < residual_threshold:
            if output_console:
                output_console.append("✅ Residual threshold met. Stopping.")
                QApplication.processEvents()
            break

        peaks, properties = find_peaks(residuals, height=residual_threshold, distance=20) ## minimum height of peak in residuals defined by user
        if len(peaks) == 0:
            if output_console:
                output_console.append("No new peaks found in residuals. Stopping.")
                QApplication.processEvents()
            break
        else:
            peak_heights = properties["peak_heights"]
            sorted_peaks = [x for _, x in sorted(zip(peak_heights, peaks))][::-1] ## sort from highest to lowest peak
            new_centers = [v[i] for i in sorted_peaks]
            new_centers_string = list_to_string(new_centers) ## list of strings

            if output_console:
                output_console.append(f"New peaks found in residuals at: {new_centers_string} cm⁻¹")
                QApplication.processEvents()

        stop_stop = "no"
        for new_center in new_centers:
            if any(abs(new_center - c) < center_deviation_threshold for c in centers):
                if output_console:
                    output_console.append(f"🔁 New center {new_center:.1f} too close to existing peaks. Skipping.")
                    QApplication.processEvents()
                if len(new_centers) == 1:
                    stop_stop = "yes"
            else:
                new_center = new_center
                break
        
        if stop_stop == "yes":
            if output_console:
                output_console.append(f"🔁 Remaining new center {new_center:.1f} too close to existing peaks. Stopping.")
                QApplication.processEvents()
            break

        centers.append(new_center)
        if output_console:
            output_console.append(f"➕ Added new center at {new_center:.1f} cm⁻¹")
            QApplication.processEvents()

    return popt, centers, v_fit, a_fit, residuals
