#!/usr/bin/env python3

import sys, glob, re, os, math, argparse
import numpy as np
np.set_printoptions(precision=10)

def gaussian_old(lambda_max, f, sigma_broadening, spectral_range):
    spectrum_points = np.linspace(spectral_range[0], spectral_range[1],
                                  num = (spectral_range[1] - spectral_range[0])*10+1,
                                  endpoint = True)
    absorbance = np.zeros( (spectrum_points.shape[0], 1) )
    for i, abs in enumerate(absorbance):
        absorbance[i] += 130629740 * (f / sigma_broadening) * math.exp( -( ( (spectrum_points[i] - lambda_max) / (1e7*sigma_broadening)) ** 2 ) )
    return spectrum_points, absorbance

def gaussian(lambda_max, f, spectral_range):
    spectrum_points = np.linspace(spectral_range[0], spectral_range[1],
                                  num = (spectral_range[1] - spectral_range[0])*10+1,
                                  endpoint = True)
    absorbance = np.zeros( (spectrum_points.shape[0], 1) )
    for i, abs in enumerate(absorbance):
#        print(f'Spectrum_point={spectrum_points[i]}, lambda_max={lambda_max}')
        absorbance[i] += 40489.994 * f * np.exp(-((1/spectrum_points[i]-1/lambda_max)/0.0003226222738)**2)
        
    return spectrum_points, absorbance
    
def eV_to_nm(energy):
    if energy != 0:
        return 1239.8/energy
    else:
        return 0.0

# Extraction functions, output format-specific. All return np.array with type np.float32, 
# column 0 - transitions, column 1 - oscillator strengths
# Keep that format to stay compatible with the main code
def extract_uv_orca(uv_file):
    with open(uv_file, 'r') as f:
#        spec_data = re.findall(r'(?m)(?s)ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS.*?-+.*?^-+\n(.*?)^\n', f.read())
        spec_data = re.findall(r'(?m)(?s)ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS.*?-+.*?^-+\n(.*?)^\n', f.read())
        spec_data = re.findall(r'(?m)(?s)^\s*\S+\s+->\s+\S+\s+([+-]?\d*\.\d+)\s+[+-]?\d*\.\d+\s+[+-]?\d*\.\d+\s+([+-]?\d*\.\d+)', spec_data[0])
    return np.array(spec_data, dtype=np.float32)

def extract_uv_openqp_mrsf(uv_file):
    with open(uv_file, 'r') as f:
        spec_data = np.array([line.split() for line in re.findall(r'(?m)^ +\d+(?: +-?\d+\.\d+){9}$', f.read())])
        spec_data = spec_data[:,[3,9]].astype(np.float32)
    return spec_data
    
def extract_uv_gamess_mrsf(uv_file):
    with open(uv_file, 'r') as f:
        spec_data = np.array([line.split() for line in re.findall(r'(?m)^ +\d+ +[A-Z](?: +-?\d+\.\d+){7}$', f.read())])
        spec_data = spec_data[:,[3,8]].astype(np.float32)
        spec_data[:,0] = spec_data[:,0] - spec_data[0,0]
    return spec_data
    
def extract_uv_openqp(uv_file):
    with open(uv_file, 'r') as f:
        spec_data = np.array([line.split() for line in re.findall(r'(?m)^ +0 -> \d+(?: +-?\d\.\d+)+\s*$', f.read())])
        spec_data = spec_data[:,[3,7]].astype(np.float32)
    return spec_data
    
def energy_to_boltzmann(energies, T=298.15):
    R = 8.314
    bws = 2625.499 * 1000 * (energies - np.min(energies)) 
    bws = np.exp(-bws/(R*T))
    bws = bws / np.sum(bws)
    bws = bws.reshape((-1,1))
    return bws

def main():
    parser = argparse.ArgumentParser(
                        prog='UV_plotter',
                        description='Plots UV from different ESS outputs (ORCA, OpenQP)',)

    parser.add_argument('filenames', nargs='+', help='Output file(s) to process')
    parser.add_argument('--ess', default='ORCA', help='Electronic structure system that produced the output file. Available: \
                                                       orca (transition velocity dipole moments), openqp (regular TDDFT), \
                                                       openqp-mrsf, gamess-mrsf. Default: orca')
    parser.add_argument('--spectral-range', type=str, default='100-800', help='Spectral range, default: 100-800. Added automatically to the output suffix')
    parser.add_argument('--normalize', type=float, default=0.0, help='Normalize to #, default: OFF')
    parser.add_argument('-s', '--suffix', type=str, default='_spectrum', help='Output file suffix, default: _spectrum')
    parser.add_argument('--dry-run', action='store_true', help='Only prints the list of excitations')
    args = parser.parse_args()
    
    allowed_ess = ['gamess-mrsf', 'orca', 'openqp', 'openqp-mrsf']
    if args.ess.lower() not in allowed_ess:
        sys.exit(f'Unknown ESS selected, choose from {" ".join([allowed_ess])}')
    
    for file in args.filenames:
        match args.ess.lower():
            case 'orca': excitations = extract_uv_orca(file)
            case 'openqp-mrsf': excitations = extract_uv_openqp_mrsf(file)
            case 'openqp': excitations = extract_uv_openqp(file)
            case 'gamess-mrsf': excitations = extract_uv_gamess_mrsf(file)

        if args.dry_run == True:
            for i, exc in enumerate(excitations, start=1):
                ev, osc = exc
                print(f'{i: >2}\t{ev: >10.3f}\t{osc: >10.3f}')
            continue        

        spec_range = [int(a) for a in args.spectral_range.split('-')]
        final_spec = np.zeros(((spec_range[1] - spec_range[0])*10+1,1))
        vec_eV_to_nm = np.vectorize(eV_to_nm)
        excitations[:,0] = vec_eV_to_nm(excitations[:,0])
        
        print('Extracted list of transitions:\n', excitations)

        
        for i, excitation in enumerate(excitations):
            if excitation[0] == 0.0 and 'mrsf' not in args.ess.lower():
                print(f'Warning: transition number {i} has E = 0 eV, but your method is not MRSF. Skipping it')
                continue
            elif excitation[0] == 0.0 and 'mrsf' in args.ess.lower():
                print(f'Warning: transition number {i} has E = 0 eV, but it is normal for MRSF. Skipping it')
                continue
            final_spec += gaussian(*excitation, spec_range)[1]

        # Normalization
        if args.normalize:
            final_spec = (final_spec - final_spec.min()) / final_spec.ptp()
            final_spec = final_spec * args.normalize
            
        spectrum_points = np.linspace(spec_range[0], spec_range[1],
                                      num = (spec_range[1] - spec_range[0])*10+1,
                                      endpoint = True)[:, None]
                                      
        spectrum_name = os.path.basename(file).split('.')[0] + f'{args.suffix}' + f'_{args.spectral_range}_nm' + '.csv'
        print(spectrum_name, 'saved')
        final_spec = np.hstack((spectrum_points, final_spec))
        np.savetxt(spectrum_name, final_spec, delimiter=',')

if __name__ == '__main__':
    main()