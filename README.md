# tools
**February 5<sup>th</sup>, 2026**

Package containing miscellaneous useful scripts:
- QuickPlotAbs: quickly plot absorption spectra from ASCII files
- plotter_TDDFT: plot TDDFT data
- molgeom
  -  molgeom: retrieve geometrical properties of molecules, e.g., obtain, distances, angles, and dihedral angles
  -  reorient: perform rotational operations on geometries of molecules
- SpectraFitter: fit Gaussian shapes to experimental spectra to obtain smooth curves useful for further processing
- PSS_Calculator: retrieve absorption spectrum of metastable isomer from stable and PSS spectrum together with ratio at PSS

# Installation
## Anaconda Powershell Prompt
```
conda create -n toolbox
conda activate toolbox
conda install git pip
git clone https://github.com/CrespiLab/toolbox
cd toolbox
pip install -e .
```

# User Instructions
Make sure to activate the environment.
```
conda activate toolbox
```

## QuickPlotAbs
```
quickplotabs 'Path\to\folder\containing\ASCII\files\' (--wavelength 'wavelength') (--legend)
```

## plotter_TDDFT
TBA

## SpectraFitter
To start the program, use the command:
```
spectrafitter
```

## PSS_Calculator
To start the program, use the command:
```
psscalc
```
