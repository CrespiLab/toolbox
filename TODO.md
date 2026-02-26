# To-Do List

## To Do
- [ ] Add Python script for fitting of exponential decay

## In Progress :)
- [ ] Update plotter_TDDFT:
  - [ ] integrate plotter_tddft.py functions into main UV_spectrum.py: options to obtain vertical transitions as well as plots

- [ ] Improve SpectraFitter:
  - [ ] Run Manual Mode on a separate thread (like Auto Mode)
  - [ ] Update plot after every newly added peak (use a separate thread)
  - [ ] Plot results in GUI
  - [ ] Add option for baseline correction
  - [ ] Add option to exclude certain region of spectrum (consider it baseline)
  - [ ] Add option to use only Gaussian bands (no Pekarians)

- [ ] Improve PSS_Calculator
  - [ ] Add feature to calculate ratio of Stable/Metastable isomers at any PSS

- [ ] Improve Epsilonator
	- [ ] Convert to command-line script for more convenient input of data
	- [ ] Convert to GUI

- [ ] Improve Averaging
	- [ ] Add Load feature

## Completed ✓


- [x] Release Version 1.0.0
- [x] Averaging: add Save feature
- [x] Add tool for averaging of values with uncertainties: various options for propagating the error
- [x] Create general module for handling of data
- [x] SpectraFitter: Improve Auto Mode
- [x] plotter_TDDFT: add options for GAMESS and OpenQP, among other things
