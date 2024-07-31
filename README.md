This code package performs piecewise PV inversion as described in detail in 
[Teubler and Riemer 2016](https://doi.org/10.1175/JAS-D-15-0162.1) in a latitudinal belt on the northern
hemisphere. The PV inversion code itself is based on 
[Davis 1992](https://doi.org/10.1175/1520-0469(1992)049<1397:PPVI>2.0.CO;2).
PV anomalies are defines as upper-level anomalies and low-level anomalies defined as deviations from
a 30-day background field.

The development of the PV inversion was part of the PhD thesis of [Franziska Teubler](https://dynmet.ipa.uni-mainz.de/dr-franziska-teubler/) under the supervision
of [Michael Riemer](https://dynmet.ipa.uni-mainz.de/pd-dr-michael-riemer/).

[MIT License](LICENSE).

## INPUT
* file with instantaneous fields
* file with background information (e.g. temporal mean) to calculate anomalies
  
The input files can be grib or netcdf and have to contain u,v,T and Geopotential on
pressure levels. The programm will automatically interpolate on 17 pressure levels needed
for PVInversion. The more pressure levels are available the better.

## OUTPUT
The output file will contain u and v for the balanced, the background, the upper-level,
and low-level flow component (Geopotential can be added easily, if required).

# Installation
To run the code the following python packages need to be installed:
  
  $ conda -c conda-forge numpy numba netcdf4 h5netcdf xarray eccodes scipy petsc petsc4py

# RUN
The main file is run_PVI.py. Execute the PVinversion by simply running

  $ python run_PVI.py

## General comments and recommentations 
* I recommend to calculate PV on your own (as it is done here) and do not use PV available in datasets
* Do not try to use a higher resosution than 1°x 1°; for higher resolution there will be no convergence 
    (nonlinear balance can not be reached)
* I recommend a maximal meridional extend from 20°N to 85°N (better 80°N). Further north and south balance condition will not be reached.
