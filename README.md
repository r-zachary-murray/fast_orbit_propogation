# Fast Orbit Propagation

A lightweight Python module for rapid orbit propagation, achieving sub-millisecond performance.

The core functionality is provided in the `keplerprop.py` module, which requires the `kep_interp.npy` file to be in the same directory. This file contains pre-computed solutions to Kepler's equation for fast interpolation. The orbit propagation code is fully vectorized over time, allowing it to predict the orbit of a typical main-belt asteroid (e.g., Ceres) daily over an entire orbital period (~2000 observations) in just 0.2 milliseconds.

A demo and the code required to generate the `kep_interp.npy` file can be found in the provided notebooks.

