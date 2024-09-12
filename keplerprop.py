#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Load the data and initialize the interpolator
data = np.load('kep_interp.npy')
interpolator = RegularGridInterpolator((data[0], data[1]), data[2:], method='linear')

def compute_trajectory(dtgrid, elements, mu):
    """
    Compute the trajectory of an orbiting body.
    
    Parameters:
    dtgrid (numpy array): Array of time intervals since M=M0, typically in days. (Units should match mu.)
    elements (tuple): Orbital elements:
        - a (float): Semi-major axis, typically in AU (units should match mu).
        - e (float): Eccentricity, should be between [0, 1].
        - i (float): Inclination in degrees.
        - omega (float): Argument of periapsis in degrees.
        - Omega (float): Longitude of the ascending node in degrees.
        - M0 (float): Mean anomaly at epoch in degrees.
    mu (float): Standard gravitational parameter.
    
    Returns:
    tuple: Two numpy arrays:
        - r (numpy array): Array of position vectors.
        - v (numpy array): Array of velocity vectors.
    """
    
    # Unpack the orbital elements
    a, e, i, omega, Omega, M0 = elements
    
    # Convert angles from degrees to radians
    i = np.radians(i)
    omega = np.radians(omega)
    Omega = np.radians(Omega)
    M0 = np.radians(M0)
    
    # Compute mean anomaly over time grid
    Mt = np.mod(M0 + dtgrid * np.sqrt(mu / a**3), 2 * np.pi)
    coords = np.column_stack((Mt, e * np.ones(len(Mt))))
    
    # Interpolate eccentric anomaly (E)
    Es = interpolator(coords)
    
    # Compute true anomaly (ν) and radial distance (r_c)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(Es / 2), np.sqrt(1 - e) * np.cos(Es / 2))
    rc = a * (1 - e * np.cos(Es))
    
    # Compute position in orbital plane
    ox = rc * np.cos(nu)
    oy = rc * np.sin(nu)
    oz = np.zeros(len(nu))
    
    # Compute velocity in orbital plane
    f = np.sqrt(mu * a) / rc
    dot_ox = -f * np.sin(Es)
    dot_oy = f * np.sqrt(1 - e**2) * np.cos(Es)
    dot_oz = np.zeros(len(Es))
    
    # Rotate the position and velocity to the inertial frame
    rx = ox * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.cos(i) * np.sin(Omega)) \
         - oy * (np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.cos(i) * np.sin(Omega))

    ry = ox * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(i) * np.cos(Omega)) \
         + oy * (np.cos(omega) * np.cos(i) * np.cos(Omega) - np.sin(omega) * np.sin(Omega))

    rz = ox * np.sin(omega) * np.sin(i) + oy * np.cos(omega) * np.sin(i)

    # Rotate velocity
    r_dot_x = dot_ox * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.cos(i) * np.sin(Omega)) \
              - dot_oy * (np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.cos(i) * np.sin(Omega))

    r_dot_y = dot_ox * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(i) * np.cos(Omega)) \
              + dot_oy * (np.cos(omega) * np.cos(i) * np.cos(Omega) - np.sin(omega) * np.sin(Omega))

    r_dot_z = dot_ox * np.sin(omega) * np.sin(i) + dot_oy * np.cos(omega) * np.sin(i)
    
    # Combine position and velocity into arrays
    r = np.array([rx, ry, rz]).T
    v = np.array([r_dot_x, r_dot_y, r_dot_z]).T
    
    return r, v

