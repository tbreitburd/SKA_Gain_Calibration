"""!@file question_2.py
@brief File containing code to answer question 2 of the test.

@details

@author Created by T.Breitburd on 19/03/2024
"""

import numpy as np
from harp_beam import compute_EEPs
import scipy.io
from plot_funcs import plot_EEP_AEP
import os

# Load the EEPs
data_dir = os.getcwd()
filename_eep = os.path.join(
    data_dir, "harp_beam", "data_EEPs_SKALA41_random_100MHz.mat"
)
SKA_data = scipy.io.loadmat(filename_eep)
pos_ant = np.array(SKA_data["pos_ant"])
x_pos = pos_ant[:, 0]
y_pos = pos_ant[:, 1]

# Define the angles over which to get EEPs
# Here we want to get EEps for theta = [ -pi/2, pi/2] and phi = 0
# Get EEPs for theta from -pi/2 to pi/2
theta_half = np.linspace(0, np.pi / 2, 500)[:, None]
theta = np.concatenate(
    (-theta_half[::-1], theta_half), axis=0
)  # to get theta from -pi/2 to pi/2
phi = np.zeros_like(theta)

# Compute the EEPs
v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX = compute_EEPs(SKA_data, theta, phi)

# ---------------------
# Plot the EEPs and AEP
# ---------------------

# Because compute EEPs returns the spherical components of the EEPs,
# the components need to be combined to get the EEPs

# Combine the EEPs
v_polX = np.abs(np.sqrt((v_theta_polX) ** 2 + (v_phi_polX) ** 2))
v_polY = np.abs(np.sqrt((v_theta_polY) ** 2 + (v_phi_polY) ** 2))

# Convert to dBV
v_polX = 10 * np.log10(v_polX)
v_polY = 10 * np.log10(v_polY)

# Plot the EEPs and AEP
plot_EEP_AEP(theta, v_polX, "polX")
plot_EEP_AEP(theta, v_polY, "polY")
