"""!@file question_5.py

@brief File containing code to plot the station beam of the SKA1-Low station
using the EEPs and AEPs model matrices, and the exact gain solution.

@details This script computes the EEPs of the SKA1-Low station, runs the SteEFCal algorithm
for the AEP and EEPs model matrices, and plots the station beam of the SKA1-Low station using
the EEPs and AEPs model matrices, and the exact gain solution.

@author Created by T.Breitburd on 19/03/2024
"""

import numpy as np
from harp_beam import compute_EEPs, StEFCal, StEFCal_2, compute_array_pattern
from plot_funcs import plot_station_beam
import scipy.io
import os


# -------------
# Load the EEPs
# -------------
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

# ----------------
# Compute the EEPs
# ----------------

# Because compute EEPs returns the spherical components of the EEPs,
# the components need to be combined to get the EEPs

# Combine the EEPs
v_polX = np.sqrt(np.abs(v_theta_polX) ** 2 + np.abs(v_phi_polX) ** 2)
v_polY = np.sqrt(np.abs(v_theta_polY) ** 2 + np.abs(v_phi_polY) ** 2)


# ----------------------------------------------------------
# Get the estimates of the EEPs and AEP model gain solutions
# ----------------------------------------------------------

# Load the measured and model visibilities, and the exact gain solution
data_dir = os.getcwd()
filename_vismat = os.path.join(data_dir, "harp_beam", "data_20feb2024_2330_100MHz.mat")
mat = scipy.io.loadmat(filename_vismat)
R = np.array(mat["R"])  # covariance matrix
M_AEP = np.array(mat["M_AEP"])  # model matrix using AEP
M_EEPs = np.array(mat["M_EEPs"])  # model matrix using all EEPs
g_sol = np.array(mat["g_sol"]).reshape(256)  # exact gain solution
G_true = np.diag(g_sol)

# Set the convergence tolerence and maximum number of iterations, and number of antennas
tau = 1e-5
i_max = 100
P = 256

# Run the SteEFCal algorithm for the AEP and EEPs model matrices
G_AEP, diff_AEP, abs_error_AEP, amp_diff_AEP, phase_diff_AEP = StEFCal(
    M_AEP, R, tau, i_max, P, g_sol
)

G_EEPs, diff_EEPs, abs_error_EEPs, amp_diff_EEPs, phase_diff_EEPs = StEFCal(
    M_EEPs, R, tau, i_max, P, g_sol
)

# ----------------------------------------------
# Get the array pattern to plot the station beam
# ----------------------------------------------

P_true_X = compute_array_pattern(
    G_true, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_EEPs_X = compute_array_pattern(
    G_EEPs, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_AEP_X = compute_array_pattern(
    G_AEP, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_true_Y = compute_array_pattern(
    G_true, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_EEPs_Y = compute_array_pattern(
    G_EEPs, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_AEP_Y = compute_array_pattern(
    G_AEP, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)


# Get the patterns in dBV
P_true_X_dB = 20 * np.log10(np.abs(P_true_X))
P_EEPs_X_dB = 20 * np.log10(np.abs(P_EEPs_X))
P_AEP_X_dB = 20 * np.log10(np.abs(P_AEP_X))
P_true_Y_dB = 20 * np.log10(np.abs(P_true_Y))
P_EEPs_Y_dB = 20 * np.log10(np.abs(P_EEPs_Y))
P_AEP_Y_dB = 20 * np.log10(np.abs(P_AEP_Y))

# Plot the station beam

plot_station_beam(
    theta,
    P_true_X_dB,
    P_true_Y_dB,
    P_EEPs_X_dB,
    P_EEPs_Y_dB,
    P_AEP_X_dB,
    P_AEP_Y_dB,
    100,
    "StEFCal",
)


# ---------------------------------------
# Repeat for the optimised algorithm
# ---------------------------------------

# Run the SteEFCal algorithm for the AEP and EEPs model matrices

G_AEP_2, diff_AEP_2, abs_error_AEP_2, amp_diff_AEP_2, phase_diff_AEP_2 = StEFCal_2(
    M_AEP, R, tau, i_max, P, g_sol
)

G_EEPs_2, diff_EEPs_2, abs_error_EEPs_2, amp_diff_EEPs_2, phase_diff_EEPs_2 = StEFCal_2(
    M_EEPs, R, tau, i_max, P, g_sol
)

# ----------------------------------------------
# Get the array pattern to plot the station beam
# ----------------------------------------------
P_true_X = compute_array_pattern(
    G_true, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_EEPs_X = compute_array_pattern(
    G_EEPs_2, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_AEP_X = compute_array_pattern(
    G_AEP_2, 1, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_true_Y = compute_array_pattern(
    G_true, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_EEPs_Y = compute_array_pattern(
    G_EEPs_2, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)
P_AEP_Y = compute_array_pattern(
    G_AEP_2, 1, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, 0, 0, 0, 100
)

# Get the patterns in dBV
P_true_X_dB = 20 * np.log10(np.abs(P_true_X))
P_EEPs_X_dB = 20 * np.log10(np.abs(P_EEPs_X))
P_AEP_X_dB = 20 * np.log10(np.abs(P_AEP_X))
P_true_Y_dB = 20 * np.log10(np.abs(P_true_Y))
P_EEPs_Y_dB = 20 * np.log10(np.abs(P_EEPs_Y))
P_AEP_Y_dB = 20 * np.log10(np.abs(P_AEP_Y))

# Plot the station beam
plot_station_beam(
    theta,
    P_true_X_dB,
    P_true_Y_dB,
    P_EEPs_X_dB,
    P_EEPs_Y_dB,
    P_AEP_X_dB,
    P_AEP_Y_dB,
    100,
    "StEFCal_2",
)
