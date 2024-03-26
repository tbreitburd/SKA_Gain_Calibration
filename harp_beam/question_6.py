"""!@file question_6.py

@brief File containing code to plot the 2D station beam of the SKA1-Low station
using the EEPs and AEPs model matrices, and the exact gain solution, pointing in the direction
theta0 = 40 degrees and phi0 = 80 degrees.

@author Created by T.Breitburd on 19/03/2024
"""

import numpy as np
from harp_beam import compute_EEPs, StEFCal, compute_array_pattern
from plot_funcs import plot_station_beam_2D
import scipy.io
import os
import warnings

# Ignore the pcolormesh warning
warnings.filterwarnings("ignore", category=UserWarning)


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
theta = np.linspace(0, np.pi / 2, 200)[:, None]
# theta = np.concatenate((-theta[::-1], theta), axis=0)  # to get theta from -pi/2 to pi/2
phi = np.linspace(0, 2 * np.pi, 200)[:, None]

theta_EEP = np.concatenate([theta] * 200)
phi_EEP = np.repeat(phi, 200)
phi_EEP = phi_EEP.reshape(40000, 1)

# ----------------
# Compute the EEPs
# ----------------

# Compute the EEPs
v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX = compute_EEPs(
    SKA_data, theta_EEP, phi_EEP
)
print("EEPs computed")
# Reshape the EEPs and angles
v_theta_polY = v_theta_polY.reshape(200, 200, 256)
v_phi_polY = v_phi_polY.reshape(200, 200, 256)
v_theta_polX = v_theta_polX.reshape(200, 200, 256)
v_phi_polX = v_phi_polX.reshape(200, 200, 256)
theta = theta_EEP.reshape(200, 200).copy()
phi = phi_EEP.reshape(200, 200).copy()

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

# Set the convergence tolerance and maximum number of iterations, and number of antennas
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
print("StEFCal done")
# ----------------------------------------------
# Get the array pattern to plot the station beam
# ----------------------------------------------
# Set the steering position
theta0 = 40 * np.pi / 180
phi0 = 80 * np.pi / 180

P_true_X = compute_array_pattern(
    G_true, 2, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, phi, theta0, phi0, 100
)

P_EEPs_X = compute_array_pattern(
    G_EEPs, 2, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, phi, theta0, phi0, 100
)

P_AEP_X = compute_array_pattern(
    G_AEP, 2, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, phi, theta0, phi0, 100
)

P_true_Y = compute_array_pattern(
    G_true, 2, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, phi, theta0, phi0, 100
)

P_EEPs_Y = compute_array_pattern(
    G_EEPs, 2, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, phi, theta0, phi0, 100
)

P_AEP_Y = compute_array_pattern(
    G_AEP, 2, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, phi, theta0, phi0, 100
)
print("Array patterns computed")


# Combine the feeds
P_true = np.sqrt(np.abs(P_true_X) ** 2 + np.abs(P_true_Y) ** 2)
P_AEP = np.sqrt(np.abs(P_AEP_X) ** 2 + np.abs(P_AEP_Y) ** 2)
P_EEPs = np.sqrt(np.abs(P_EEPs_X) ** 2 + np.abs(P_EEPs_Y) ** 2)

# Get the difference between the true and the EEPs
P_diff = P_true - P_EEPs

# Convert the array patterns to dBV
P_true_dB = 20 * np.log10(np.abs(P_true))
P_AEP_dB = 20 * np.log10(np.abs(P_AEP))
P_EEPs_dB = 20 * np.log10(np.abs(P_EEPs))
P_diff_dB = 20 * np.log10(np.abs(P_diff))

# Define the sine-cosine coordinates
x = np.zeros((200, 200))
y = np.zeros((200, 200))

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)

# Plot the station beams
# and an anomaly one
plot_station_beam_2D(x, y, P_true_dB, 100, "True")
plot_station_beam_2D(x, y, P_EEPs_dB, 100, "EEPs")
plot_station_beam_2D(x, y, P_AEP_dB, 100, "AEP")
plot_station_beam_2D(x, y, P_diff_dB, 100, "True - EEPs")


# For the frequency extremes:
P_EEPs_X = compute_array_pattern(
    G_EEPs, 2, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, phi, 0, 0, 50
)
P_EEPs_Y = compute_array_pattern(
    G_EEPs, 2, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, phi, 0, 0, 50
)


P_EEPs_50 = np.sqrt(np.abs(P_EEPs_X) ** 2 + np.abs(P_EEPs_Y) ** 2)
P_EEPs_50_dB = 20 * np.log10(np.abs(P_EEPs_50))

P_EEPs_X = compute_array_pattern(
    G_EEPs, 2, v_theta_polX, v_phi_polX, x_pos, y_pos, theta, phi, 0, 0, 350
)
P_EEPs_Y = compute_array_pattern(
    G_EEPs, 2, v_theta_polY, v_phi_polY, x_pos, y_pos, theta, phi, 0, 0, 350
)

P_EEPs_350 = np.sqrt(np.abs(P_EEPs_X) ** 2 + np.abs(P_EEPs_Y) ** 2)
P_EEPs_350_dB = 20 * np.log10(np.abs(P_EEPs_350))

plot_station_beam_2D(x, y, P_EEPs_50_dB, 50, "EEPs")
plot_station_beam_2D(x, y, P_EEPs_350_dB, 350, "EEPs")
