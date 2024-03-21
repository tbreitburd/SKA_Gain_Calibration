"""!@file question_4.py
@brief File containing code to answer question 4 of the SKA-low mini project.

@details

@author Created by T.Breitburd on 19/03/2024
"""
import numpy as np
from harp_beam import StEFCal
from plot_funcs import plot_convergence
import scipy.io
import os

# Load the measured and model visibilities, and the exact gain solution
data_dir = os.getcwd()
filename_vismat = os.path.join(data_dir, "harp_beam", "data_20feb2024_2330_100MHz.mat")
mat = scipy.io.loadmat(filename_vismat)
R = np.array(mat["R"])  # covariance matrix
M_AEP = np.array(mat["M_AEP"])  # model matrix using AEP
M_EEPs = np.array(mat["M_EEPs"])  # model matrix using all EEPs
g_sol = np.array(mat["g_sol"]).reshape(256)  # exact gain solution

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


# Plot the convergence of the algorithm
# (abs. error of gain sols, of their amplitude and their phase difference)

plot_convergence(abs_error_AEP, i_max, "AEP", "Absolute")
plot_convergence(amp_diff_AEP, i_max, "AEP", "Amplitude")
plot_convergence(phase_diff_AEP, i_max, "AEP", "Phase")

plot_convergence(abs_error_EEPs, i_max, "EEPs", "Absolute")
plot_convergence(amp_diff_EEPs, i_max, "EEPs", "Amplitude")
plot_convergence(phase_diff_EEPs, i_max, "EEPs", "Phase")


# Get the minimum error achieved in the AEP and EEPs model matrices
min_abs_error_AEP = min(abs_error_AEP)
min_abs_error_EEPs = min(abs_error_EEPs)

print("Minimum absolute error achieved in the AEP model matrix: ", min_abs_error_AEP)
print("Reached at iteration number: ", np.argmin(abs_error_AEP))
print("Minimum absolute error achieved in the EEPs model matrix: ", min_abs_error_EEPs)
print("Reached at iteration number: ", np.argmin(abs_error_EEPs))
