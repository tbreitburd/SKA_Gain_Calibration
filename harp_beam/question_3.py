"""!@file question_3.py

@brief File containing code to run the SteEFCal algorithm for the AEP and EEPs model matrices
and plot the convergence of the algorithm for the 2 cases.

@author Created by T.Breitburd on 19/03/2024
"""
import numpy as np
from harp_beam import StEFCal, StEFCal_2
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

# Set the convergence tolerance, maximum number of iterations, and number of antennas
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
# just for the absolute error of gain sols

plot_convergence(diff_AEP, i_max, "AEP", "Difference")
plot_convergence(diff_EEPs, i_max, "EEPs", "Difference")
plot_convergence(np.log10(diff_EEPs), i_max, "EEP", "Log Absolute error")

# Get the number of iterations for the AEP and EEPs model matrices

print("Number of iterations for the AEP model matrix: ", len(abs_error_AEP))
print("Number of iterations for the EEPs model matrix: ", len(abs_error_EEPs))


# Run the optimised version of the SteEFCal algorithm for the AEP and EEPs model matrices
# and get the number of iterations

G_AEP_opt, diff_AEP_opt, _, __, ___ = StEFCal_2(M_AEP, R, tau, i_max, P, g_sol)
G_EEPs_opt, diff_EEPs_opt, _, __, ____ = StEFCal_2(M_EEPs, R, tau, i_max, P, g_sol)

# Plot the convergence of the algorithm
# just for the absolute error of gain sols

plot_convergence(
    diff_AEP_opt,
    i_max,
    "AEP_2",
    "Difference",
)
plot_convergence(diff_EEPs_opt, i_max, "EEPs_2", "Difference")
plot_convergence(np.log10(diff_EEPs_opt), i_max, "EEP_2", "Log Absolute error")

# Get the number of iterations for the AEP and EEPs model matrices

print("Number of iterations for the AEP model matrix: ", len(diff_AEP_opt))
print("Number of iterations for the EEPs model matrix: ", len(diff_EEPs_opt))
