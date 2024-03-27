"""!@file test_harp_beam.py

@brief File containing tests for the harp_beam module.

@details This file contains tests for the functions in the harp_beam module.

@author Created by T.Breitburd on 21/03/2024"""


import numpy as np
import os
import scipy.io
from harp_beam import harp_beam

# Load the station data
data_dir = os.getcwd()
filename_eep = os.path.join(
    data_dir, "harp_beam", "data_EEPs_SKALA41_random_100MHz.mat"
)
SKA_data = scipy.io.loadmat(filename_eep)
pos_ant = np.array(SKA_data["pos_ant"])
x_pos = pos_ant[:, 0]
y_pos = pos_ant[:, 1]

# Define the angles over which to get EEPs
theta_half = np.linspace(0, np.pi / 2, 500)[:, None]
theta = np.concatenate(
    (-theta_half[::-1], theta_half), axis=0
)  # to get theta from -pi/2 to pi/2
phi = np.zeros_like(theta)


# Load the measured and model visibilities, and the exact gain solution
data_dir = os.getcwd()
filename_vismat = os.path.join(data_dir, "harp_beam", "data_20feb2024_2330_100MHz.mat")
mat = scipy.io.loadmat(filename_vismat)
R = np.array(mat["R"])  # covariance matrix
M_AEP = np.array(mat["M_AEP"])  # model matrix using AEP
M_EEPs = np.array(mat["M_EEPs"])  # model matrix using all EEPs
g_sol = np.array(mat["g_sol"]).reshape(256)  # exact gain solution
G_true = np.diag(g_sol)

V_theta_polY, V_phi_polY, V_theta_polX, V_phi_polX = harp_beam.compute_EEPs(
    SKA_data, theta, phi
)


def test_compute_EEPs():
    """!@brief Test the compute_EEPs function"""

    assert V_theta_polY.shape == (
        len(theta),
        len(x_pos),
    )  # Check the shape of the output
    assert np.all(V_phi_polX) >= 0  # All values should be positive


# Set the convergence tolerance, maximum number of iterations, and number of antennas
tau = 1e-5
i_max = 100
P = 256


# Run the SteEFCal algorithm for the AEP and EEPs model matrices
G_AEP, diff_AEP, abs_error_AEP, amp_diff_AEP, phase_diff_AEP = harp_beam.StEFCal(
    M_AEP, R, tau, i_max, P, g_sol
)

G_EEPs, diff_EEPs, abs_error_EEPs, amp_diff_EEPs, phase_diff_EEPs = harp_beam.StEFCal(
    M_EEPs, R, tau, i_max, P, g_sol
)


def test_StEFCal():
    """!@brief Test the StEFCal algorithm"""

    # Check the output of the StEFCal algorithm
    assert G_AEP.shape == (256, 256)  # Check the shape of the output
    assert G_EEPs.shape == (256, 256)  # Check the shape of the output

    # Check the gain solutions are diagonal
    assert np.allclose(G_AEP, np.diag(np.diag(G_AEP)))
    assert np.allclose(G_EEPs, np.diag(np.diag(G_EEPs)))

    # Check the absolute errors are positive
    assert np.all(np.array(abs_error_AEP) >= 0)
    assert np.all(np.array(abs_error_EEPs) >= 0)
    assert np.all(np.array(amp_diff_AEP) >= 0)
    assert np.all(np.array(amp_diff_EEPs) >= 0)

    # Check that the difference is indeed below the tolerance
    assert diff_AEP[-1] < tau
    assert diff_EEPs[-1] < tau


def test_compute_array_pattern():
    """!@brief Test the compute_array_pattern function"""

    P_true_X = harp_beam.compute_array_pattern(
        G_true, 1, V_theta_polX, V_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
    )
    P_EEPs_X = harp_beam.compute_array_pattern(
        G_EEPs, 1, V_theta_polX, V_phi_polX, x_pos, y_pos, theta, 0, 0, 0, 100
    )

    assert P_true_X.shape == (len(theta),)  # Check the shape of the output
    assert np.all(P_EEPs_X >= 0)  # Check that all values are positive
