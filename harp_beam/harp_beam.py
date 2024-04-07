"""!@file harp_beam.py
@brief Module containing tools for calibrating the SKA-LOW station

@details This module contains tools to calibrate the SKA-LOW station and is part
of the SKA-low mini project. It contains functions to calculate the spherical wave modes,
the EEPs, and to estimate the gains using the SteEFCal algorithm.

@author Created by Q. Gueuning (qdg20@cam.ac.uk) and O. O'Hara on 01/03/2024
Modified by T.Breitburd on 19/03/2024
"""

import numpy as np
from scipy.special import lpmv, factorial


def legendre(deg, x):
    """
    @brief Calculate the associated Legendre function for integer orders and degree at value x.

    @param deg Degree of the Legendre function, float
    @param x Position to evaluate function, float

    @return np.array Legendre function for all integer orders from 0 to deg.
    """
    return np.asarray([lpmv(i, deg, x) for i in range(deg + 1)])[:, 0, :]


def legendre3(n, u):
    """
    @brief Calculate all associated Legendre functions up to max order n at value x.

    @param deg Degree of the Legendre function, float
    @param x Position to evaluate function, float

    @return np.array Legendre functions (Pnm,Pnm/costheta,dPnmdsintheta)
    for all integer orders from 0 to deg.
    """
    pn = legendre(n, u)
    pnd = np.divide(pn, np.ones_like(n + 1) * np.sqrt(1 - u**2))

    mv = np.arange(n)

    dpns = np.zeros((n + 1, len(u[0])))
    dpns[:-1, :] = (
        np.multiply(-(mv[:, None]), np.divide(u, 1 - u**2)) * pn[mv, :]
        - pnd[mv + 1, :]
    )
    dpns[n, :] = np.multiply(-n, np.divide(u, 1 - u**2)) * pn[n, :]
    dpns *= np.sqrt(1 - u**2)
    return pn, pnd, dpns


def smodes_eval(order, alpha_tm, alpha_te, theta, phi):
    """
    @brief Calculate spherical wave modes TE and TM according to definitions in the book:
    J.E. Hansen, Spherical near-field measurements

    @param order Max order of the Legendre function, float
    @param alpha_tm coefficients for TM modes, 3d array of size
    (num_mbf, 2 * max_order + 1, max_order)
    @param alpha_te coefficients for TE modes, 3d array of size
    (num_mbf, 2 * max_order + 1, max_order)
    @param theta Zenith angle, np.array, float
    @param phi Azimuth angle, np.array, float

    @return np.array (complex double) gvv
    @return np.array (complex double) ghh
    """
    tol = 1e-5
    theta[theta < tol] = tol

    Na = len(alpha_tm[:, 1, 1])

    u = np.cos(theta.T)
    gvv = np.zeros((Na, theta.shape[0]), dtype=complex)
    ghh = np.zeros((Na, theta.shape[0]), dtype=complex)

    EE = np.exp(1j * np.arange(-order, order + 1) * phi).T
    for n in range(1, order + 1):
        mv = np.arange(-n, n + 1)
        pn, pnd, dpns = legendre3(n, u)
        pmn = np.row_stack((np.flipud(pnd[1:]), pnd))
        dpmn = np.row_stack((np.flipud(dpns[1:]), dpns))

        Nv = (
            2
            * np.pi
            * n
            * (n + 1)
            / (2 * n + 1)
            * factorial(n + np.abs(mv))
            / factorial(n - abs(mv))
        )
        Nf = np.sqrt(2 * Nv)
        ee = EE[mv + order]
        qq = -ee * dpmn
        dd = ee * pmn

        mat1 = np.multiply(np.ones((Na, 1)), 1 / Nf)
        mat2 = np.multiply(np.ones((Na, 1)), mv * 1j / Nf)
        an_te_polY = alpha_te[:, n - 1, (mv + n)]
        an_tm_polY = alpha_tm[:, n - 1, (mv + n)]

        gvv += np.matmul(an_tm_polY * mat1, qq) - np.matmul(an_te_polY * mat2, dd)
        ghh += np.matmul(an_tm_polY * mat2, dd) + np.matmul(an_te_polY * mat1, qq)

    return gvv.T, ghh.T


def wrapTo2Pi(phi):
    return phi % (2 * np.pi)


def compute_EEPs(mat, theta, phi):
    """
    @brief Compute the EEPs for a given set of angles theta and phi.

    @param mat Dictionary containing the data from the .mat file.
    @param theta Zenith angle, np.array, float
    @param phi Azimuth angle, np.array, float

    @return np.array (complex double) v_theta_polY
    @return np.array (complex double) v_phi_polY
    @return np.array (complex double) v_theta_polX
    @return np.array (complex double) v_phi_polX
    """

    # Make deep copies of theta and phi to avoid modifying the input
    theta_ = np.copy(theta)
    phi_ = np.copy(phi)
    ind = theta < 0
    theta_[ind] = -theta_[ind]
    phi_[ind] = wrapTo2Pi(phi_[ind] + np.pi)

    freq = 100
    c0 = 299792458  # speed of light
    k0 = 2 * np.pi * freq / c0 * 10**6  # wavenumber

    max_order = int(mat["max_order"])
    num_mbf = int(mat["num_mbf"])
    coeffs_polX = np.array(mat["coeffs_polX"])
    coeffs_polY = np.array(mat["coeffs_polY"])
    alpha_te = np.array(mat["alpha_te"])
    alpha_tm = np.array(mat["alpha_tm"])
    pos_ant = np.array(mat["pos_ant"])
    x_pos = pos_ant[:, 0]
    y_pos = pos_ant[:, 1]

    # reshaping
    alpha_te = np.ndarray.transpose(
        np.reshape(alpha_te, (num_mbf, 2 * max_order + 1, max_order), order="F"),
        (0, 2, 1),
    )
    alpha_tm = np.ndarray.transpose(
        np.reshape(alpha_tm, (num_mbf, 2 * max_order + 1, max_order), order="F"),
        (0, 2, 1),
    )

    num_dir = len(theta)
    num_ant = len(pos_ant)
    num_beam = len(coeffs_polY[0])
    num_mbf = len(alpha_tm)

    ux = np.sin(theta_) * np.cos(phi_)
    uy = np.sin(theta_) * np.sin(phi_)

    v_mbf_theta, v_mbf_phi = smodes_eval(max_order, alpha_tm, alpha_te, theta_, phi_)

    # Beam assembling
    v_theta_polY = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_phi_polY = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_theta_polX = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_phi_polX = np.zeros((num_dir, num_beam), dtype=np.complex128)
    phase_factor = np.exp(1j * k0 * (ux * x_pos + uy * y_pos))
    for i in range(num_mbf):
        p_thetai = v_mbf_theta[:, i]
        p_phii = v_mbf_phi[:, i]

        c_polY = np.matmul(
            phase_factor, coeffs_polY[np.arange(num_ant) * num_mbf + i, :]
        )
        c_polX = np.matmul(
            phase_factor, coeffs_polX[np.arange(num_ant) * num_mbf + i, :]
        )

        v_theta_polY += p_thetai[:, None] * c_polY
        v_phi_polY += p_phii[:, None] * c_polY
        v_theta_polX += p_thetai[:, None] * c_polX
        v_phi_polX += p_phii[:, None] * c_polX

    v_theta_polY *= np.conj(phase_factor)
    v_phi_polY *= np.conj(phase_factor)
    v_theta_polX *= np.conj(phase_factor)
    v_phi_polX *= np.conj(phase_factor)

    return v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX


def StEFCal(M, R, tau, i_max, P, g_sol):
    """
    @brief Estimate the gains using the SteEFCal algorithm.

    @param M Model covariance matric of observed scene, diagonal set to 0, np.array, complex
    @param R array of covariance matrix, diagonal set to 0, np.array, complex
    @param tau Tolerance, float
    @param i_max Maximum number of iterations, int
    @param P Number of antennas, int
    @param g_sol Exact gain solution, np.array, complex

    @return np.array (complex double) G_new
    @return np.array (float) diff
    @return np.array (float) abs_error
    @return np.array (float) amp_diff
    @return np.array (float) phase_diff
    """

    g_old = np.ones(P, dtype=complex)
    g_new = np.ones(P, dtype=complex)
    diff = []
    # g_new = np.array(np.diag(G_new), dtype = complex)
    # g_old = np.array(np.diag(G_old), dtype = complex)

    abs_error = []
    phase_diff = []
    amp_diff = []

    for i in range(0, i_max):
        for p in range(0, P):
            G_old = np.diag(g_old)
            z = np.dot(G_old, M[:, p])
            g_p = np.dot(np.conjugate(R[:, p]), z) / (np.dot(np.conjugate(z), z))
            g_new[p] = g_p.flatten()[0]

        # Check G_new is diagonal
        G_new = np.diag(g_new)
        if not np.allclose(G_new, np.diag(np.diagonal(G_new))):
            raise ValueError("G_new is not diagonal")

        norm_diff = np.linalg.norm(np.abs(g_new) - np.abs(g_old))
        norm_g = np.linalg.norm(np.abs(g_new))
        diff.append(norm_diff / norm_g)
        # Check if the difference is smaller than tau
        if i % 2 == 0:
            # Get the absolute error between the estimated gains and the true gains
            if norm_diff / norm_g <= tau:
                abs_error.append(np.linalg.norm(np.abs(g_new - g_sol)))
                amp_diff.append(np.linalg.norm(np.abs(np.abs(g_new) - np.abs(g_sol))))
                phase_diff.append(
                    np.linalg.norm(np.abs(np.angle(g_new) - np.angle(g_sol)))
                )
                G_new = np.diag(g_new)
                return G_new, diff, abs_error, amp_diff, phase_diff
            else:
                g_new = (g_new + g_old) / 2
        g_old = g_new.copy()

        abs_error.append(np.linalg.norm(np.abs(g_new - g_sol)))
        amp_diff.append(np.linalg.norm(np.abs(np.abs(g_new) - np.abs(g_sol))))
        phase_diff.append(np.linalg.norm(np.abs(np.angle(g_new) - np.angle(g_sol))))

    G_new = np.diag(g_new)
    print("Did not converge before max iteration")
    return G_new, diff, abs_error, amp_diff, phase_diff


def StEFCal_2(M, R, tau, i_max, P, g_sol):
    """
    @brief Estimate the gains using the SteEFCal algorithm.

    @param M Model covariance matric of observed scene, diagonal set to 0, np.array, complex
    @param R array of covariance matrix, diagonal set to 0, np.array, complex
    @param tau Tolerance, float
    @param i_max Maximum number of iterations, int
    @param P Number of antennas, int
    @param g_sol Exact gain solution, np.array, complex

    @return np.array (complex double) G_new
    @return np.array (float) diff
    @return np.array (float) abs_error
    @return np.array (float) amp_diff
    @return np.array (float) phase_diff
    """

    g_old = np.ones(P, dtype=complex)
    g_new = np.ones(P, dtype=complex)
    diff = []
    # g_new = np.array(np.diag(G_new), dtype = complex)
    # g_old = np.array(np.diag(G_old), dtype = complex)

    abs_error = []
    phase_diff = []
    amp_diff = []

    for i in range(0, i_max):
        for p in range(0, P):
            g_old = g_new.copy()
            G_old = np.diag(g_old)
            z = np.dot(G_old, M[:, p])
            g_p = np.dot(np.conjugate(R[:, p]), z) / (np.dot(np.conjugate(z), z))
            g_new[p] = g_p.flatten()[0]

        # Check G_new is diagonal
        # if not np.allclose(G_new, np.diag(np.diagonal(G_new))):
        #    raise ValueError('G_new is not diagonal')

        # Check if the difference is smaller than tau
        norm_diff = np.linalg.norm(np.abs(g_new) - np.abs(g_old))
        norm_g = np.linalg.norm(np.abs(g_new))
        diff.append(norm_diff / norm_g)
        # Get the absolute error between the estimated gains and the true gains
        if norm_diff / norm_g <= tau:
            abs_error.append(np.linalg.norm(np.abs(g_new - g_sol)))
            amp_diff.append(np.linalg.norm(np.abs(np.abs(g_new) - np.abs(g_sol))))
            phase_diff.append(np.linalg.norm(np.abs(np.angle(g_new) - np.angle(g_sol))))
            G_new = np.diag(g_new)
            return G_new, diff, abs_error, amp_diff, phase_diff

        g_old = g_new.copy()

        abs_error.append(np.linalg.norm(np.abs(g_new - g_sol)))
        amp_diff.append(np.linalg.norm(np.abs(np.abs(g_new) - np.abs(g_sol))))
        phase_diff.append(np.linalg.norm(np.abs(np.angle(g_new) - np.angle(g_sol))))

    G_new = np.diag(g_new)
    print("Did not converge before max iteration")
    return G_new, diff, abs_error, amp_diff, phase_diff


def compute_array_pattern(
    G_sol, dim, v_theta, v_phi, x_pos, y_pos, theta, phi, theta_0, phi_0, frequency
):
    """
    @brief Compute the array pattern for a given set of EEPs

    @param G_sol Exact gain solution, np.array, complex
    @param dim Dimension of the beam, can be 1 or 2, int
    @param v_theta EEPs for theta, np.array, complex
    @param v_phi EEPs for phi, np.array, complex
    @param x_pos x position of antennas, np.array, float
    @param y_pos y position of antennas, np.array, float
    @param theta Zenith angle, np.array, float
    @param phi Azimuth angle, np.array, float
    @param theta_0 Zenith angle of the source, float
    @param phi_0 Azimuth angle of the source, float
    @param frequency Frequency of the source, float

    @return np.array (complex double) array_pattern
    """

    freq = frequency * (10**6)  # frequency
    c0 = 299792458  # speed of light
    k0 = 2 * np.pi / (c0 / freq)  # wavenumber

    # Check if the gain solution is diagonal
    if not np.allclose(G_sol, np.diag(np.diagonal(G_sol))):
        raise ValueError("G_sol is not diagonal")

    g = np.diag(G_sol)

    EEP = np.sqrt(np.abs(v_theta) ** 2 + np.abs(v_phi) ** 2)

    if dim == 1:
        array_pattern = np.zeros((theta.shape[0]), dtype=complex)
        phase_factor = np.zeros((theta.shape[0]), dtype=complex)
    else:
        array_pattern = np.zeros((theta.shape[0], phi.shape[0]), dtype=complex)
        phase_factor = np.zeros((theta.shape[0], phi.shape[0]), dtype=complex)

    for i in range(256):
        w = np.exp(
            1j
            * k0
            * (
                (x_pos[i] * np.sin(theta_0) * np.cos(phi_0))
                + (y_pos[i] * np.sin(theta_0) * np.sin(phi_0))
            )
        )
        phase_factor = np.exp(
            -1j
            * k0
            * (
                (x_pos[i] * np.sin(theta) * np.cos(phi))
                + (y_pos[i] * np.sin(theta) * np.sin(phi))
            )
        )

        if dim == 1:
            array_pattern += w * g[i] * EEP[:, i] * phase_factor[:, 0]
        else:
            array_pattern += w * g[i] * EEP[:, :, i] * phase_factor[:, :]

    return np.abs(array_pattern)
