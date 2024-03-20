"""!@file plot_funcs.py

@brief Plotting functions for the calibration of the ska-low station mini-project.

@details This module contains functions to plot the EEPs and AEPs,
and the convergence of the SteEFCal algorithm.

@author Created by T.Breitburd on 19/03/2024
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_EEP_AEP(theta, EEP, pol):
    """
    @brief Plot the EEPs and AEPs for a given polarization.

    @param theta Zenith angle, np.array, float
    @param EEP EEPs, np.array, float
    @param pol Polarization, str

    @return Plot of the EEPs and AEPs for a given polarization.
    """
    for i in range(EEP.shape[1]):
        plt.plot(
            theta, EEP[:, i], color="grey", alpha=0.5, label="EEP's" if i == 0 else ""
        )

    plt.plot(theta, np.mean(EEP, axis=1), label="AEP", color="black")
    plt.xlabel("Theta (radians)")
    plt.ylabel("E-field (dBV)")
    plt.title(pol + ", EEPs, $\phi$ = 0")  # noqa: W605
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "EEP_AEP_" + pol + ".png")
    plt.savefig(plot_path)
    plt.close()


def plot_convergence(errors, iteration_number, model, error_type):
    """
    @brief Plot the convergence of the SteEFCal algorithm.

    @param errors Absolute errors of the gain solutions, np.array, float
    @param iteration_number Number of iterations, int
    @param error_name Name of the error, str

    @return Plot of the absolute errors of the gain solutions vs the iteration number.
    """

    plt.plot(errors[:iteration_number])
    plt.xlabel("Iteration")
    plt.ylabel(error_type + " absolute error")
    plt.title(
        error_type
        + " absolute error vs iteration number for the"
        + model
        + " model matrix"
    )
    plt.grid()
    plt.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(
        plot_dir, model + "_" + error_type + "_error@" + str(iteration_number) + ".png"
    )
    plt.savefig(plot_path)
    plt.close()


def plot_station_beam(
    theta, P_dB_X, P_dB_Y, P_EEPs_dB_X, P_EEPs_dB_Y, P_AEP_dB_X, P_AEP_dB_Y, frequency
):
    """
    @brief Plot the station beam

    @param theta Zenith angle, np.array, float
    @param P_dB_X True array pattern for X feed, np.array, float
    @param P_dB_Y True array pattern for Y feed, np.array, float
    @param P_EEPs_dB_X Estimated array pattern for X feed using EEPs, np.array, float
    @param P_EEPs_dB_Y Estimated array pattern for Y feed using EEPs, np.array, float
    @param P_AEP_dB_X Estimated array pattern for X feed using AEP, np.array, float
    @param P_AEP_dB_Y Estimated array pattern for Y feed using AEP, np.array, float
    @param frequency Frequency, float

    @return Plot of the station beam
    """

    plt.style.use("ggplot")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    ax[0].plot(theta, P_dB_X, label="True array pattern X feed", color="black")
    ax[0].plot(
        theta,
        P_EEPs_dB_X,
        label="Estimated array pattern X feed using EEPs",
        color="red",
    )
    ax[0].plot(
        theta,
        P_AEP_dB_X,
        label="Estimated array pattern X feed using AEP",
        color="blue",
    )
    ax[0].set_xlabel("Theta")
    ax[0].set_ylabel("Voltage (dBV)")
    ax[0].set_title("Station beam for X feed, frequency = " + str(frequency) + "MHz")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(theta, P_dB_Y, label="True array pattern Y feed", color="black")
    ax[1].plot(
        theta,
        P_EEPs_dB_Y,
        label="Estimated array pattern Y feed using EEPs",
        color="red",
    )
    ax[1].plot(
        theta,
        P_AEP_dB_Y,
        label="Estimated array pattern Y feed using AEP",
        color="blue",
    )
    ax[1].set_xlabel("Theta")
    ax[1].set_ylabel("Voltage (dBV)")
    ax[1].set_title("Station beam for Y feed, frequency = " + str(frequency) + "MHz")
    ax[1].grid()
    ax[1].legend()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "Station_beam.png")
    plt.savefig(plot_path)
    plt.close()
