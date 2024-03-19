"""
@brief Plotting functions for the harp beam.

"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_EEP_AEP(theta, EEP, pol):
    """ 
    """
    for i in range(EEP.shape[1]):
        plt.plot(theta, EEP[:, i], color = 'grey', alpha = 0.5, label = "EEP's" if i == 0 else '')

    plt.plot(theta, np.mean(EEP, axis = 1), label = 'AEP', color = 'black')
    plt.xlabel('Theta (radians)')
    plt.ylabel('E-field (dBV)')
    plt.title(pol + ', EEPs, $\phi$ = 0')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "EEP_AEP_"+pol+".png")
    plt.savefig(plot_path)
    plt.close()


    