# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@authors: Nicolai Almskou Rasmussen, Victor Mølbach Nissen,
          Peter Kjær Fisker, Claus Meyer Larsen & Dennis Kjærsgaard Sand
"""

# %% Imports

import numpy as np
import scipy.io as scio
import functions as fun

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

# %% functions


# %% Main

if __name__ == '__main__':
    # ---- Parameters ----
    Res = [500, 500]  # Resolution for [theta, tau]

    # SNR in [dB]
    SNRdb = 'inf'

    # Number of sources from synthetic data
    M = 5

    # Matrix dim. of given data
    N_row = 71
    N_column = 66
    N_freq_samples = 101

    # Dimensions for sub-array
    N_row_subarray = 19
    N_column_subarray = 19
    N_freq_samples_subarray = 20

    # Dimensions for array used in spacial smoothing
    N_row_smoothingarray = 10
    N_column_smoothingarray = 10
    N_freq_samples_smoothingarray = 10

    # Create numpy-arrays with dimensions
    array_size = np.array([N_row, N_column, N_freq_samples])
    subarray_size = np.array([N_row_subarray, N_column_subarray, N_freq_samples_subarray])
    smoothing_array_size = np.array([N_row_smoothingarray, N_column_smoothingarray, N_freq_samples_smoothingarray])

    # Search Space for delays, Tau: 0 ns - 300 ns.
    tau_search = [0, 5e-7]

    # Flag to determine the type of plots:
    # 1 = Heatmaps w. matplotlib (Bartlett, Capon & MUSIC)
    # 2 = 3D-plot w. plotly (only MUSIC)
    plot = 1

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")

    # Choose synthetic or measured data
    synthetic = True

    # Choose to add noise to data
    noise = False

    # Get samples from freq. domain measurement
    if synthetic:
        X = dat['X_synthetic']  # Synthetic data w.o. noise, for algorithm tests
    else:
        X = dat['X']  # Real measured data

    # ---- Add noise ----
    if synthetic and noise:
        # SNR in [dB]
        SNRdb = -10
        X = fun.addNoise(X, SNRdb)

    # Get the index of the antennas and freq. samples in the chosen subarray.
    idx_array = fun.getSubarray(array_size, subarray_size, offset=[0, 0, 0], spacing=2)

    # Pick the data elements corresponding to the antennas in the subarray
    X_sub = X[idx_array[0], idx_array[1]]

    # ----- Estimate Covariance Matrix -----

    # For Capon and MUSIC we need full rank estimation of covariance
    # so we use spatial smoothing to increase rank at cost of resolution
    print("smooth start")
    RFB = fun.spatialSmoothing(X_sub, subarray_size, smoothing_array_size)

    # For Bartlett we use the subarray data as is:
    X_sub = X_sub.flatten(order='F').reshape(len(X_sub.flatten(order='F')), 1, order='F')
    R = X_sub @ np.conjugate(X_sub).T  # Estimation of true covariance

    # ----- Run Algorithms -----
    print("Algorithms")

    print("Bartlett")
    Pm_Barlett = fun.bartlett(R, Res, dat, idx_array[1], idx_array[0], tau_search)

    # Indexes for picking antennas and samples when using smoothed covariance
    idx_smoothingarray = fun.getSubarray(array_size, smoothing_array_size, offset=[0, 0, 0], spacing=2)

    print("Capon")
    Pm_Capon = fun.capon(RFB, Res, dat, idx_smoothingarray[1], idx_smoothingarray[0], tau_search)

    print("MUSIC")
    Pm_MUSIC = fun.MUSIC(RFB, Res, dat, idx_smoothingarray[1], idx_smoothingarray[0], tau_search, M=M)

    # ----- Plots -----

    # Angles and delays of "true" sources, given in dataset. Used for verification.
    AoA = (dat['smc_param'][0][0][1]) * 180 / np.pi  # In degrees
    AoA[AoA < 0] = AoA[AoA < 0] + 360  # place negative angles in right location

    TDoA = ((dat['smc_param'][0][0][2]) * (1 / 3e8) + np.abs(dat['tau'][0])) * 1E9  # In nano-seconds

    if plot == 1:

        plt.figure(1)
        plt.title(f"Bartlett - resolution: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10 * np.log10(Pm_Barlett))
        plt.imshow(10 * np.log10(Pm_Barlett), vmin=pm_max - 40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0] * 1E9, tau_search[1] * 1E9],
                   aspect="auto", origin="lower")
        plt.colorbar().set_label('Output Power [dB]', size=12)
        plt.ylabel("Tau [ns]")
        plt.xlabel("Theta [degrees]")
        plt.show()

        plt.figure(2)
        plt.title(f"Capon - resolution: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10 * np.log10(Pm_Capon))
        plt.imshow(10 * np.log10(Pm_Capon), vmin=pm_max - 40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0] * 1E9, tau_search[1] * 1E9],
                   aspect="auto", origin="lower")
        plt.colorbar().set_label('Output Power [dB]', size=12)
        plt.ylabel("Tau [ns]")
        plt.xlabel("Theta [degrees]")
        plt.show()

        plt.figure(3)
        plt.title(f"MUSIC - resolution: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10 * np.log10(Pm_MUSIC))
        plt.imshow(10 * np.log10(Pm_MUSIC), vmin=pm_max - 40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0] * 1E9, tau_search[1] * 1E9],
                   aspect="auto", origin="lower")
        plt.colorbar().set_label('Output Power [dB]', size=12)
        plt.ylabel("Tau [ns]")
        plt.xlabel("Theta [degrees]")
        plt.show()

    elif plot == 2:
        x = np.linspace(0, 360, Res[0])
        y = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
        z = 10 * np.log10(Pm_MUSIC)

        x, y = np.meshgrid(x, y)
        fig2 = go.Figure(data=[go.Surface(z=z, x=x,
                                          y=y)])

        fig2.update_layout(scene=dict(
            xaxis_title='Azimuth Angle - degrees',
            yaxis_title='Delay - micro-seconds'),
            title=f"Sweep - res: {Res}X{Res} points",
        )

        fig2.show()
