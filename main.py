# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np
import scipy.io as scio
import functions as fun

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

# %% functions


# %% Main

if __name__ == '__main__':
    # ---- Parameters ----
    Res = 20

    M = 5

    # For: getSubarray
    L1 = 10  # Number of sub rows
    L2 = 10  # Number of sub columns

    # plot
    plot = 1

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")

    # Freq. domain
    X = dat['X_synthetic']

    # Index data vector for antennas in subarray
    idx_array = fun.getSubarray(L1, L2, 2)

    # How many freq. points we want to look at
    idx_tau = np.arange(0, np.size(dat['tau'], axis=0))

    # We need a L*Lf vector. Need to flatten it columnmajor (Fortran)
    X_sub = X[idx_array, idx_tau].flatten(order='F')
    X_sub = X_sub.reshape(len(X_sub), 1, order='F')

    # ----- Spatial Smoothing -----
    """
    print("smooth start")
    RFB = fun.spatialSmoothing(X_sub,
                               np.array([L1, L2, len(idx_tau)]),
                               np.array([6, 6, 101]))
    print("smooth done")
    idx_array_v2 = fun.getSubarray(L1, L2, 6, 6, 1)
    """
    # Need to use spatial smoothing when using MUSIC as rank is 1
    R = X_sub @ (np.conjugate(X_sub).T)

    # Do the MUSIC
    print("RA")
    Pm = fun.barlettRA(X[idx_array, idx_tau], Res, dat, idx_tau, idx_array)
    # PmM = fun.test(X[idx_array, idx_tau], Res, dat, idx_tau, idx_array)

    print("MUSIC")
    # PmMM = fun.MUSIC(RFB, Res, dat, idx_tau, idx_array[idx_array_v2], M)
    PmMM = fun.barlett(R, Res, dat, idx_tau, idx_array)

    # %% Plot
    Theta = np.linspace(0, np.pi, Res)

    if plot == 1:
        plt.figure()
        plt.title(f"PM - Sweep - res: {Res}")
        plt.imshow(np.abs(Pm.T), norm=LogNorm(),
                   extent=[0, 360,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

        plt.figure()
        plt.title(f"Barlett- Sweep - res: {Res}")
        plt.imshow(PmMM.T, norm=LogNorm(),
                   extent=[-180, 180,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   vmin = 0.1, 
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

    elif plot == 2:
        P2 = Pm * 1000
        Tau = dat['tau'] * 1E6
        Angles = Theta * 180 / np.pi
        fig2 = go.Figure(data=[go.Surface(z=P2, x=Angles,
                                          y=Tau.reshape(len(Tau)))])

        fig2.update_layout(scene=dict(
            xaxis_title='Azimuth Angle - degrees',
            yaxis_title='Delay - micro-seconds'),
            title=f"Sweep - res: {Res}X{Res} points",
        )

        fig2.show()

    print("Hello World")  # Prints "Hello World"
