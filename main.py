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
    Res = 40

    L2d = [71, 66]

    M = 5

    # Sub array lengths
    Ls = 40

    # For : getSubarray
    N_row = 71
    N_column = 66
    L1 = 10
    L2 = 10

    # plot
    plot = 1

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")
    f = dat["f"]

    f0 = dat['f0']

    # Time domain:
    x = dat['x_synthetic']

    # Freq. domain
    X = dat['X_synthetic']

    idx_array = fun.getSubarray(N_row, N_column, L1, L2, 1)

    # How many freq. points we want to look at
    idx_tau = np.arange(0, np.size(dat['tau'], axis=0))

    # We need a L*Lf vector. Need to flatten it columnmajor (Fortran)
    X_sub = X[idx_array, idx_tau].flatten(order='F')
    X_sub = X_sub.reshape(len(X_sub), 1, order='F')

    # ----- Spatial Smoothing -----
    print("smooth start")
    RFB = fun.spatialSmoothing(X_sub,
                               np.array([L1, L2, len(idx_tau)]),
                               np.array([4, 4, 101]))
    print("smooth done")
    idx_array_v2 = fun.getSubarray(L1, L2, 4, 4, 1)

    # Need to use spatial smoothing when usin MUSIC as rank is 1
    # R = X_sub @ (np.conjugate(X_sub).T)

    # Do the MUSIC
    print("RA")
    Pm = fun.barlettRA(X[idx_array, idx_tau], Res, dat, idx_tau, idx_array)

    print("MUSIC")
    PmM = fun.MUSIC(RFB, Res, dat, idx_tau, idx_array[idx_array_v2], M)

    # %% Plot
    Theta = np.linspace(0, np.pi, Res)
    
    if plot == 1:
        plt.figure()
        plt.title(f"PM - Sweep - res: {Res}, SNR: {10}db")
        plt.imshow(Pm.T, norm=LogNorm(),
                   extent=[0, 360,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [rad]")

        plt.figure()
        plt.title(f"PMM - Sweep - res: {Res}, SNR: {10}db")
        plt.imshow(PmM.T, norm=LogNorm(),
                   extent=[0, 360,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [rad]")

    elif plot == 2:
        P2 = Pm * 1000
        Tau = dat['tau'] * 1E6
        Angles = Theta * 180 / np.pi
        fig2 = go.Figure(data=[go.Surface(z=P2, x=Angles,
                                          y=Tau.reshape(len(Tau)))])

        fig2.update_layout(scene=dict(
            xaxis_title='Azimuth Angle - degrees',
            yaxis_title='Delay - micro-seconds'),
            title=f"Sweep - res: {Res}X{Res} points, SNR: {10}dB",
        )

        fig2.show()
    
    print("Hello World")  # Prints "Hello World"
