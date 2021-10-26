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
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

# %% functions


# %% Main

if __name__ == '__main__':

    # ---- Parameters ----
    Res = 10

    L2d = [71, 66]

    M = 5

    # For : getSubarray
    N_row = 71
    N_column = 66
    L1 = 4
    L2 = 4

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
    X_sub = X_sub.reshape(len(X_sub), 1)

    # Need to use spatial smoothing when usin MUSIC as rank is 1
    R = X_sub@(np.conjugate(X_sub).T)

    # Do the MUSIC
    Pm = fun.MUSIC(R, Res, M, dat, idx_tau, idx_array)

    Theta = np.linspace(0, np.pi, Res)

    fig2 = go.Figure(data=[go.Surface(z=Pm, x=dat['tau'],
                                      y=Theta * 180 / np.pi)])

    fig2.update_layout(scene=dict(
        xaxis_title='Azimuth Angle',
        yaxis_title='Elevation Angle'),
        title=f"Sweep - res: {Res}X{Res} points, SNR: {10}dB",
    )

    fig2.show()

    print("Hello World")  # Prints "Hello World"
