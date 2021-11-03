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
    Res = [100, 100]  # Res for [theta, tau]

    M = 5

    N_row = 71
    N_column = 66
    freq_samples = 101

    # For: getSubarray
    L1 = 6  # Number of sub rows
    L2 = 6  # Number of sub columns
    L3 = 101  # Number of sub samples

    array_size = [N_row, N_column, freq_samples]
    subarray_size = [L1, L2, L3]

    tau_search = [0, 5e-7]

    # plot
    plot = 1

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")

    # Freq. domain
    X = dat['X_synthetic']

    # X = np.arange(0,N_row*N_column*freq_samples,1)
    # X = np.reshape(X,(N_row*N_column,freq_samples))
    # test = fun.getSubarray(array_size, subarray_size, offset = [1,1,10], spacing=1)
    # X_sub = X[test[0], test[1]].flatten(order='F')
    # Index data vector for antennas in subarray
    idx_array = fun.getSubarray(array_size, subarray_size, offset = [0,0,0], spacing=2)

    # We need a L*Lf vector. Need to flatten it columnmajor (Fortran)
    X_sub = X[idx_array[0], idx_array[1]].flatten(order='F')
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
    #print("RA")
    #Pm = fun.barlettRA(X[idx_array, idx_tau], Res, dat, idx_tau, idx_array)
    # PmM = fun.test(X[idx_array, idx_tau], Res, dat, idx_tau, idx_array)

    print("MUSIC")
    # PmMM = fun.MUSIC(RFB, Res, dat, idx_tau, idx_array[idx_array_v2], M)
    # Pm_Capon = fun.capon(R, Res, dat, idx_tau, idx_array)
    Pm_Barlett = fun.barlett(R, Res, dat, idx_array[1], idx_array[0], tau_search)

    # %% Plot
    Theta = np.linspace(0, np.pi, Res[0])
    AoA = (dat['smc_param'][0][0][1])*180/np.pi
    TDoA = (dat['smc_param'][0][0][2])*(1/3e8) + np.abs(dat['tau'][0])

    if plot == 1:
        """
        plt.figure()
        plt.title(f"PM - Sweep - res: {Res}")
        plt.imshow(np.abs(Pm.T), norm=LogNorm(),
                   extent=[0, 360,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")
        """
        plt.figure()
        plt.title(f"Barlett- Sweep - res: {Res}")
        plt.scatter(AoA, TDoA, color='r', marker='x')
        pm_max = np.max(10*np.log10(Pm_Barlett.T))
        # pm_max = 20
        plt.imshow(10*np.log10(Pm_Barlett), vmin=pm_max-40, vmax=pm_max,
                   extent=[-180, 180,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
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
"""
        plt.figure()
        plt.title(f"Barlett- Sweep - res: {Res}")
        plt.imshow(Pm_Barlett.T, norm=LogNorm(vmin=0.01),
                   extent=[-180, 180,
                           np.min(dat['tau']), np.max(dat['tau'])],
                   aspect="auto")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")

"""

# %%
if plot == 1:
    from matplotlib import cm
    x = np.linspace(-180, 180, Res[0])
    y = np.linspace(tau_search[0], tau_search[1], Res[1], endpoint=True)
    z = 10*np.log10(Pm_Barlett)

    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    plt.ylabel("Tau [s]")
    plt.xlabel("Theta [degrees]")
    plt.show()
