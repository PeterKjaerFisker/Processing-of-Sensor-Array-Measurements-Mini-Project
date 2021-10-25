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


# %% functions


# %% Main

if __name__ == '__main__':

    # ---- Parameters ----
    Res = 10

    L2d = [71, 66]

    M = 5

    # ---- Initialise data ----
    # Load datafile
    dat = scio.loadmat("data.mat")
    f = dat["f"]

    f0 = dat['f0']

    # Time domain:
    x = dat['x_synthetic']

    # Freq. domain
    X = dat['X_synthetic']

    # pos
    r = dat['r']

    # Lambda:
    lambda_ = 3e8/f0

    # Get Subarray
    N_row = 71
    N_column = 66
    L1 = 4
    L2 = 4
    idx_array = fun.getSubarray(N_row, N_column, L1, L2, 1)

    idx_tau = np.arange(0, np.size(dat['tau'], axis=0))

    # Theta = np.linspace(0, np.pi, Res)

    # Ts = dat['tau'][1]-dat['tau'][0]  # Delay Spacing

    # Tau = np.linspace(0, 1, Res)

    R = np.cov(X[idx_array, idx_tau], bias=True)

    Pm = fun.MUSIC(R, Res, M, np.prod(L2d),
                   r[:, idx_array].reshape(2, L1*L2), f0)

    """
    for theta in Theta:
        for tau in Tau:
            tmp = fun.delay_respons_vector(lambda_, theta, r, f0, tau)
    """
    print("Hello World")  # Prints "Hello World"
