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

    Theta = np.linspace(0, np.pi, Res)

    Tau = np.linspace(0, 1, Res)

    R = np.cov(X, bias=True)

    Pm = fun.MUSIC(R, Res, M, np.prod(L2d), r, f0)

    # for theta in Theta:
    #     for tau in Tau:
    #         tmp = fun.delay_respons_vector(lambda_, theta, r, f0, tau)

    print("Hello World")  # Prints "Hello World"
