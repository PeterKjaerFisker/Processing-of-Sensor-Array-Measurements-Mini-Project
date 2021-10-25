# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:04:24 2021

@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np

from scipy.signal import find_peaks

# %% Functions


# ---- Modified ----
def delay_respons_vector(lambda_, theta, r, f0, tau):
    return (np.exp(-1j*2*np.pi*(1/lambda_) *
                   np.array([np.cos(theta), np.sin(theta)]).T@r) *
            np.exp(-1j*f0*tau)).reshape([np.size(r, axis=1), 1])


def MUSIC(R, Res, M, L, r, f0):
    # ------ Step 3 - Form U ------
    E, U = np.linalg.eig(R)
    Un = U[:, M:]

    # ------ Step 4 - Calculate Freq. estimate ------

    # Create the sweep parameters
    Theta = np.linspace(0, np.pi, Res)
    Tau = np.linspace(0, 1, Res)
    Pm = np.zeros([Res, Res])

    # Do the caluclations
    for i in range(len(Theta)):
        for j in range(len(Tau)):
            # Calculate for the different steering matrix
            As = delay_respons_vector(L, Theta[i], r, f0, Tau[j])

            Ash = np.conjugate(As).T
            Unh = np.conjugate(Un).T

            Pm[i, j] = 1/np.abs(Ash@Un@Unh@As)

    return Pm


# ---- Old ----
def steering_matrix_1d(L, theta, d, lambda_):
    return np.exp(1j*2*np.pi*np.arange(0, L).reshape(L, 1) *
                  (d/lambda_)*np.cos(theta))


def steering_matrix_2d(L2d, AoA, d, lambda_):
    theta, phi = AoA
    Lx, Ly = L2d
    dx, dy = d

    eta_x = (dx/lambda_)*np.sin(theta)*np.cos(phi).reshape(1, np.size(theta))
    eta_y = (dy/lambda_)*np.sin(theta)*np.sin(phi).reshape(1, np.size(theta))

    A = []

    for i in range(Lx):
        A.append(np.exp(1j*2*np.pi*(
            np.ones([Lx, 1])*i*eta_x +
            np.arange(0, Ly).reshape(Ly, 1)*eta_y)))

    return np.concatenate(A)


def get_peaks(Pm, Res, M):
    Theta = np.linspace(0, np.pi, Res)*(180/np.pi)

    peaks, _ = find_peaks(Pm.reshape(Res))
    tmp = Pm[peaks]
    tmp_sort = np.argsort(tmp, axis=0)

    return Theta[peaks[tmp_sort[-M:]]].reshape([np.min([M, len(tmp_sort)])])


def DMLE(theta, x, L, M):
    R = np.cov(x, bias=True)

    A = steering_matrix_1d(L, theta)
    AT = np.linalg.pinv(A)
    A_ort = np.eye(L)-A@AT

    return (1/(L-M))*np.trace(A_ort@R)


def ESPRIT(R, Res, M, L2d):

    E, U = np.linalg.eig(R)
    Us = U[:, :M]

    # Calculate selections matrices
    J1 = np.eye(L2d[0])[:-1, :]
    J2 = np.eye(L2d[0])[1:, :]

    Ju1 = np.kron(np.eye(L2d[1]), J1)
    Ju2 = np.kron(np.eye(L2d[1]), J2)

    Jv1 = np.kron(J1, np.eye(L2d[1]))
    Jv2 = np.kron(J2, np.eye(L2d[1]))

    # ------ Step 4 - Calculate C ------
    # Calculate subarrays
    phi_u = np.linalg.pinv(Ju1@Us)@Ju2@Us
    phi_v = np.linalg.pinv(Jv1@Us)@Jv2@Us

    # ------ Step 5 - Calculate Eigenvalues ------
    Eu, Uu = np.linalg.eig(phi_u)
    Ev, Uv = np.linalg.eig(phi_v)

    u = (1/np.pi)*np.angle(Eu)
    v = (1/np.pi)*np.angle(Ev)

    theta = np.angle(u+1j*v)
    phi = np.arcsin(np.abs(u+1j*v))

    return theta, phi
