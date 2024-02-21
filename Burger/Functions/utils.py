#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:58:44 2022

@author: bukka
"""

import torch
import numpy as np
from scipy import linalg
import scipy.linalg as la


def loss_func(u, u_pred, f_pred):

    loss_data = torch.mean((u - u_pred) ** 2)
    loss_eq = torch.mean(f_pred ** 2)
    loss = loss_data + loss_eq

    return loss, loss_data, loss_eq


def implicit_euler(t, q0, A, B, U):
    """Solve the system

        dq / dt = Aq(t) + Bu(t),    q(0) = q0,

    over a uniform time domain via the implicit Euler method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    q0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.
    B : (n,) or (n, 1) ndarray
        Input matrix.
    U : (k,) ndarray
        Inputs over the time array.

    Returns
    -------
    q : (n, k) ndarray
        Solution to the ODE at time t; that is, q[:,j] is the
        computed solution corresponding to time t[j].
    """
    # Check and store dimensions.
    k = len(t)
    n = len(q0)
    B = np.ravel(B)
    assert A.shape == (n, n)
    assert B.shape == (n,)
    assert U.shape == (k,)
    I = np.eye(n)

    # Check that the time step is uniform.
    dt = t[1] - t[0]
    assert np.allclose(np.diff(t), dt)

    # Factor I - dt*A for quick solving at each time step.
    factored = la.lu_factor(I - dt * A)

    # Solve the problem at each time step.
    q = np.empty((n, k))
    q[:, 0] = q0.copy()
    for j in range(1, k):
        q[:, j] = la.lu_solve(factored, q[:, j - 1] + dt * B * U[j])

    return q
