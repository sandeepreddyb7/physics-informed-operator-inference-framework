#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:24:06 2022

@author: bukka
"""


from scipy import linalg
import scipy.linalg as la

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

from Functions.utils import implicit_euler

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import rom_operator_inference as roi

import os


## change the pathname accordingly to load the corresponding denoised data
pathname = "results_heat/results_hf_32_hl_3_ep_2_noise_30_perc_100/u_data_hf_32_hl_3_ep_2_noise_30_perc_100.npy"
pltname_u_opf = "results_heat/results_hf_32_hl_3_ep_2_noise_30_perc_100/u_opf_hf_32_hl_3_ep_2_noise_30_perc_100"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()


U_pred_full = data_load["pinn"]
Exact_full = data_load["orig"]
filter_u = data_load["filter"]

heat_data = np.load("data/data_heat.npy", allow_pickle=True)

data = heat_data.item()


input_func = np.ones_like  # Constant input function u(t) = 1.
t_rom = data["t"]
x_rom = data["x"]
x_all = data["x_all"]
U_all = input_func(t_rom)
t_full = data["t"].flatten()[0:100, None]
dt = t_full[1] - t_full[0]
### operator inference in denoised data


U_pred_t = U_pred_full.T
U_final_opf = U_pred_t[:, 0:100]
U_pred_t_exact = Exact_full.T
U_final_opf_filter = filter_u.T
U_final_opf_exact = U_pred_t_exact[:, 0:100]
U_final_opf_extra = U_pred_t[:, 0:50]
U_final_opf_extra_clean = U_pred_t_exact[:, 0:50]
U_final_opf_extra_filter = U_final_opf_filter[:, 0:50]


### operator inference for extrapolation

time_deriv_extra = roi.pre.ddt_uniform(U_final_opf_extra, dt, order=6)


V_extra, svdvals_extra = roi.pre.pod_basis(U_final_opf_extra)
roi.pre.minimal_projection_error(
    U_final_opf_extra, V_extra[:, :50], eps=1e-5, plot=True
)
Vr_extra = V_extra[:, :4]


model_opf_extra = roi.InferredContinuousROM(modelform="AB")


model_opf_extra.fit(Vr_extra, U_final_opf_extra, time_deriv_extra, U_all[:50], P=1e-1)


## initial condition
alpha = 100
q0 = np.exp(alpha * (x_rom - 1)) + np.exp(-alpha * x_rom) - np.exp(-alpha)

# Project the initial condition.
q0_ = Vr_extra.T @ q0

U_final_ROM_extra = Vr_extra @ implicit_euler(
    t_rom[:100], q0_, model_opf_extra.A_, model_opf_extra.B_, U_all[:100]
)


## adding the boundary points
U_final_ROM_extra_f = np.zeros((129, 100))

U_final_ROM_extra_f[0, :] = np.ones((100))
U_final_ROM_extra_f[-1, :] = np.ones((100))
U_final_ROM_extra_f[1:-1, :] = U_final_ROM_extra


U_final_opf_f = np.zeros((129, 100))
U_final_opf_f[0, :] = np.ones((100))
U_final_opf_f[-1, :] = np.ones((100))
U_final_opf_f[1:-1, :] = U_final_opf

error_pinn = np.abs(U_final_opf_f - U_final_ROM_extra_f) / linalg.norm(
    U_final_opf_f, "fro"
)


### operator inference for extrapolation on cleaned data

time_deriv_extra_clean = roi.pre.ddt_uniform(U_final_opf_extra_clean, dt, order=6)


V_extra_clean, svdvals_extra_clean = roi.pre.pod_basis(U_final_opf_extra_clean)
roi.pre.minimal_projection_error(
    U_final_opf_extra_clean, V_extra_clean[:, :50], eps=1e-5, plot=True
)
Vr_extra_clean = V_extra_clean[:, :4]


model_opf_extra_clean = roi.InferredContinuousROM(modelform="AB")


model_opf_extra_clean.fit(
    Vr_extra_clean, U_final_opf_extra_clean, time_deriv_extra_clean, U_all[:50], P=1e-1
)


# Project the initial condition.
q0_ = Vr_extra_clean.T @ q0
U_final_ROM_extra_clean = Vr_extra_clean @ implicit_euler(
    t_rom[:100], q0_, model_opf_extra_clean.A_, model_opf_extra_clean.B_, U_all[:100]
)

U_final_ROM_extra_clean_f = np.zeros((129, 100))

U_final_ROM_extra_clean_f[0, :] = np.ones((100))
U_final_ROM_extra_clean_f[-1, :] = np.ones((100))
U_final_ROM_extra_clean_f[1:-1, :] = U_final_ROM_extra_clean


U_final_opf_exact_f = np.zeros((129, 100))
U_final_opf_exact_f[0, :] = np.ones((100))
U_final_opf_exact_f[-1, :] = np.ones((100))
U_final_opf_exact_f[1:-1, :] = U_final_opf_exact

error_orig = np.abs(U_final_opf_exact_f - U_final_ROM_extra_clean_f) / linalg.norm(
    U_final_opf_exact_f, "fro"
)


### operator inference for extrapolation on filtered

time_deriv_extra_filter = roi.pre.ddt_uniform(U_final_opf_extra_filter, dt, order=6)


V_extra_filter, svdvals_extra_filter = roi.pre.pod_basis(U_final_opf_extra_filter)
roi.pre.minimal_projection_error(
    U_final_opf_extra_filter, V_extra_filter[:, :50], eps=1e-5, plot=True
)
Vr_extra_filter = V_extra_filter[:, :4]


model_opf_extra_filter = roi.InferredContinuousROM(modelform="AB")


model_opf_extra_filter.fit(
    Vr_extra_filter,
    U_final_opf_extra_filter,
    time_deriv_extra_filter,
    U_all[:50],
    P=1e-1,
)


# Project the initial condition.
q0_ = Vr_extra_filter.T @ q0
U_final_ROM_extra_filter = Vr_extra_filter @ implicit_euler(
    t_rom[:100], q0_, model_opf_extra_filter.A_, model_opf_extra_filter.B_, U_all[:100]
)


U_final_ROM_extra_filter_f = np.zeros((129, 100))

U_final_ROM_extra_filter_f[0, :] = np.ones((100))
U_final_ROM_extra_filter_f[-1, :] = np.ones((100))
U_final_ROM_extra_filter_f[1:-1, :] = U_final_ROM_extra_filter


U_final_opf_filter_f = np.zeros((129, 100))
U_final_opf_filter_f[0, :] = np.ones((100))
U_final_opf_filter_f[-1, :] = np.ones((100))
U_final_opf_filter_f[1:-1, :] = U_final_opf_filter

error_filter = np.abs(U_final_opf_filter_f - U_final_ROM_extra_filter_f) / linalg.norm(
    U_final_opf_filter_f, "fro"
)


fig4 = plt.figure(figsize=(18, 12))
ax = fig4.add_subplot(331)


h1 = ax.imshow(
    U_final_opf_exact_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h1, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Original", fontsize=20)

ax = fig4.add_subplot(332)


h2 = ax.imshow(
    U_final_ROM_extra_clean_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h2, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Op Inf", fontsize=20)


ax = fig4.add_subplot(333)

h3 = ax.imshow(
    error_orig.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h3, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Normalized error", fontsize=20)


ax = fig4.add_subplot(334)


h4 = ax.imshow(
    U_final_opf_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h4, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Denoised", fontsize=20)

ax = fig4.add_subplot(335)


h5 = ax.imshow(
    U_final_ROM_extra_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h5, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Op Inf", fontsize=20)


ax = fig4.add_subplot(336)

h6 = ax.imshow(
    error_pinn.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h6, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Normalized error", fontsize=20)

ax = fig4.add_subplot(337)


h7 = ax.imshow(
    U_final_opf_filter_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h7, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Filtered", fontsize=20)

ax = fig4.add_subplot(338)


h8 = ax.imshow(
    U_final_ROM_extra_filter_f.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h8, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Op Inf", fontsize=20)


ax = fig4.add_subplot(339)

h9 = ax.imshow(
    error_filter.T,
    interpolation="nearest",
    extent=[x_all.min(), x_all.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig4.colorbar(h9, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Normalized error", fontsize=20)


fig4.tight_layout(pad=3.0)
plt.savefig(pltname_u_opf, dpi=300)
