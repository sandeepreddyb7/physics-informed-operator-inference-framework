#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:23:39 2022

@author: bukka
"""

from scipy import linalg
import scipy.linalg as la

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import rom_operator_inference as roi

import os


## change the pathname accordingly to load the corresponding denoised data
pathname = "results_burgers/results_hf_32_hl_3_ep_2_noise_30_perc_100/u_data_hf_32_hl_3_ep_2_noise_30_perc_100.npy"
pltname_u_opf = "results_burgersresults_hf_32_hl_3_ep_2_noise_30_perc_100/u_opf_hf_32_hl_3_ep_2_noise_30_perc_100"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()


U_pred_full = data_load["pinn"]
Exact_full = data_load["orig"]
filter_u = data_load["filter"]


data = scipy.io.loadmat("data/burgers.mat")


## preparing and normalizing the input and output data

t_full = data["t"].flatten()[:, None]
dt = t_full[1] - t_full[0]
x = data["x"].flatten()[:, None]


### operator inference in denoised data


input_con = np.zeros((100,))
U_pred_t = U_pred_full.T
U_final_opf = U_pred_t[:, 1:101]
U_pred_t_exact = Exact_full.T
U_final_opf_filter = filter_u.T[:, 1:101]
U_final_opf_exact = U_pred_t_exact[:, 1:101]
U_final_opf_extra = U_pred_t[:, 1:51]
U_final_opf_extra_clean = U_pred_t_exact[:, 1:51]
U_final_opf_extra_filter = U_final_opf_filter[:, 0:51]

t_intg = np.reshape(t_full[1:101], (100,))


### operator inference for extrapolation

time_deriv_extra = roi.pre.ddt_uniform(U_final_opf_extra, dt, order=6)


V_extra, svdvals_extra = roi.pre.pod_basis(U_final_opf_extra)
roi.pre.minimal_projection_error(
    U_final_opf_extra, V_extra[:, :50], eps=1e-5, plot=True
)
Vr_extra = V_extra[:, :10]


model_opf_extra = roi.InferredContinuousROM(modelform="AH")


model_opf_extra.fit(Vr_extra, U_final_opf_extra, time_deriv_extra, P=1e-4)

input_condition_extra = U_pred_t[:, 0]


U_final_ROM_extra = model_opf_extra.predict(
    input_condition_extra, t_intg, method="BDF", max_step=0.1
)

error_pinn = np.abs(U_final_opf - U_final_ROM_extra) / linalg.norm(U_final_opf, "fro")


### operator inference for extrapolation on cleaned data

time_deriv_extra_clean = roi.pre.ddt_uniform(U_final_opf_extra_clean, dt, order=6)


V_extra_clean, svdvals_extra_clean = roi.pre.pod_basis(U_final_opf_extra_clean)
roi.pre.minimal_projection_error(
    U_final_opf_extra_clean, V_extra_clean[:, :50], eps=1e-5, plot=True
)
Vr_extra_clean = V_extra_clean[:, :10]


model_opf_extra_clean = roi.InferredContinuousROM(modelform="AH")


model_opf_extra_clean.fit(
    Vr_extra_clean, U_final_opf_extra_clean, time_deriv_extra_clean, P=1e-4
)

input_condition_extra_clean = U_pred_t_exact[:, 0]


U_final_ROM_extra_clean = model_opf_extra_clean.predict(
    input_condition_extra_clean, t_intg, method="BDF", max_step=0.1
)

error_orig = np.abs(U_final_opf_exact - U_final_ROM_extra_clean) / linalg.norm(
    U_final_opf_exact, "fro"
)


### operator inference for extrapolation on filtered data

time_deriv_extra_filter = roi.pre.ddt_uniform(U_final_opf_extra_filter, dt, order=6)


V_extra_filter, svdvals_extra_filter = roi.pre.pod_basis(U_final_opf_extra_filter)
roi.pre.minimal_projection_error(
    U_final_opf_extra_filter, V_extra_filter[:, :50], eps=1e-5, plot=True
)
Vr_extra_filter = V_extra_filter[:, :10]


model_opf_extra_filter = roi.InferredContinuousROM(modelform="AH")


model_opf_extra_filter.fit(
    Vr_extra_filter, U_final_opf_extra_filter, time_deriv_extra_filter, P=1e-4
)

input_condition_extra_filter = filter_u.T[:, 0]


U_final_ROM_extra_filter = model_opf_extra_filter.predict(
    input_condition_extra_filter, t_intg, method="BDF", max_step=0.1
)

error_filter = np.abs(U_final_opf_filter - U_final_ROM_extra_filter) / linalg.norm(
    U_final_opf_filter, "fro"
)

fig4 = plt.figure(figsize=(18, 12))
ax = fig4.add_subplot(331)


h1 = ax.imshow(
    U_final_opf_exact.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    U_final_ROM_extra_clean.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    U_final_opf.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    U_final_ROM_extra.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    U_final_opf_filter.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    U_final_ROM_extra_filter.T,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
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
