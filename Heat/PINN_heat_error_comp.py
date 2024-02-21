#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 12:59:50 2022

@author: bukka
"""
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np

## loading data corresponding to noise of 5 percent
pathname = "results_heat/results_hf_32_hl_3_ep_2_noise_5_perc_100/u_data_hf_32_hl_3_ep_2_noise_5_perc_100.npy"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()

error_pinn_5 = data_load["error_pinn"]
error_filter_5 = data_load["error_filter"]

error_pinn_5_norm = linalg.norm(error_pinn_5, "fro")
error_filter_5_norm = linalg.norm(error_filter_5, "fro")

## loading data corresponding to noise of 10 percent
pathname = "results_heat/results_hf_32_hl_3_ep_2_noise_10_perc_100/u_data_hf_32_hl_3_ep_2_noise_10_perc_100.npy"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()


error_pinn_10 = data_load["error_pinn"]
error_filter_10 = data_load["error_filter"]

error_pinn_10_norm = linalg.norm(error_pinn_10, "fro")
error_filter_10_norm = linalg.norm(error_filter_10, "fro")

## loading data corresponding to noise of 20 percent
pathname = "results_heat/results_hf_32_hl_3_ep_2_noise_20_perc_100/u_data_hf_32_hl_3_ep_2_noise_20_perc_100.npy"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()


error_pinn_20 = data_load["error_pinn"]
error_filter_20 = data_load["error_filter"]

error_pinn_20_norm = linalg.norm(error_pinn_20, "fro")
error_filter_20_norm = linalg.norm(error_filter_20, "fro")

## loading data corresponding to noise of 30 percent
pathname = "results_heat/results_hf_32_hl_3_ep_2_noise_30_perc_100/u_data_hf_32_hl_3_ep_2_noise_30_perc_100.npy"

data_load = np.load(pathname, allow_pickle=True)

data_load = data_load.item()


error_pinn_30 = data_load["error_pinn"]
error_filter_30 = data_load["error_filter"]

error_pinn_30_norm = linalg.norm(error_pinn_30, "fro")
error_filter_30_norm = linalg.norm(error_filter_30, "fro")


error_perc = [5, 10, 20, 30]

error_pinn = [
    error_pinn_5_norm,
    error_pinn_10_norm,
    error_pinn_20_norm,
    error_pinn_30_norm,
]
error_filter = [
    error_filter_5_norm,
    error_filter_10_norm,
    error_filter_20_norm,
    error_filter_30_norm,
]

fig3 = plt.figure(figsize=(6, 4))
plt.semilogy(
    error_perc, error_pinn, "*k--", label="Physics informed", linewidth=2, markersize=12
)
plt.semilogy(
    error_perc,
    error_filter,
    "*r--",
    label="Low pass filter",
    linewidth=2,
    markersize=12,
)

plt.legend(loc="best", frameon=False, fontsize=20)
plt.xlabel("Noise level", fontsize=20)
plt.ylabel("Normalized error", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("results_heat/comp_error_filter_pinn.png", dpi=300)

fig3 = plt.figure(figsize=(6, 4))
plt.plot(
    error_perc, error_pinn, "*k--", label="Physics informed", linewidth=2, markersize=12
)
plt.plot(
    error_perc,
    error_filter,
    "*r--",
    label="Low pass filter",
    linewidth=2,
    markersize=12,
)

plt.legend(loc="best", frameon=False, fontsize=20)
plt.xlabel("Noise level", fontsize=20)
plt.ylabel("Normalized error", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("results_heat/comp_error_filter_pinn_nolog.png", dpi=300)
