#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch


from scipy import linalg

import scipy.linalg as la

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

from Functions.modules import Siren
from Functions.utils import loss_func

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import rom_operator_inference as roi

import os
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from scipy import signal

b, a = signal.butter(3, 0.1, btype="lowpass", analog=False)


## calculation of derivatives


def features_calc(X, u):

    u_t = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    features = torch.cat([u_t, u * u_x, u_xx], dim=1)
    return features


## equation model
class params_equation(torch.nn.Module):
    def __init__(self, num_params):
        super().__init__()

        self.params = torch.zeros((num_params, 1))
        self.params = torch.nn.Parameter(self.params)

    def forward(self, features):
        residual = torch.matmul(features[:, 1:3], self.params) + features[:, 0:1]

        return residual


### Traininig and network paramters

learning_rate_inr = 5e-4  ## learning rate value for the SIREN network
learning_rate_params = 5e-3  ## learning rate value for the equation model
num_params = 2  ## number of equation parameters to be learned
perc = 1.0  ## percentage of the date to be considered for training
noise = 0.3  ## percentage of noise to be added
in_features = 2  ## number of input features for the network
out_features = 1  ## number of output variables to be predicted
hidden_features = 32  ## number of hidden neurons in each layer of the network

hidden_layers = 3  ## number of hidden layers in the network
batch_size = 512  ## batch size for the training
num_epochs = 2  ## total number of epochs

### File saving parameters

string_f = (
    "_hf_"
    + str(hidden_features)
    + "_hl_"
    + str(hidden_layers)
    + "_ep_"
    + str(int(num_epochs))
    + "_noise_"
    + str(int(100 * noise))
    + "_perc_"
    + str(int(100 * perc))
)
result_path = "results_burgers/results" + string_f
p = Path(result_path)
if not p.exists():
    os.mkdir(result_path)

filename_u = (
    result_path + "/" + "u_data" + string_f + ".npy"
)  ## path where the primary variable phi data is saved

filename_l = (
    result_path + "/" + "Loss_collect" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved
filename_p = (
    result_path + "/" + "params_collect" + string_f + ".npy"
)  ## path where the parameter data for every epoch is saved
filename_model_u = (
    result_path + "/" + "model_u" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved


pltname_u = result_path + "/" + "u" + string_f

pltname_l = result_path + "/" + "LP" + string_f + ".png"
pltname_p1 = result_path + "/" + "PP1" + string_f + ".png"
pltname_p2 = result_path + "/" + "PP2" + string_f + ".png"
pltname_u_opf = result_path + "/" + "u_opf" + string_f

pltname_modes = result_path + "/" + "modes" + string_f


# number of samples
N_u = 25856

# loading the data
data = scipy.io.loadmat("data/burgers.mat")


## preparing and normalizing the input and output data

t_full = data["t"].flatten()[:, None]
x = data["x"].flatten()[:, None]

min_t = t_full.min()
max_t = t_full.max()


t_norm_full = ((t_full - min_t) / (max_t - min_t) - 0.5) * 2

min_x = x.min()
max_x = x.max()

# x_norm = ((x-min_x)/(max_x-min_x)-0.5)*2
x_norm = x / 8


## normalization parameters for equation. These are used to convert the coefficients back to real scale from normalized scale.
alpha = 0.2
beta = 0.125


Exact_full = np.real(data["usol"]).T


X_full, T_full = np.meshgrid(x_norm, t_norm_full)


X_star_full = np.hstack((X_full.flatten()[:, None], T_full.flatten()[:, None]))


u_star_full = Exact_full.flatten()[:, None]
N_u = u_star_full.shape[0]


# create training set

## no noise case
idx = np.random.choice(X_star_full.shape[0], N_u, replace=False)
X_u_train = X_star_full[idx, :]


## noisy case


u_star_noisy_full = u_star_full + noise * np.std(u_star_full) * np.random.randn(
    u_star_full.shape[0], u_star_full.shape[1]
)

u_train = u_star_noisy_full[idx, :]


# siren model initialization
model_inr = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)

# equation model initialization
model_params = params_equation(num_params=num_params).to(device)


# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_inr.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 1e-6,
        },
        {
            "params": model_params.parameters(),
            "lr": learning_rate_params,
            "weight_decay": 1e-6,
        },
    ]
)

# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optim_adam, step_size=500, gamma=0.1)


# converting numpy to torch
X_t = torch.tensor(X_u_train, requires_grad=True).float().to(device)
u_train_t = torch.tensor(u_train).float().to(device)

# dataloader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, u_train_t), batch_size=batch_size, shuffle=True
)

Loss_collect = np.zeros((num_epochs, 3))
params_collect = np.zeros((num_epochs, num_params))
# Training loop
for epoch in range(num_epochs):
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for batch_idx, (local_batch, output) in loop:

        ## foward pass siren
        u_pred = model_inr(local_batch)

        ## features calculation
        features = features_calc(local_batch, u_pred)

        ## equation residual
        residual = model_params(features)

        ## loss evaluation
        loss, loss_data, loss_eq = loss_func(output, u_pred, residual)

        # Backward and optimize
        optim_adam.zero_grad()
        loss.backward()
        optim_adam.step()

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(
            loss=loss.item(), loss_data=loss_data.item(), loss_eq=loss_eq.item()
        )
        # loop.set_postfix(loss = loss.item(), loss_data = loss_data.item())
        loss_epoch += loss.item()
        loss_data_epoch += loss_data.item()
        loss_eq_epoch += loss_eq.item()
    loss_epoch = loss_epoch / len(train_loader)
    loss_data_epoch = loss_data_epoch / len(train_loader)
    loss_eq_epoch = loss_eq_epoch / len(train_loader)
    # scheduler step
    scheduler.step()
    if epoch % 1 == 0:
        print(
            "It: %d, Loss: %.3e, Loss_data: %.3e, Loss_eq: %.3e, Lambda_1: %.3f, Lambda_2: %.6f"
            % (
                epoch,
                loss.item(),
                loss_data.item(),
                loss_eq.item(),
                model_params.params[0].item(),
                # torch.exp(model.lambda_2).item()
                model_params.params[1].item(),
            )
        )
        Loss_collect[epoch, 0] = loss_epoch
        Loss_collect[epoch, 1] = loss_data_epoch
        Loss_collect[epoch, 2] = loss_eq_epoch
        params_collect[epoch, 0] = model_params.params[0].item()
        params_collect[epoch, 1] = model_params.params[1].item()

        # Backward and optimize


params_collect[:, 0:1] = params_collect[:, 0:1] * alpha / beta
params_collect[:, 1:2] = params_collect[:, 1:2] * alpha / (beta ** 2)

np.save(filename_l, Loss_collect)
np.save(filename_p, params_collect)

X_star_full = torch.tensor(X_star_full, requires_grad=True).float().to(device)

u_pred_total_full = model_inr(torch.tensor(X_star_full).float())

u_pred_total_full = u_pred_total_full.cpu()
u_pred_total_full = u_pred_total_full.detach().numpy()


X_star_full = X_star_full.cpu()
X_star_full = X_star_full.detach().numpy()

U_pred_full = griddata(
    X_star_full, u_pred_total_full.flatten(), (X_full, T_full), method="cubic"
)

U_noisy_full = griddata(
    X_star_full, u_star_noisy_full.flatten(), (X_full, T_full), method="cubic"
)


filter_psi = np.zeros_like(U_noisy_full)

filter_u = signal.filtfilt(b, a, U_noisy_full)


error_full_pinn = np.abs(U_pred_full - Exact_full) / linalg.norm(Exact_full, "fro")

error_full_filter = np.abs(filter_u - Exact_full) / linalg.norm(Exact_full, "fro")


u_data = {
    "orig": Exact_full,
    "pinn": U_pred_full,
    "noisy": U_noisy_full,
    "filter": filter_u,
    "error_pinn": error_full_pinn,
    "error_filter": error_full_filter,
}

np.save(filename_u, u_data, allow_pickle=True)


fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(231)


h1 = ax.imshow(
    U_noisy_full,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Noisy:" + str(100 * noise) + "%", fontsize=20)

ax = fig.add_subplot(232)


h2 = ax.imshow(
    filter_u,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Denoised: low pass filter", fontsize=20)

ax = fig.add_subplot(233)


h3 = ax.imshow(
    U_pred_full,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Denoised: Physics informed", fontsize=20)

ax = fig.add_subplot(234)
h4 = ax.imshow(
    Exact_full,
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h4, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Original", fontsize=20)

ax = fig.add_subplot(235)
h5 = ax.imshow(
    error_full_filter[:, :],
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h5, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Normalized Error", fontsize=20)

ax = fig.add_subplot(236)
h6 = ax.imshow(
    error_full_pinn[:, :],
    interpolation="nearest",
    extent=[x.min(), x.max(), t_full.min(), t_full.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h6, cax=cax)
# cbar.ax.tick_params(labelsize=15)
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("t", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_title("Normalized Error", fontsize=20)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png", dpi=300)


fig2 = plt.figure(figsize=(6, 4))
plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
# plt.legend( loc='upper right', prop={'size': 17}, frameon=False)
plt.legend(loc="upper right", frameon=False, fontsize=20)
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("L2 Loss", fontsize=20)
plt.xticks([0, 500, 1000, 1500, 2000], fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(pltname_l, dpi=300)


fig3 = plt.figure(figsize=(6, 6))
plt.plot(params_collect[:, 0:1], label="$\lambda_1$")


plt.legend(loc="lower right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p2)

fig4 = plt.figure(figsize=(6, 6))
plt.plot(params_collect[:, 1:2], label="$\lambda_2$")
plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p2)

tmp, orig_svd_f, tmp1 = linalg.svd(Exact_full.T, full_matrices=True)
tmp, pinn_svd_f, tmp1 = linalg.svd(U_pred_full.T, full_matrices=True)
tmp, noisy_svd_f, tmp1 = linalg.svd(U_noisy_full.T, full_matrices=True)
tmp, filter_svd_f, tmp1 = linalg.svd(filter_u.T, full_matrices=True)

fig3 = plt.figure(figsize=(6, 4))
plt.semilogy(orig_svd_f[0:20,], label="Original")
plt.semilogy(pinn_svd_f[0:20,], label="Denoised")
plt.semilogy(noisy_svd_f[0:20,], label="Noisy")
plt.semilogy(filter_svd_f[0:20,], label="Filtered")

plt.legend(loc="best", frameon=False, fontsize=20)
plt.xlabel("Modes", fontsize=20)
plt.ylabel("Singular values", fontsize=20)
plt.xticks([1, 5, 10, 15, 20], fontsize=20)

plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(pltname_modes, dpi=300)
