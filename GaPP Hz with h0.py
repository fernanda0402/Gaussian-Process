#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:46:54 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True
from gapp import gp

plt.rcParams['text.usetex'] = True


# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data (cópia).csv', delimiter=', ')

z = data_Hz[:, 0]
H = data_Hz[:, 1]

sig_H = data_Hz[:, 2]


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 67.27 #km/s/Mpc



##################### PROCESSO GAUSSIANO GAPP ###########################

# nomeando
x_gapp = z
y_gapp = H
e = sig_H

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


# salvando os dados reconstruídos

F = xi, y_pred, sigma
#np.savetxt('Hz_recon_gapp_h0.csv', np.transpose(F), delimiter=', ')



# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

h_lcdm = h0*ccl.background.h_over_h0(cosmo, a)



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.ylim(0, 180)
plt.tick_params(labelsize=14, color='purple')
plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='purple', markersize=10, label='Data')
plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="-")
plt.plot(zlcdm, h_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                        (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'forestgreen', ec='None')

# legenda, label e título
plt.ylim(50,150)
plt.xlim(0,1.0)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$H(z)$', fontsize=15)
plt.legend(loc='best')
#plt.savefig('Hz_with_h0_recon_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()