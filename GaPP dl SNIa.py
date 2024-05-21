#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:34:52 2023

@author: usuario
"""



import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
from gapp import gp
plt.rcParams['text.usetex'] = True

# 1) BAIXANDO O ARQUIVO 

data = np.genfromtxt('/home/usuario/Documentos/Dados/dlc_snia.dat', delimiter='\t')

zCMB = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc


######## PROCESSO GAUSSIANO USANDO GAPP #################

# nomeando
x_gapp = zCMB
y_gapp = dl
e = dlerr


# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = min(x_gapp)
xmax = max(x_gapp)
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



# salvando os dados

N =  xi, y_pred, sigma

np.savetxt('dl_recon.csv', np.transpose(N), delimiter=', ')


# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 2, 1000)

a = 1. / (1. + zlcdm)

dl_lcdm = ccl.background.luminosity_distance(cosmo, a) / c 




# Plot the function, the prediction and the 95% confidence interval
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.errorbar(x_gapp, y_gapp, e, fmt='r.', color='purple', markersize=10, label='Data')
plt.plot(zlcdm, dl_lcdm, color='red', label='$\Lambda$CDM')
plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="--")
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$D_L(z)$', fontsize=15)
plt.legend(loc='best')
plt.title('GaPP')
#plt.savefig('dl_recon.pdf', format='pdf', bbox_inches='tight')
plt.show()


