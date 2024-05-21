#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:35:21 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True


from gapp import dgp, covariance


# 1) BAIXANDO O ARQUIVO 


data = np.genfromtxt('/home/usuario/Documentos/Dados/dlc_snia.dat', delimiter='\t')

zCMB = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc



################ PROCESSO GAUSSIANO GAPP ##################


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
g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))


# training of the hyperparameters and reconstruction of the function
(d2rec, theta) = g.d2gp(theta=initheta)


xi     = d2rec[:, 0]
y_pred = d2rec[:, 1]
sigma  = d2rec[:, 2]


# salvando os dados
N =  xi, y_pred, sigma

#np.savetxt('d2dl_rec.csv', np.transpose(N), delimiter=', ')



# MODELO LCDM
cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 2.2, 1000)

a = 1. / (1. + zlcdm)

dl_lcdm = ccl.background.luminosity_distance(cosmo, a) / c

ddl_lcdm = np.gradient(dl_lcdm, a)

d2dl_lcdm = np.gradient(ddl_lcdm, a)



# Plot the function, the prediction and the 95% confidence interval
plt.figure()
plt.tick_params(labelsize=14,color='red')
plt.plot(xi, y_pred, color='green', label='Prediction', linestyle="--")
plt.plot(zlcdm, d2dl_lcdm, color='red', label='$\Lambda$CDM')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('dd$D_L$/dz', fontsize=16)
plt.legend(loc='best')
plt.title('Second Derivative GaPP')
#plt.savefig('d2dl_rec.pdf', format='pdf', bbox_inches='tight')
plt.show()


















