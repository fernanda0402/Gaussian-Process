#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:33:36 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True
from gapp import gp

# primeira derivada

from gapp import dgp


# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data.csv', delimiter=', ')

z = data_Hz[:, 0]
H = data_Hz[:, 1]

sig_H = data_Hz[:, 2]


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc


########################## PROCESSO GAUSSIANO GAPP ##################################


# nomeando
x_gapp = z
y_gapp = H
e = sig_H

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.4
nstar = 200


# initial values of the hyperparameters
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(drec, theta) = g.dgp(thetatrain=initheta)

# the second and third derivatives use g.d2gp() and g.d3gp()

xi     = drec[:, 0]
y_pred = drec[:, 1]
sigma  = drec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


# salvando os dados reconstruídos

dF = xi, y_pred, sigma
#np.savetxt('dHz_recon_gapp.csv', np.transpose(dF), delimiter=', ')



# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.4, 1000)

a = 1. / (1. + zlcdm)

h_lcdm = h0*ccl.background.h_over_h0(cosmo, a)

dh_lcdm = np.gradient(h_lcdm, zlcdm)



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="--")
plt.plot(zlcdm, dh_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None', label=r'$95\%$ confidence interval')

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$dH(z)/dz$', fontsize=15)
plt.legend(loc='best')
plt.title('GaPP Cosmic Chronometers')
#plt.savefig('dHz_recon_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()





