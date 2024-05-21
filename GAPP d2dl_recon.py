#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:15:42 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
from gapp import gp
plt.rcParams['text.usetex'] = True


# primeira derivada

from gapp import dgp



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

np.savetxt('d2dl_rec2.csv', np.transpose(N), delimiter=', ')



# Plot the function, the prediction and the 95% confidence interval
plt.figure()
plt.tick_params(labelsize=14,color='red')
plt.plot(xi, y_pred, color='green', label='Prediction', linestyle="--")
#lt.plot(zlcdm, ddl_lcdm, color='red', label='$\Lambda$CDM')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$d^2 D_L/dz^2$', fontsize=16)
plt.legend(loc='best')
plt.title('Second Derivative GaPP')
plt.savefig('d2dl_rec2.pdf', format='pdf', bbox_inches='tight')
plt.show()




