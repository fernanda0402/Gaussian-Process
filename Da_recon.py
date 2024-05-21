#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:19:58 2024

@author: felipe_avila
"""

import matplotlib.pyplot as plt
import numpy as np

from gapp import gp, covariance

plt.rcParams['text.usetex'] = True

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)


# BAIXANDO OS DADOS BINADOS

data = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/angular_diameter/codes/angular_diameter_data.csv', delimiter=', ')

z = data[:, 0]

da = data[:, 1]

eda = data[:, 2]

plt.xlim(min(z), max(z))
plt.errorbar(z, da, eda, fmt='s', color='blue', label='Data')


############################################################################### TESTANDO UMA FUNÇÃO MÉDIA
def T(x):
        
    # return ccl.background.angular_diameter_distance(cosmo, 1/(1+x))

    c = 3 * (10 ** 5)
    H0 = 70
    
    t1 = (c / H0)
    
    return (t1 / (1+x)) * np.log(1+x)
    
    
    # c  = 3. * (10 ** 5)
    # H0 = 70
    
    # t1 = (2.*c) / H0
    
    # return (t1 / (1+x)) * (1. - (1. / np.sqrt(1.+x))) 

# from scipy.optimize import curve_fit

# popt, pcov = curve_fit(T, z, da, sigma=eda,p0=(0.3, 0.7),
#                        bounds=([0, 0], [2, 2]), method='trf')

# Omc, h = popt


# nomeando
x_gapp = z
y_gapp = da
e = eda

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = min(z)
xmax = max(z)
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
# initheta = [520, 2.0]

# initialization of the Gaussian Process

g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar),
                        mu=None)

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp()

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]


# salvando os dados

Q = xi, y_pred, sigma

#np.savetxt('da_recon.dat', np.transpose(Q), delimiter='\t')


plt.plot(xi, y_pred, color='red', label='Prediction')
plt.fill_between(xi, y_pred-sigma, y_pred+sigma, alpha=0.4, color='red')
plt.fill_between(xi, y_pred-1.96*sigma, y_pred+1.96*sigma, alpha=0.2, color='red')


# MODELO LCDM

zi = np.linspace(0, 3, 1000)

Dai = ccl.background.angular_diameter_distance(cosmo, 1/(1+zi))

plt.plot(zi, Dai, color='black', label='$\Lambda$CDM')


# legenda e eixos

plt.legend(loc='best')
plt.xlabel('$z$')
plt.ylabel('$D_A$')


#plt.plot(zi, T(zi), color='black', ls='dashed')





