#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:10:17 2024

@author: felipe
"""

import matplotlib.pyplot as plt
import numpy as np

from gapp import dgp

plt.rcParams['text.usetex'] = True

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)


# BAIXANDO OS DADOS BINADOS

data = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/luminosity_distance/codes/luminosity_data.csv', delimiter=', ')

z = data[:, 0]

dl = data[:, 1]

edl = data[:, 2]


# plt.xlim(min(z), max(z))
# plt.errorbar(z, dl, edl, fmt='s', color='blue')


############################################################################### TESTANDO UMA FUNÇÃO MÉDIA
def T(x):
    
    
    c  = 3. * (10 ** 5)
    H0 = 70
    
    t1 = (2.*c) / H0
    
    return (t1 * (1+x)) * (1. - (1. / np.sqrt(1.+x))) 

def dT(x):
    
    c  = 3. * (10 ** 5)
    H0 = 70
    
    t1 = (2.*c) / H0
    
    return (T(x)/(1+x)) + ((t1 * (1+x)) * 0.5 * ((1+x)**(-1.5)))
    
    



# nomeando
x_gapp = z
y_gapp = dl
e = edl

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = min(z)
xmax = max(z)
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
# initheta = [520, 2.0]

# initialization of the Gaussian Process

# g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar),
#                         mu=T, dmu=dT)

g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar),
                        mu=None)

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.dgp()

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]


# salvando os dados

Q = xi, y_pred, sigma

#np.savetxt('ddl_recon.dat', np.transpose(Q), delimiter='\t')




plt.plot(xi, y_pred, color='red', label='Prediction')
plt.fill_between(xi, y_pred-sigma, y_pred+sigma, alpha=0.4, color='red')
plt.fill_between(xi, y_pred-1.96*sigma, y_pred+1.96*sigma, alpha=0.2, color='red')


# MODELO LCDM

zi = np.linspace(min(xi), max(xi), 1000)

Dli = ccl.background.luminosity_distance(cosmo, 1/(1+zi))

plt.plot(zi, np.gradient(Dli, zi), color='black', label='$\Lambda$CDM')




# legenda e eixos

plt.legend(loc='best')
plt.xlabel('$z$')
plt.ylabel('$dD_L/dz$')




# plt.plot(zi, T(zi), color='black', ls='dashed')


