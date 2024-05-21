#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:35:03 2024

@author: felipe_avila
"""

import matplotlib.pyplot as plt
import numpy as np

from gapp import dgp

plt.rcParams['text.usetex'] = True

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)


# BAIXANDO OS DADOS

data = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/angular_diameter/codes/angular_diameter_data.csv', delimiter=', ')

z = data[:, 0]

da = data[:, 1]

eda = data[:, 2]


############################################################################### TESTANDO UMA FUNÇÃO MÉDIA
def T(x):
    
    
    # cosmo_T = ccl.Cosmology(
    #     Omega_c=0.2506, Omega_b=0.0494, 
    #     h=0.70, sigma8=0.8120, n_s=0.9649)
        
    # return ccl.background.angular_diameter_distance(cosmo_T, 1/(1+x))

    c = 3 * (10 ** 5)
    H0 = 70
    
    t1 = (c / H0)
    
    return (t1 / (1+x)) * np.log(1+x)
    
    
    # c  = 3. * (10 ** 5)
    # H0 = 70
    
    # t1 = (2.*c) / H0
    
    # return (t1 / (1+x)) * (1. - (1. / np.sqrt(1.+x))) 
    
    

# nomeando
x_gapp = z
y_gapp = da
e = eda

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = min(z)
xmax = max(z)
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
#initheta = [2.0, 2.0]

# initialization of the Gaussian Process

g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar),
                        mu=None)

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.dgp()

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]


# salvando os dados

Q = xi, y_pred, sigma

#np.savetxt('dda_recon.dat', np.transpose(Q), delimiter='\t')



plt.plot(xi, y_pred, color='red', label='Prediction')
plt.fill_between(xi, y_pred-sigma, y_pred+sigma, alpha=0.4, color='red')
plt.fill_between(xi, y_pred-1.96*sigma, y_pred+1.96*sigma, alpha=0.2, color='red')

plt.axhline(y=0, color='blue', linestyle='-', linewidth=1)


# MODELO LCDM

zi = np.linspace(min(xi), max(xi), 1000)

Dai = ccl.background.angular_diameter_distance(cosmo, 1/(1+zi))

dDai = np.gradient(Dai, zi)

plt.xlim(min(z), max(z))
plt.plot(zi, dDai, color='black', label='$\Lambda$CDM')


# legenda e eixos

plt.legend(loc='best')
plt.xlabel('$z$')
plt.ylabel('$dD_A/dz$')


#plt.hlines(0, 0, 3, color='black', ls='dashed')

#plt.savefig('dda_teste_recon.pdf', format='pdf', bbox_inches='tight') 


