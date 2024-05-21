#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:35:30 2023

@author: usuario
"""

import numpy as np
from matplotlib import pyplot as plt
import pyccl as ccl
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
plt.rcParams['text.usetex'] = True


# dados de gamma
data_g = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/gamma_recon_fs8_h0.csv', delimiter=', ')

z = data_g[:,0]
gamma = data_g[:,1]
egz = data_g[:,2]


# constantes
Om0 = 0.3

Om = (Om0 * (1+z)**3) / (Om0 * ((1+z)**3) + 1 - Om0)

f = ( Om )**gamma

#sig_f = (np.gradient(f, gamma)) * egz

sig_f = (f * np.log(Om))*egz


# curva do modelo teórico
cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649)
    
z_lcdm = np.linspace(0, 1.4, 200)

a = 1/(1+z_lcdm)
    
f_lcdm = ccl.background.growth_rate(cosmo, a)




# plote
plt.ylim(0,1.5)
plt.tick_params(labelsize=14,color='red')
plt.plot(z, f, 'green', label='Prediction', ls='-.')
plt.plot(z_lcdm, f_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([f - 1.9600 * sig_f,
                        (f + 1.9600 * sig_f)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label=r'$95\%$ confidence interval')

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$f(z)$', fontsize=16)
plt.legend(loc=2, fontsize=12)
plt.title('GaPP Cosmic Chronometers')
#plt.savefig('fz_reverso.pdf', format='pdf', bbox_inches='tight')
plt.show()


