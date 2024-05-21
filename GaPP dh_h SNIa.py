#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:13:16 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc



# BAIXANDO OS DADOS DE Q

data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/q_recon.csv', delimiter=', ')

z = data[:, 0]
q = data[:, 1]
sig_q = data[:, 2]


dh_h = (q+1) / (1+z)

sig_hh = np.gradient(dh_h, q)*sig_q



# MODELO LCDM


cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 2.2, 1000)

a = 1. / (1. + zlcdm)

h_lcdm = ccl.background.h_over_h0(cosmo, a)

dh_hlcdm = np.gradient(h_lcdm, a) / h_lcdm


# plote
plt.figure()
plt.ylim(-1, 1)
plt.plot(z, dh_h, color='green', label='Prediction', linestyle="--")
plt.plot(zlcdm, h_lcdm, color='red', label='$\Lambda$CDM')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dh_h - 1.0000 * sig_hh,
                        (dh_h + 1.0000 * sig_hh)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None', label='$68\%$ confidence interval')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dh_h - 1.9600 * sig_hh,
                        (dh_h + 1.9600 * sig_hh)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label='$95\%$ confidence interval')

# legenda, label e título
plt.legend(loc='best')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{H}(z)$', fontsize=16)
plt.title('$H(z)$ from SNIa')
#plt.savefig('h_snia_rec.pdf', format='pdf', bbox_inches='tight')
plt.show()

















