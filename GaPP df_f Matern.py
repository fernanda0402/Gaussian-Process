#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:51:24 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True

# baixando os dados de f
fz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/fz_recon_gapp_Matern.csv', delimiter=', ')

z = fz[:, 0]
f_z = fz[:, 1]

ef = fz[:, 2]


# baixando os dados da derivada de f

dfz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dfz_recon_gapp_Matern.csv', delimiter=', ')

df_z = dfz[:, 1]
edf = dfz[:, 2]

df_f = df_z / f_z

sigma_ff = np.sqrt(((df_f) ** 2) * (((ef / f_z) ** 2) + ((edf / df_z) ** 2)))



H = z, df_f, sigma_ff
#np.savetxt('df_f_recon_gapp_Matern.dat', np.transpose(H), delimiter='\t')


# MODELO LCDM
cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.4, 1000)

a = 1. / (1. + zlcdm)

flcdm = ccl.background.growth_rate(cosmo, a)

dff_lcdm = np.gradient(flcdm, zlcdm) / flcdm


# PLOTE
plt.plot(z, df_f, color='green', label='Prediction')
plt.plot(zlcdm, dff_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([df_f - 1.0000 * sigma_ff,
                        (df_f + 1.0000 * sigma_ff)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([df_f - 1.9600 * sigma_ff,
                        (df_f + 1.9600 * sigma_ff)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label='$95\%$ confidence interval')

# legenda, label e título
plt.title('GaPP Matern Kernel')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{F}(z)$', fontsize=16)
plt.legend(loc='best')
plt.savefig('dfz_fz_recon_gapp_Matern.pdf', format='pdf', bbox_inches='tight')
plt.show()
