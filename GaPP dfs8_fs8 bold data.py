#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:15:04 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True

# baixando os dados de f
fz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/fs8_recon_bold_gapp.csv', delimiter=', ')

z = fz[:, 0]
fs8 = fz[:, 1]

efs8 = fz[:, 2]


# baixando os dados da derivada de f

dfz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dfs8_recon_bold_gapp.csv', delimiter=', ')

dfs8 = dfz[:, 1]
edfs8 = dfz[:, 2]


df_fs8 = dfs8 / fs8

#sigma_ffs8 = np.sqrt(((df_fs8) ** 2) * (((efs8 / fs8) ** 2) + ((edfs8 / dfs8) ** 2)))

sigma_ffs8 = np.sqrt( (dfs8*efs8 / (fs8)**2)**2 + (edfs8 / fs8)**2 )


H = z, df_fs8, sigma_ffs8
np.savetxt('dfs8_fs8_recon_bolddata_gapp.csv', np.transpose(H), delimiter='\t')



# MODELO LCDM
cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

fs8_lcdm = ccl.growth_rate(cosmo, a)*0.812*ccl.growth_factor(cosmo, a)

dffs8_lcdm = np.gradient(fs8_lcdm, zlcdm) / fs8_lcdm




# PLOTE
plt.plot(z, df_fs8, color='green', label='Prediction')
plt.plot(zlcdm, dffs8_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([df_fs8 - 1.0000 * sigma_ffs8,
                        (df_fs8 + 1.0000 * sigma_ffs8)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([df_fs8 - 1.9600 * sigma_ffs8,
                        (df_fs8 + 1.9600 * sigma_ffs8)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label='$95\%$ confidence interval')

# legenda, label e título
plt.title('GaPP')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{S}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('dfs8_fs8_recon_bolddata_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()

