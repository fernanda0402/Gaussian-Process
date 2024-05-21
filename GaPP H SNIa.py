#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:33:39 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True



# BAIXANDO O ARQUIVO DE DL

data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/dl_recon.csv', delimiter=',')

z = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]


# BAIXANDO A PRIMEIRA DERIVADA DE DL

data_d = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/ddl_rec.csv', delimiter=',')

ddl = data_d[:, 1]
ddlerr = data_d[:, 2]

# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc


# definindo H
h = (((1+z)**2)) / ((1+z)*ddl - dl)
h_1 = ((1+z)**3)
h_2 = ((1+z)*ddl - dl)**2
herr = np.sqrt( ((h_1/h_2)*ddlerr)**2 + (( (((1+z)**2))/h_2 )*dlerr)**2 )


# salvando os dados

H = z, h, herr

#np.savetxt('h_snia_rec.csv', np.transpose(H), delimiter=', ')



# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 2.2, 1000)

a = 1. / (1. + zlcdm)

h_lcdm = h0*ccl.background.h_over_h0(cosmo, a)


# plote
plt.figure()
plt.ylim(0, 500)
plt.plot(z, h, color='green', label='Prediction', linestyle="--")
plt.plot(zlcdm, h_lcdm, color='red', label='$\Lambda$CDM')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([h - 1.0000 * herr,
                        (h + 1.0000 * herr)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None', label='$68\%$ confidence interval')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([h - 1.9600 * herr,
                        (h + 1.9600 * herr)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label='$95\%$ confidence interval')

# legenda, label e título
plt.legend(loc='best')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$H(z)$', fontsize=16)
plt.title('$H(z)$ from SNIa')
#plt.savefig('h_snia_rec.pdf', format='pdf', bbox_inches='tight')
plt.show()







