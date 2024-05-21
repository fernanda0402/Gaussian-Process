#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:01:41 2024

@author: felipe
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True



 # BAIXANDO OS DADOS DE DA

da_recon = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/angular_diameter/codes/da_recon.dat', delimiter='\t')

z = da_recon[:, 0]

da = da_recon[:, 1]

eda = da_recon[:, 2]



# BAIXANDO OS DADOS DE DL

dl_recon = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/luminosity_distance/codes/dl_recon.dat', delimiter='\t')

#z = dl_recon[:, 0]

dl = dl_recon[:, 1]

edl = dl_recon[:, 2]


# DEFININDO ETA

f = dl / da

ef = np.sqrt(((f/dl) * edl) ** 2 + (-(f/da) * eda)**2) 

eta = (((1+z)**-2) * f) - 1

sig_eta = np.sqrt((((1+z)**-2) * ef) ** 2)



# PLOTE

plt.plot(z, eta, color='red', label='Prediction')
plt.fill_between(z,eta-sig_eta,eta+sig_eta,alpha=0.4,color='red')
plt.fill_between(z,eta-1.96*sig_eta,eta+1.96*sig_eta,alpha=0.2, color='red')
plt.hlines(0, 0, 3, color='black', ls='dashed', label='CP')


# legenda e eixos

plt.ylim(-1,1)
plt.xlim(min(z), max(z))
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\eta(z)$', fontsize=16)
plt.legend(loc='best')





# baixando os dados de DL e DA


data_dl = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/luminosity_distance/codes/luminosity_data.csv', delimiter=', ')

z = data_dl[:, 0]

dl = data_dl[:, 1]

edl = data_dl[:, 2]


data_da = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/angular_diameter/codes/angular_diameter_data.csv', delimiter=', ')

z = data_da[:, 0]

da = data_da[:, 1]

eda = data_da[:, 2]



# DEFININDO ETA

g = dl / da

eg = np.sqrt(((g/dl) * edl) ** 2 + (-(g/da) * eda)**2) 

eta = (((1+z)**-2) * g) - 1

sig_eta = np.sqrt((((1+z)**-2) * eg) ** 2)


plt.errorbar(z, eta, sig_eta, fmt='s', color='blue')

#plt.savefig('eta_recon.pdf', format='pdf', bbox_inches='tight')    








