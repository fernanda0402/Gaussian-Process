#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:52:22 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
plt.rcParams['text.usetex'] = True



# baixando os dados de H
Hz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/Hz_recon_gapp_h0.csv', delimiter=', ')

z = Hz[:, 0]
H = Hz[:, 1]

eH = Hz[:, 2]


# baixando os dados da derivada de H

dHz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dHz_recon_gapp_h0.csv', delimiter=', ')

dHz = dHz[:, 1]
edH = dHz[2]


dh_h = dHz/ H  # H'/H

sigma_H = np.sqrt(((dh_h) ** 2) * (((eH / H) ** 2) + ((edH / dHz) ** 2)))



# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc


# definindo Om'/Om
dOm_Om = (3/(1+z)) - (2*dh_h)
eOm = np.sqrt( (2*sigma_H)**2 )


# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.4, 1000)

a = 1. / (1. + zlcdm)

Om_lcdm = ccl.background.omega_x(cosmo, a, 'matter')

dO_lcdm = np.gradient(Om_lcdm, zlcdm) / Om_lcdm



# plote
plt.plot(z, dOm_Om, color='purple', label='Prediction')
plt.plot(zlcdm, dO_lcdm, color='red', label='$\Lambda$CDM')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{O}(z)$', fontsize=16)
plt.legend(loc='best')
plt.title('GaPP Cosmic Chronometers')
#plt.savefig('dO_O_recon_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()


# baixando os dados de f
fz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fz_recon_gapp.csv', delimiter=', ')

z = fz[:, 0]
f_z = fz[:, 1]

ef = fz[:, 2]


# baixando os dados da derivada de f

dfz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/dfz_recon_gapp.csv', delimiter=', ')

df_z = dfz[:, 1]
edf = dfz[:, 2]



df_f = df_z / f_z

sigma_ff = np.sqrt(((df_f) ** 2) * (((ef / f_z) ** 2) + ((edf / df_z) ** 2)))


# DEFININDO GAMMA

gamma_rec = df_f / dOm_Om

sigma_g = np.sqrt( (sigma_ff / dOm_Om )**2 + ( ( (df_f*eOm)/ ((dOm_Om)**2) ) )**2)



# salvando os dados

N = z, gamma_rec, sigma_g

#np.savetxt('gamma_recon_h0.csv', np.transpose(N), delimiter=', ')



# plote
fig, ax = plt.subplots()
plt.ylim(-2,2)
plt.xlim(0,1.0)
plt.tick_params(labelsize=14, color='red')
plt.plot(z, gamma_rec, color='darkgreen', label='GP #Prediction', linestyle="--")
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([gamma_rec - 1.0000 * sigma_g,
                        (gamma_rec + 1.0000 * sigma_g)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([gamma_rec - 1.9600 * sigma_g,
                        (gamma_rec + 1.9600 * sigma_g)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')
plt.axhline(y=0.55, color='red', linestyle='-', linewidth=1, label='$\Lambda$CDM')


# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\gamma$', fontsize=16)
plt.legend(loc='best')
plt.savefig('gamma_CC.pdf', format='pdf', bbox_inches='tight')
plt.show()



print(gamma_rec[0])
print(sigma_g[0])

