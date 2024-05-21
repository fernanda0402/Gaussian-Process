#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:50:44 2023

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

#sigma_H = np.sqrt(((dh_h) ** 2) * (((eH / H) ** 2) + ((edH / dHz) ** 2)))

sigma_H = np.sqrt( (dHz*eH / (H**2) )**2 + (edH / H)**2 )


# definindo constantes
c = 2.9*10**5 # velocidade da luz
h0 = 70 #km/s/Mpc



# salvando os dados

H = z, dh_h, sigma_H
#np.savetxt('dh_h_recon_gapp_h0.dat', np.transpose(H), delimiter='\t')




# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

h_lcdm = h0*ccl.background.h_over_h0(cosmo, a)

dh_lcdm = np.gradient(h_lcdm, zlcdm) / h_lcdm




# PLOTE
plt.plot(z, dh_h, color='green', label='Prediction', linestyle="--")
plt.plot(zlcdm, dh_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dh_h - 1.0000 * sigma_H,
                        (dh_h + 1.0000 * sigma_H)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dh_h - 1.96 * sigma_H,
                        (dh_h + 1.96 * sigma_H)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None', label=' confidence interval')

# legenda, label e título
plt.ylim(-1, 2)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{H}(z)$', fontsize=16)
plt.legend(loc='best')
plt.title('GaPP Cosmic Chronometers')
plt.savefig('dh_h_recon_gapp_h0.pdf', format='pdf', bbox_inches='tight')
plt.show()


