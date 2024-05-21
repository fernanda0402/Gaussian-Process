#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:26:56 2023

@author: usuario
"""

# bibliotecas
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

plt.rcParams['text.usetex'] = True


# baixando os dados de H
Hz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/Hz_recon_gapp_Matern.csv', delimiter=', ')

z = Hz[:, 0]
H = Hz[:, 1]

eH = Hz[:, 2]


# baixando os dados da derivada de H

dHz = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dHz_recon_gapp_Matern.csv', delimiter=', ')

dHz = dHz[:, 1]
edH = dHz[2]


dh_h = dHz/ H  # H'/H

sigma_H = np.sqrt(((dh_h) ** 2) * (((eH / H) ** 2) + ((edH / dHz) ** 2)))



# definindo E(z)
h0 = 70
Ez = H / H[0]
eEz = eH / H[0]


# integral 
k = (1. + z) / (Ez ** 3)
ez = 0
ek = ((1. + z) / (Ez ** 3)) * np.sqrt((eEz / Ez) ** 2 + (ez / (1. + z)) ** 2)

Int = cumtrapz(k, x=z, initial=0)

I = 1. - Int

T = k/I


# contraste
dD_D = (dh_h) - (k/(1-Int))

#edD_D = np.sqrt((edH / H) ** 2 + ((dHz / H) * eH / H) ** 2 + (ek / (1. - I)) ** 2)

sig_I = np.sqrt( (np.gradient(Int, Ez)*eEz)**2 )

d1 = 3*(1+z)*eEz / ( (Ez**4)*(1-Int) )

d2 = (1+z)*sig_I / ( (Ez**3)*(1-Int) )

edD_D = np.sqrt( (sigma_H)**2 + ( d1 )**2 + ( d2 )**2 )


# salvando os dados

H = z, dD_D, edD_D
np.savetxt('dD_D_recon_gapp_Matern.csv', np.transpose(H), delimiter=', ')


# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649)
    
x = np.linspace(0, 1.4, 1000)

a = 1/(1+x)

D_lcdm = ccl.growth_factor(cosmo, a)

dD_D_lcdm = np.gradient(D_lcdm, x) / D_lcdm



# plote
plt.plot(z, dD_D, c='green', label='Prediction')
plt.plot(x, dD_D_lcdm, color='red', label='$\Lambda$CDM')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dD_D - 1.0000 * edD_D,
                        (dD_D + 1.0000 * edD_D)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dD_D - 1.9600 * edD_D,
                        (dD_D + 1.9600 * edD_D)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')

plt.ylim(-2.5, 1.5)
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{D}(z)$', fontsize=16)
plt.title('GaPP Cosmic Chronometers - Matern Kernel')
plt.legend(loc='best', fontsize=12)
plt.savefig('dD_D_recon_gapp_Matern.pdf', format='pdf', bbox_inches='tight')
plt.show()


