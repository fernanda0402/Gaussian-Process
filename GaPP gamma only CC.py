#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:37:43 2023

@author: usuario
"""

# bibliotecas
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

plt.rcParams['text.usetex'] = True

# baixando os dados

data_dfs8_fs8 = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dfs8_fs8_recon_gapp.csv', delimiter='\t')

dfs8_fs8 = data_dfs8_fs8[:, 1]
sigma_ffs8 = data_dfs8_fs8[:, 2]


##################################################################################

# baixando os dados
data_Hrec = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/Hz_recon_gapp_h0.csv', delimiter=', ')

z = data_Hrec[:, 0]
H = data_Hrec[:, 1]
eH = data_Hrec[:, 2]


# baixando os dados
data_dHrec = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP CC/dHz_recon_gapp_h0.csv', delimiter=', ')

dH = data_dHrec[:, 1]

edH = data_dHrec[:, 2]


dh_h = dH / H

sigma_H = np.sqrt(((dh_h) ** 2) * (((eH / H) ** 2) + ((edH / dH) ** 2)))


# definindo E(z)
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

#k_plus = (1. + z) / ( (Ez + eEz) ** 3)

#I_plus = cumtrapz(k_plus, x=z, initial=0)

#k_minus = (1. + z) / ( (Ez - eEz) ** 3)

#I_minus = cumtrapz(k_minus, x=z, initial=0)

#sig_I = (I_plus - I_minus) / 2


d1 = 3*(1+z)*eEz / ( (Ez**4)*(1-Int) )

d2 = (1+z)*sig_I / ( (Ez**3)*(1-Int) )

edD_D = np.sqrt( (sigma_H)**2 + ( d1 )**2 + ( d2 )**2 )



# definindo Om'/Om
dOm_Om = (3/(1+z)) - (2*dh_h)
eOm = np.sqrt((2. * (edH / H)) ** 2 + ((2. * dH) / ((1. + z) ** 2 * H) ** 2 * ez ** 2))



# gamma
gamma_rec = (0 - dD_D) / dOm_Om

sigma_g = np.sqrt((0 ** 2 + edD_D ** 2) / dOm_Om ** 2 + (0 - dD_D) ** 2 * eOm ** 2 / dOm_Om ** 4)


# salvando os dados

N = z, gamma_rec, sigma_g

#np.savetxt('gamma_recon_onlyCC_h0.csv', np.transpose(N), delimiter=', ')




# gamma teórico

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649)
    
zlcdm = np.linspace(0.001, 1.4, 1000)

a = 1. / (1. + zlcdm)

D_lcdm = ccl.growth_factor(cosmo, a)

dD_D_lcdm = np.gradient(D_lcdm, zlcdm) / D_lcdm

Om_lcdm = ccl.background.omega_x(cosmo, a, 'matter')

dO_lcdm = np.gradient(Om_lcdm, zlcdm) / Om_lcdm

gamma_teo = - dD_D_lcdm / dO_lcdm



# plote
fig, ax = plt.subplots()
plt.ylim(-2,2)
plt.xlim(0,1.0)
plt.tick_params(labelsize=14, color='red')
plt.plot(z, gamma_rec, color='darkgreen', label='GP Prediction', linestyle="--")
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([gamma_rec - 1.0000 * sigma_g,
                        (gamma_rec + 1.0000 * sigma_g)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([gamma_rec - 1.9600 * sigma_g,
                        (gamma_rec + 1.9600 * sigma_g)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')
plt.plot(zlcdm, gamma_teo, color='purple', label='$\mathcal{J}(z)$ theoretical')
plt.axhline(y=0.55, color='red', linestyle='-', linewidth=1, label='0.55')


# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{J}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('gamma_nofs8_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()


print(gamma_rec[0])
print(sigma_g[0])

