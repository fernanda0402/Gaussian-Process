#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:02:55 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
#import scipy.interpolate as interp
from scipy.interpolate import UnivariateSpline
from gapp import gp, covariance


plt.rcParams['text.usetex'] = True


import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)




# BAIXANDO OS DADOS DE DL

dl_recon = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/luminosity_distance/codes/dl_recon.dat', delimiter='\t')

z = dl_recon[:, 0]

dl = dl_recon[:, 1]

edl = dl_recon[:, 2]



# BAIXANDO OS DADOS DA DERIVADA DE DL

ddl_recon = np.genfromtxt('/home/usuario/Documentos/Eta study/eta_study/luminosity_distance/codes/ddl_recon.dat', delimiter='\t')

#z = dl_recon[:, 0]

ddl = ddl_recon[:, 1]

eddl = ddl_recon[:, 2]



# DEFININDO H(z)

c = 3 * (10 ** 5)

h1 = (1+z)*ddl - dl

h = c * ((1+z)**2) / h1


f1 = c*(1+z)**2 / h1**2 # dh/dDl

f2 = -c*(1+z)**3 / h1**2  # dh/dDl'


eh = np.sqrt( ( (edl**2) * (f1**2) ) + ( (eddl**2) * (f2**2) ) )


#h_mc = []
#for i in range(5000):
                         
#    dli = np.random.normal(dl, edl)
 #   ddli = np.random.normal(ddl, eddl)
 #  
 #   h_mc.append( c * ((1+z)**2) / ( (1+z)*ddli - dli ) )

#h_mc = np.array(h_mc)   


#sigma_h = []
#for i in range(len(z)):
    
 #   hi = h_mc[:, i]
  #  sigma_h.append(np.std(hi))

#sigma_h = np.array(sigma_h) 


# BUSCANDO SUAVIZAR O ERRO DE H
#erro_h = scipy.interpolate.UnivariateSpline(z[z<1.5] , sigma_h[z<1.5], w=None, bbox=[None, None], k=3, s=None, ext=0, check_finite=False)


# MODELO LCDM

zlcdm = np.linspace(0.001, max(z), 1000)

a = 1. / (1. + zlcdm)

h0 = 67.27

h_lcdm = h0*ccl.background.h_over_h0(cosmo, a)



# PLOTE

fig, ax = plt.subplots()
plt.plot(z, h, color='red', label='Prediction')
plt.plot(zlcdm, h_lcdm, color='black', label='$\Lambda$CDM')

plt.fill_between(z, h - eh, h + eh, alpha=0.4, fc='red', ec='None')
plt.fill_between(z, h - 1.96*eh, h + 1.96*eh, alpha=0.2, fc='red', ec='None')



# legenda e eixos

plt.ylim(0,300)
plt.xlim(min(z), max(z))
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$H(z)^L$', fontsize=16)
plt.legend(loc='best')

#plt.errorbar(z, h, fmt='s', color='blue')


#plt.savefig('h_dl_recon.pdf', format='pdf', bbox_inches='tight')    




