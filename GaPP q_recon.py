#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:00:34 2023

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np


# BAIXANDO O ARQUIVO DOS DADOS DE DL


data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/dl_recon.csv', delimiter=', ')

z = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]


# BAIXANDO O ARQUIVO DA PRIMEIRA DERIVADA DE DL

data1 = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/ddl_rec2.csv', delimiter=', ')

z = data[:, 0]
d1l = data[:, 1]
d1lerr = data[:, 2]


# BAIXANDO O ARQUIVO DA SEGUNDA DERIVADA DE DL

data2 = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/d2dl_rec2.csv', delimiter=', ')

d2l = data[:, 1]
d2lerr = data[:, 2]



# definindo q(z)

q = ( ( ((1+z)**2)*d2l ) / (dl - (1+z)*d1l) ) + 1

q1 = np.gradient(q, dl)
q2 = np.gradient(q, d1l)
q3 = np.gradient(q, d2l)
sig_q = np.sqrt( (q1**2) * (dlerr**2) + (q2**2) * (d1lerr**2) + (q3**2) * (d2lerr**2) )



# salvando os dados
N =  z, q, sig_q

np.savetxt('q_recon.csv', np.transpose(N), delimiter=', ')



# plote
plt.figure()
plt.plot(z, q, color='green', label='Prediction', linestyle="--")
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([q - 1.9600 * sig_q,
                        (q + 1.9600 * sig_q)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$q(z)$', fontsize=16)
plt.legend(loc='best')
#plt.title('Second Derivative GaPP')
plt.savefig('q_recon.pdf', format='pdf', bbox_inches='tight')
plt.show()


















