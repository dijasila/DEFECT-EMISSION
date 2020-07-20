import os 
import sys
import json
import pickle
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append("/home/niflheim/markas/TEMPO/defect-emission/tensor_networks/classical_tempo_cavity")

import tempo_class as tempo
import spectral_density_prep as SD




hbarc_cmtoev = 1.9746E-5
hbar_meVps = 6.582E-7
kb = 8.617E-5 # eV K^-1


# Import data and construct the spectral data
sd_file = '/home/niflheim/markas/TEMPO/defect-emission/tensor_networks/classical_tempo_cavity/data.xlsx'
phon_width = 200 * hbarc_cmtoev
SD_sampling = 120000
dt = 0.00001
w_max = (2 * np.pi/dt)*hbarc_cmtoev
# Generating
omegas, spectral_data = SD.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)
J_data = [omegas, spectral_data]


##### Invoke TEMPO to propagate

# Hamiltonian


#Find the reorganisation energy from the spectral density:
reorg = np.trapz(omegas * spectral_data, dx = (omegas[1]-omegas[0]))
print('The reorgansiation energy is: ', reorg )


#define the  desired system raising and lowering operators:
psi_c = np.array([1,0,0])
psi_e = np.array([0,1,0])
psi_g = np.array([0,0,1])

a = np.outer(psi_g, psi_c)
a_dag = np.outer(psi_c, psi_g)
sp = np.outer(psi_e, psi_g)
sm = np.outer(psi_g, psi_e)

eps = 0
wcav = 0

# Setting the propagation stuff
params = json.load(open(os.getcwd() + "/params.json", "r"))
g = params["g"]
kpoints = params["mem_cut"]
kappa = params["kappa"]



# Hamiltonian
Hsys = (eps + reorg) * np.outer(psi_e, psi_e) 
Hsys += wcav*np.outer(psi_c, psi_c) # Cavity mode Hamiltonian
Hsys += g * (np.outer(psi_e, psi_c) + np.outer(psi_c, psi_e)) # Additing coupling to light


# Coupling to environment via system operator e><e
coupling_operator = np.outer(psi_e, psi_e)

sys = tempo.tempo_prop(hilbert_dim = 3, coupop = coupling_operator)







sys.kpoints = 100
sys.dt = 0.01
sys.dkmax = kpoints


# Adding the bath
sys.construct_bath_data(J_data, norm = "chemistry")





sys.system_Ham = Hsys # Uncomment if there is no dissipation
#include spontaneous emission:
rate_and_op = [[0.01, sm], [kappa, a]]
sys.add_dissipation(rate_and_op)

#define initial condition
r0 = np.outer(psi_e, psi_e) # Staring in the excited state
sys.initial =  r0.reshape(sys.d)
sys.precision = 10**(-0.1 * 60)

sys.prep()
sys.prop(10000)

sys.get_expect([np.outer(psi_e, psi_e), np.outer(psi_c, psi_c)])


pickle.dump(sys, open("results.pckl", "wb"))


"""
fig, ax = plt.subplots(ncols=2)
ax[0].plot(sys.trange, sys.expect_data[0], '.')


ax[1].plot(sys.trange, sys.expect_data[0], '.')


for a in ax:
    a.set_xlim(0,sys.trange[-1])
    a.set_ylim(0,1)


#plt.plot(sys.trange, np.exp(reorg) * corr_func.real)
plt.show()
"""

