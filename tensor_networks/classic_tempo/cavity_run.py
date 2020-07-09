import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import tempo_class as tempo
import spectral_density_prep as SD
from scipy import interpolate
#important conversion parameters:
hbarc_cmtoev = 1.9746e-05  # eVcm
hbar_meVps = 6.582E-7
kb = 8.617E-5  # eV K^-1



# import and construct the spectral data for HBN
sd_file = 'data.xlsx'
phon_width = 200 * hbarc_cmtoev
SD_sampling = 120000
dt = 0.00001
w_max = (2 * np.pi/dt)*hbarc_cmtoev

#construct a spectral density from this data:
omegas, spectral_data = SD.gen_spectral_dens(
    sd_file, phon_width, SD_sampling, w_max)
J_data = [omegas, spectral_data]


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
g = 5
kappa = 25

Hsys = (eps + reorg) * np.outer(psi_e, psi_e) + wcav * np.outer(psi_c, psi_c)
Hsys += g * (np.outer(psi_e, psi_c) + np.outer(psi_c, psi_e))

#coupling operator;
coupop = np.outer(psi_e, psi_e)


# #initialise TEMPO:
sys = tempo.tempo_prop(hilbert_dim=3, coupop = coupop)

#set the number of points and time steps
sys.kpoints = 100
sys.dt = 0.005

# #set a memory cut-off
sys.dkmax = 100

#add the main bath
sys.construct_bath_data(J_data, norm = 'chemistry')
#
# sys.bath.check_sample_rate() #uncomment to check how well the correlation function has been sampled


#define the free Hamiltonian:
sys.system_Ham = Hsys # Uncomment if there is no dissipation
#include spontaneous emission:
rate_and_op = [[nv.GamOpt, sm], [kappa, a]]
sys.add_dissipation(rate_and_op)

#define initial condition
r0 = np.outer(psi_e, psi_e)
sys.initial =  r0.reshape(sys.d)
sys.precision = 10**(-0.1 * 60)

sys.prep()
sys.prop(500)

sys.get_expect([np.outer(psi_e, psi_e), np.outer(psi_c, psi_c)])

fig, ax = plt.subplots(ncols=2)
ax[0].plot(sys.trange, sys.expect_data[0], '.')


ax[1].plot(sys.trange, sys.expect_data[0], '.')


for a in ax:
    a.set_xlim(0,sys.trange[-1])
    a.set_ylim(0,1)


#plt.plot(sys.trange, np.exp(reorg) * corr_func.real)
plt.show()


