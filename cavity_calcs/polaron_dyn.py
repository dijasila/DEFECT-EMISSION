import numpy as np
import scipy as sc
import scipy.signal
import qutip as qt
import phonon_correlation_tools as phonon
import matplotlib.pyplot as plt
import ind_tools as ind
import pandas as pd
import pickle
from cavity_lio import build_meq


hbarc_cmtoev = 1.9746e-05  # eVcm
sd_file = './data.xlsx'
phon_width = 200 * hbarc_cmtoev
SD_sampling = 120000
dt = 0.00001
w_max = (2 * np.pi/dt)*hbarc_cmtoev

kb = 8.617E-5#eV K^-1
Temp_range = np.array([4, 25, 50, 75, 100])
beta_range = 1/(kb * Temp_range)
beta = beta_range[0]





#units are in eV
#define the cavity parameters:
dim=2 #cavity dimension
g=1 #light-matter coupling strength
omega_c=0 #cavity detuning
kappa=0.1 #cavity loss

#and extra parameters
eps=0 #system two level energy
Gam_opt = 0.001 #spontaneous emission rate
driving = 0 #cavity driving, if you are into that kind of thing.

#build and extract the Liouvillian.
lio  = build_meq(dim, eps, g, omega_c, driving, 
kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max, beta)

#define the time interval and initial condition for the ODE solver:
r0 = qt.tensor(qt.create(2) * qt.destroy(2), qt.fock_dm(dim, 0)) 
#initialised in the excited electronic state.
trange = np.linspace(0,50,1000)

#define which expectation values that are required:
expec_list = []
expec_list.append(qt.tensor(qt.create(2) * qt.destroy(2), qt.qeye(dim)) )
expec_list.append(qt.tensor(qt.qeye(dim), qt.create(dim) * qt.destroy(dim)))

#integrate the master equation
sol = qt.mesolve(lio, r0, trange, e_ops=expec_list, progress_bar=True)

plt.plot(trange, sol.expect[0])
plt.plot(trange, sol.expect[1])
plt.show()
