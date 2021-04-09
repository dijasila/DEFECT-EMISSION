import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import spectral_density_prep as SD
from tensornetwork.tempo.deg_tempo_class import tempo_prop
import indiboson.ibm as ibm 
from scipy.interpolate import interp1d
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#important conversion parameters:
hbarc_cmtoev = 1.9746e-05  # eVcm
hbar_meVps = 6.582E-7
kb = 8.617E-5  # eV K^-1

# import and construct the spectral data for HBN
sd_file = 'data.xlsx'
phon_width = 200 * hbarc_cmtoev
SD_sampling = 12000
dt = 0.00025
w_max = (2 * np.pi/dt)*hbarc_cmtoev

#construct a spectral density from this data:
omegas, spectral_data = SD.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)
J_data = [omegas, (omegas**2)*spectral_data]

#define the  desired system raising and lowering operators:
sp = np.array([[0,1],[0,0]])
sm = sp.T


#Find the reorganisation energy from the spectral density:
reorg = np.trapz(omegas * spectral_data, dx = (omegas[1]-omegas[0]))
print('The reorgansiation energy is: ', reorg )

#set the system  parameters and Hamiltonian:
eps= reorg
V=0
#choose a spontaneous emission rate of 100 ps:
T1 = 1
Gamma = 0.00001
T_range = [10]

fig, ax = plt.subplots(1,2,figsize=(15,6))
for n, T  in enumerate(T_range):
    Temp  = kb * T

    Hs =  np.array([[reorg,V],[V,0]])





    rho0 = 0.5 * np.ones(4)
    kpoints= 100
    prec = 1E-6
    dt = 10


    trange = np.linspace(0,(kpoints+1) * dt,500)
    j_func = interp1d(J_data[0].real, J_data[1].real, kind='cubic')
    indi = ibm.ibm(j_func, T=Temp,cutoff = omegas[-1])
    indi.construct_phi(trange)
    indi.build_decoherence_func(numeric =False)


    tempo = tempo_prop()
    tempo.initial = rho0
    tempo.system_Ham = Hs

    tempo.dt = dt
    tempo.kpoints=kpoints
    tempo.precision = prec
    tempo.free_evolution()
    tempo.construct_bath_data(J_data, Temp=Temp)
    tempo.prep()

    print(tempo.I_dk(0))
    print(tempo.I_dk(1))
    tempo.prop(100)
    data = np.array(tempo.state_list)



    ax[0].plot(J_data[0] * 1E3, J_data[1])
    ax[0].set_xlabel(r'Frequency, $\hbar\omega$ meV')
    ax[0].set_ylabel(r'Spectral density, $J(\omega)$')
    ax[0].set_xlim(0,250)

    # ax[0].plot( trange, np.abs(0.5 * indi.Gamma),'-',lw = 1.75)
    ax[1].plot( trange/hbar_meVps, np.abs(0.5 * indi.Gamma),'-',lw = 1.75, label='IBM', color=colors[n])
    ax[1].plot(np.array(tempo.trange)/hbar_meVps, np.abs(data[:,0,1]),'o', label='TEMPO', color=colors[n], fillstyle='none')

    ax[1].set_ylabel(r'Coherence, $\langle\sigma(t)\rangle$')
    ax[1].set_xlabel(r'Time (ps)')
    ax[1].legend()
plt.show()