import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import tempo_class as tempo
import spectral_density_prep as SD

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
omegas, spectral_data = SD.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)
J_data = [omegas, spectral_data]

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
Gamma = 0.01

Hsys =  eps * sp @ sm + V * (sp + sm)

#initialise TEMPO:
sys = tempo.tempo_prop()

#set the number of points and time steps
sys.kpoints = 350
sys.dt = 3

# #set a memory cut-off 
sys.dkmax = 350

#add the main bath
sys.construct_bath_data(J_data, norm = 'chemistry')
#sys.bath.check_sample_rate() #uncomment to check how well the correlation function has been sampled


#define the free Hamiltonian:
sys.system_Ham = Hsys # Uncomment if there is no dissipation
#include spontaneous emission:
rate_and_op = [[Gamma, sm]]
sys.add_dissipation(rate_and_op)




sys.initial =  0.5 * np.ones(4)
sys.precision = 10**(-0.1 * 60)

sys.prep()
sys.prop(500)


def nocc(w, beta):
    return 1/(np.exp(beta * w) - 1)


def phonon_correlation_function(omegas, spectral_data, beta):

    dw = np.abs(omegas[0] - omegas[1])

    if beta == None:
        spectral_density = np.zeros_like(omegas)
        boo_array = omegas == 0
        spectral_density[boo_array] = 0

        boo_array = omegas > 0
        spectral_density[boo_array] = spectral_data[boo_array]
        # np.conj(np.fft.rfft(spectral_data.real))
        corr_func = np.fft.rfft(spectral_density.real)

    elif beta > 0:
        absorption = np.zeros_like(omegas)
        emission = np.zeros_like(omegas)

        boo_array = omegas == 0
        absorption[boo_array] = 0
        emission[boo_array] = 0

        boo_array = omegas > 0
        absorption[boo_array] = spectral_data[boo_array] * \
            nocc(omegas[boo_array], beta)
        emission[boo_array] = spectral_data[boo_array] * \
            (nocc(omegas[boo_array], beta)+1)

        corr_func = (np.fft.rfft(emission.real))
        corr_func += np.conj(np.fft.rfft(absorption.real))

    trange = np.fft.rfftfreq(omegas.size, d=dw)*2*np.pi

    return trange, corr_func * dw


def polaron_density_op(omegas, spectral_data, r0, beta=None, decay = None):
       #find the phonon correlation function:
    tcorr, phon_corr = phonon_correlation_function(omegas, spectral_data, beta)

    #tfull = np.append(np.delete(np.flip(tcorr),-1),tcorr)
    #exponentiate into polaron form:
    Gt = np.exp(phon_corr)

    #The Frank-Condon factor is simply the correlation function evaluated at t=0:
    FC = 1/Gt[0]

    # # we also want the steady state of this function
    # phon_steady = Gt[-1]
   
    coherence = Gt * FC
    if decay == None:
        
        rcoh = r0[0][1] * coherence
        rpop = r0[0,0]

    else: 
        rpop = r0[0, 0] * np.exp(- tcorr * decay)
        

        rcoh = r0[0, 1] * coherence * np.exp(-0.5 * tcorr * decay)
        

    
    return tcorr, rcoh, rpop


tcorr, rcoh, rpop = polaron_density_op(omegas, spectral_data,0.5 * np.array([[1,1],[1,1]]), decay = Gamma)

fig, ax = plt.subplots(ncols=3)

ax[0].plot(tcorr, rcoh.real)
ax[0].plot(sys.trange, [np.real(r[0][1]) for r in sys.state_list], '.')

ax[1].plot(tcorr, rcoh.imag)
ax[1].plot(sys.trange, [np.imag(r[0][1]) for r in sys.state_list], '.')

ax[2].plot(tcorr, rpop)
ax[2].plot(sys.trange, [np.real(r[0,0]) for r in sys.state_list], '.')
for a in ax:
    a.set_xlim(0,sys.trange[-1])

ax[2].set_ylim(0,1)
#plt.plot(sys.trange, np.exp(reorg) * corr_func.real)
plt.show()
# #sys.bath.check_sample_rate()



