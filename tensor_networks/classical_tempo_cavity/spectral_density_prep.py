import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt



def broadening(w, w0, sigma):
    # Replacement for the dirac delta functions
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((w-w0)/sigma)**2)


def gen_spectral_dens(file, phon_width, SD_sampling, w_max,  plot=None):
    #Function to generate the spectral density from data

    # Setting up the domains
    N_max = SD_sampling  # SD_sampling #number of frequency points to sample
    # extract the interval required to hit the max frequency.
    dt = 2 * np.pi / w_max

    times = np.arange(0, N_max, dtype="complex128")*dt
    omegas = times/dt*(2*np.pi/(N_max*dt))  # Centered around zero!

    # First we need to load in the relevant data
    df = pd.read_excel(file)

    df = df[df["Freq"] > 0]  # Removing negative phonon frequencies
    frequencies = 2*np.pi*df["Freq"].values  # Turning into angular frequencies
  #  hc_cmtoev = 1.2408E-4 #eVcm
    hbarc_cmtoev = 1.9746e-05  # eVcm

    frequencies = hbarc_cmtoev * frequencies  # convert to electron volts
    huang_rhys_factors = df["Huang RhysFactor Si=λi/hʋi"].values

    # extract the spectral density, S(omega)
    spectral_function = np.zeros_like(omegas, dtype="complex128")
    for index in range(len(huang_rhys_factors)):
        sk = huang_rhys_factors[index] * \
            broadening(omegas, frequencies[index], phon_width)
        spectral_function += sk

    if plot == True:
        fig = plt.figure()
        plt.plot(omegas.real, spectral_function.real, '.')
        #plt.xlim(0,10000)
        plt.tight_layout()
        #fig.savefig("test_sk_new.png")
        plt.show()

    return omegas, spectral_function
