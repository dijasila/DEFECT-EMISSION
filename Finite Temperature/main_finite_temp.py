# Code Written by: Mark Kamper Svendsen 05/03-2020
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, rfft, fftshift, ifftshift, irfft, rfft
# Matplotlib settings
matplotlib.rcParams["font.size"] = 20
matplotlib.use('Agg')


# Replacement for the dirac delta functions
def broadening(w, w0, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((w-w0)/sigma)**2)


# Setting up the new St from Jake
def phonon_correlation_function(omegas, spectral_data, beta):

    dw = np.abs(omegas[0] - omegas[1])


    if beta==None:
        spectral_density = np.zeros_like(omegas)
        boo_array = omegas==0
        spectral_density[boo_array] = 0

        boo_array= omegas > 0 
        spectral_density[boo_array] = spectral_data[boo_array]
        # Go to the time domain
        corr_func = np.fft.rfft(spectral_density.real)
        
       
    elif beta >0:
        absorption = np.zeros_like(omegas)
        emission = np.zeros_like(omegas)


        boo_array = omegas==0
        absorption[boo_array]=0
        emission[boo_array] = 0 

        boo_array= omegas > 0 
        absorption[boo_array] = spectral_data[boo_array] * nocc(omegas[boo_array], beta)
        emission[boo_array] = spectral_data[boo_array] * (nocc(omegas[boo_array], beta)+1)

        corr_func =(np.fft.rfft(emission.real))
        corr_func += np.conj(np.fft.rfft(absorption.real) )

    
    trange = np.fft.rfftfreq(omegas.size, d=dw)*2*np.pi 
    return trange, corr_func * dw


def get_spectal_function(omegas, huang_rhys_factors, angular_frequencies_HRF):
    # Getting S(omega)
    spectral_function = np.zeros_like(omegas)
    for index in range(len(huang_rhys_factors)):
        sk = huang_rhys_factors[index]*broadening(omegas, frequencies[index] ,gamma)
        spectral_function += sk


    # Getting S0
    s0 = np.sum(huang_rhys_factors)


    return spectral_function, s0



# Broadening parameters
gamma = 120
gamma2 = 220


# Temperature parameter
beta = None


# Setting up the domains
dt = 0.00001
N = 18
N_max = 2**N # In principle this should be 2**N
times = np.arange(0, N_max)*dt
omegas = times/dt*(2*np.pi/(N_max*dt)) # Centered around zero!
dw = abs(omegas[1] - omegas[0])

# megas = np.fft.rfftfreq(N_max, d = dw)


# First we need to load in the relevant data
df = pd.read_excel("../data.xlsx")
df_Sajid_spectrum = pd.read_excel("../data.xlsx", "Emission and Absorption spectra")
df = df[df["Freq"] > 0] # Removing negative phonon frequencies
frequencies = 2*np.pi*df["Freq"].values # Turning into angular frequencies
huang_rhys_factors = df["Huang RhysFactor Si=λi/hʋi"].values # Extracting the Partial Huang Rhys factors

# Benchmark spectrum
freqs_sajid = df_Sajid_spectrum["Freq.(1000 cm^-1)"]*1000
emissions_sajid = df_Sajid_spectrum["Emissin.(band strenght)"]
dw_sajid = abs(np.diff(freqs_sajid)[0])


# Getting spectral function
spectral_function, s0 = get_spectal_function(omegas, huang_rhys_factors, frequencies)
# Getting phonon correlation function
times, td_spectral_function = phonon_correlation_function(omegas, spectral_function, None)



fig = plt.figure(figsize=(12, 6))
plt.plot(times, td_spectral_function)
fig.savefig("St_test.png")


G = np.exp(td_spectral_function - s0)*np.exp(-gamma2*times)
A = fftshift(np.fft.irfft(G))

print(gamma)
print(gamma2)
fig = plt.figure(figsize=(12, 6))
plt.plot(omegas - np.max(omegas)/2, A)
# plt.xlim([-3000, 10000])
fig.savefig("test_A.png")

A = A/(np.trapz(A, dx = dw))
emissions_sajid = emissions_sajid/np.trapz(emissions_sajid, dx = 2*np.pi*dw_sajid)

fig = plt.figure(figsize=(12, 6))
plt.title("Checking rFFT method")
plt.plot((omegas - np.max(omegas)/2)/(2*np.pi), A, label = "Numpy rFFT method")
plt.plot(freqs_sajid, emissions_sajid, label = "Sajid", linestyle = "--")
plt.xlim([-3000, np.max(freqs_sajid)])
plt.xlabel(r"$\nu$ $[\mathrm{cm}^{-1}]$")
plt.ylabel("$A(\hbar \omega)$ [A.U.]")
plt.grid()
plt.legend()
plt.tight_layout()
fig.savefig("Sideband_spectrum_comparison2.png")


