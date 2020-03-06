# Code Written by: Mark Kamper Svendsen 05/03-2020
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, rfft, fftshift, ifftshift, irfft, rfft


# Matplotlib settings
matplotlib.rcParams["font.size"] = 20
matplotlib.use('Agg')

# Broadening parameters
gamma = 100
gamma2 = 250



# Replacement for the dirac delta functions
def broadening(w, w0, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((w-w0)/sigma)**2)


# Setting up the domains
dt = 0.00001
N_max = 6000 # In principle this should be 2**N
times = np.arange(0, N_max, dtype = "complex128")*dt
omegas = times/dt*(2*np.pi/(N_max*dt)) # Centered around zero!
dw = abs(omegas[1] - omegas[0])





# First we need to load in the relevant data
df = pd.read_excel("data.xlsx")
df_Sajid_spectrum = pd.read_excel("data.xlsx", "Emission and Absorption spectra")
# Benchmark spectrum
freqs_sajid = df_Sajid_spectrum["Freq.(1000 cm^-1)"]*1000
emissions_sajid = df_Sajid_spectrum["Emissin.(band strenght)"]
dw_sajid = abs(np.diff(freqs_sajid)[0])





df = df[df["Freq"] > 0] # Removing negative phonon frequencies
frequencies = 2*np.pi*df["Freq"].values # Turning into angular frequencies
huang_rhys_factors = df["Huang RhysFactor Si=λi/hʋi"].values # Extracting the Partial Huang Rhys factors

# Getting S(omega)
spectral_function = np.zeros_like(omegas, dtype = "complex128")
for index in range(len(huang_rhys_factors)):
    sk = huang_rhys_factors[index]*broadening(omegas, frequencies[index] ,gamma)
    spectral_function += sk

# Getting S0
s0  = np.sum(huang_rhys_factors)

# Time domain S(t)
St = ifft(spectral_function)*6E5 # Multiplicative factor for numerical stability

# Getting G and the spectral function A
G = np.exp(St - s0)*np.exp(-gamma2*times)
A = fftshift(fft(G))


# Normalization steps
A = A/(np.trapz(A, dx = dw))
emissions_sajid = emissions_sajid/np.trapz(emissions_sajid, dx = 2*np.pi*dw_sajid)


# Plotting 
shifted_omegas = omegas - omegas[int(len(omegas)/2)]

# Figure 
fig = plt.figure(figsize=(12, 6))

plt.title("Comparing Alkauskas and Duschinsky")
plt.plot(shifted_omegas/(2*np.pi), 2*A, label = "Numpy FFT based Alkauskas method")
plt.plot(freqs_sajid, emissions_sajid, label = "Sajid", linestyle = "--")
plt.xlim([-3000, np.max(freqs_sajid)])
plt.xlabel(r"$\nu$ $[\mathrm{cm}^{-1}]$")
plt.ylabel("$A(\hbar \omega)$ [A.U.]")
plt.grid()
plt.legend()
plt.tight_layout()
fig.savefig("Sideband_spectrum_comparison.png")




