import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, rfft, fftshift, ifftshift, irfft, rfft


# Matplotlib settings
matplotlib.rcParams["font.size"] = 20
matplotlib.use('Agg')

# Broadening parameters
gamma = 150
gamma2 = 200



# Replacement for the dirac delta functions
def broadening(w, w0, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((w-w0)/sigma)**2)


# Setting up the domains
dt = 0.00001
N_max = 6000
times = np.arange(0, N_max, dtype = "complex128")*dt
omegas = times/dt*(2*np.pi/(N_max*dt)) # Centered around zero!
dw = abs(omegas[1] - omegas[0])





# First we need to load in the relevant data
df = pd.read_excel("../data.xlsx")
df_Sajid_spectrum = pd.read_excel("../data.xlsx", "Emission and Absorption spectra")
print(df_Sajid_spectrum.columns)

freqs_sajid = df_Sajid_spectrum["Freq.(1000 cm^-1)"]*1000
emissions_sajid = df_Sajid_spectrum["Emissin.(band strenght)"]
dw_sajid = abs(np.diff(freqs_sajid)[0])





df = df[df["Freq"] > 0] # Removing negative phonon frequencies
frequencies = 2*np.pi*df["Freq"].values # Turning into angular frequencies
huang_rhys_factors = df["Huang RhysFactor Si=λi/hʋi"].values

# Getting S(omega)
spectral_function = np.zeros_like(omegas, dtype = "complex128")
for index in range(len(huang_rhys_factors)):
    sk = huang_rhys_factors[index]*broadening(omegas, frequencies[index] ,gamma)
    spectral_function += sk

s0  = np.sum(huang_rhys_factors)



fig = plt.figure()
plt.plot(omegas.real, spectral_function.real)
fig.savefig("test_sk_new.png")





# S(t)
# St2 = ifft(ifftshift(spectral_function)) REAL!
St2 = ifft(spectral_function)


# Getting S(t)
St = np.zeros_like(times)

# Optimize here. Rewrite in terms of fft
for index in range(times.shape[0]):
    St[index] = np.sum(spectral_function*np.exp(-1j*omegas*times[index]))*dw


fig = plt.figure(figsize = (12, 6))
plt.plot(St, label = "Numpy FFT")
plt.plot(St2*6E5, label = "Home cooked")
plt.yscale("log")
plt.legend()
fig.savefig("FFT_test.png")


G2= np.exp(St2*6E5 - s0)*np.exp(-gamma2*times)
A2 = fftshift(fft(G2))

G= np.exp(St - s0)*np.exp(-gamma2*times)
A = np.flip(fftshift(fft(G)))


fig = plt.figure(figsize=(12, 6))
plt.plot(times, G)
plt.plot(times, G2)
fig.savefig("time_domain_test_G.png")


# Normalization steps
A = A/(np.trapz(A, dx = dw))
A2 = A2/np.trapz(A2, dx = dw)
emissions_sajid = emissions_sajid/np.trapz(emissions_sajid, dx = 2*np.pi*dw_sajid)




shifted_omegas = omegas - omegas[int(len(omegas)/2)]
fig = plt.figure(figsize=(12, 6))
plt.plot(shifted_omegas/(2*np.pi), A, label = "Home cooked")
plt.plot(shifted_omegas/(2*np.pi), A2, label = "Numpy")
plt.plot(freqs_sajid, emissions_sajid, label = "Sajid", linestyle = "--")
# plt.plot(omegas, np.flip(fft(G)), label = "Home cooked")
# plt.plot(omegas, fft(G2), label = "Numpy")
plt.xlim([-3000, np.max(freqs_sajid)])
# plt.ylim([0, 200])
plt.legend()
fig.savefig("fre_domain_test_G.png")






##########################################
# St = ifft(ifftshift(spectral_function))
# Sw = fftshift(fft(St))
##########################################

# fig = plt.figure(figsize = (12, 6))
# plt.plot(omegas.real, Sw, label = "Unshifted")
# plt.plot(omegas.real, fftshift(Sw), label = "Shifted")
# plt.legend()
# fig.savefig("FFT_test.png")

################ Works till here! ###############






"""
St = ifft(ifftshift(spectral_function))



G = np.exp(St - s0)*np.exp(-gamma2*np.abs(times))

G = np.exp(-gamma2*np.abs(times))
A = fftshift(fft(G))

emission_spectrum = np.flip(A.real)
emission_spectrum = emission_spectrum/np.trapz(emission_spectrum, dx = dw)
emissions_sajid = emissions_sajid/np.trapz(emissions_sajid, dx = 2*np.pi*abs(freqs_sajid[1] - freqs_sajid[0]))


fig = plt.figure(figsize=(10, 10))

plt.title("PL spectrum")
plt.plot(omegas.real/(2*np.pi), 2*emission_spectrum, linewidth = 3, label = "Alkauskas")
plt.plot(freqs_sajid, emissions_sajid, linewidth = 3, linestyle = "--", label = "Duschinsky")
plt.grid()
plt.xlabel(r"$\nu$ [$\mathrm{cm}^{-1}$]")
plt.ylabel(r"$A(\hbar \omega)$ [A.U.]")
# plt.xlim([0, np.max(freqs_sajid)])
plt.legend()
plt.tight_layout()
fig.savefig("compare_spectrum_test2.png")
"""

