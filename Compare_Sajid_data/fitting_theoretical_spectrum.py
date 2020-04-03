import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from utils import phonon_sideband_spectrum, gen_spectral_dens, phonon_correlation_function



data = pd.read_csv('emission.dat.txt', sep="\t", header=None)
data.columns = ["frequencies", "spectrum"]
print(data.head())

fig = plt.figure(figsize = (12, 6))
plt.plot((data["frequencies"] - 16)*1000, data["spectrum"])
fig.savefig("test.png")


# Generating zero T results
#### Fourier domain
dt = 0.00001
N_pts = 2**18 # FFT runs much faster for N_pts = 2**N

file = "data.xlsx"



gamma_opt = 220 # np.sqrt(2*np.log(2))
gamma_deph = 0
gamma_ph = 300

omegas, spectral_data = gen_spectral_dens(file, gamma_ph, N_pts, 2*np.pi/dt)


detunings = -(omegas - np.max(omegas)/2)


sideband_spectrum = phonon_sideband_spectrum(detunings, omegas, spectral_data, None,2 *  gamma_opt, gamma_deph)
# sideband_spectrum = sideband_spectrum*omegas**3


spectrum_normalized = sideband_spectrum/np.max(sideband_spectrum*np.abs(detunings[0]-detunings[1]))


fig = plt.figure(figsize = (12, 6))
plt.plot(detunings/(2*np.pi)*0.00012, spectrum_normalized/np.max(spectrum_normalized))
plt.plot((data["frequencies"] - 16)*1000*0.00012, data["spectrum"]/np.max(data["spectrum"]))
plt.xlim(np.array([-6000, 2000])*0.00012)
plt.axhline(0)
fig.savefig("calc_spec_test.png")

fig = plt.figure(figsize = (12, 6))
plt.plot(omegas - np.max(omegas)/2, omegas**3)
plt.xlim([-6000, 2000])
plt.yscale("log")
fig.savefig("Checking prefactor.png")



# Picking part of the spectrum
detunings = detunings*0.00012/(2*np.pi)

boo_array = (detunings < 2.3)&(detunings > -2.3) 
detunings = detunings[boo_array] + 2.1
spectrum_normalized_cut = spectrum_normalized[boo_array]

luminescence_function = spectrum_normalized_cut*detunings**3
luminescence_function /= np.max(luminescence_function)

spectrum_normalized_cut /= np.max(spectrum_normalized_cut)


fig = plt.figure(figsize = (12, 6))
plt.plot(detunings, spectrum_normalized_cut, label = "Old")
plt.plot((data["frequencies"] - 16)*1000*0.00012 + 2.1, data["spectrum"]/np.max(data["spectrum"]), label = "Sajid")
plt.xlim([1, 2.3])
plt.plot(detunings, luminescence_function, label = "L(hv)")
plt.legend()
fig.savefig("testrun.png")

