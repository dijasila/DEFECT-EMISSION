import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
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



gamma_opt = 400 # np.sqrt(2*np.log(2))
gamma_deph = 0
gamma_ph = 300

omega_ZPL = 17000*2*np.pi

omegas, spectral_data = gen_spectral_dens(file, gamma_ph, N_pts, 2*np.pi/dt)



# Doing some dirty ticks here to make it possible to calculate the sideband spectra with the initial frequency grid as detunings.
detunings = -(omegas - np.max(omegas)/2) # Setting the peak at the ZPL omega
sideband_spectrum = phonon_sideband_spectrum(detunings, omegas, spectral_data, None, 2 *  gamma_opt, gamma_deph)
spectrum_normalized = sideband_spectrum/np.max(sideband_spectrum*np.abs(detunings[0]-detunings[1]))


# More trickery
boo_array = (detunings > -omega_ZPL)&(detunings < omega_ZPL)
spectrum_normalized = sideband_spectrum[boo_array]
detunings = detunings[boo_array] + omega_ZPL
spectrum_normalized = spectrum_normalized*detunings**3
spectrum_normalized /= np.max(spectrum_normalized*np.abs(detunings[0] - detunings[1]))


 # To eV


final_data_frame = pd.DataFrame()
final_data_frame["Detunings [cm^-1]"] = (detunings/(2*np.pi)).real
final_data_frame["Zero temperature spectrum"] = (spectrum_normalized/np.max(spectrum_normalized)).real


# Setting up temperature range
kb =69.36#(cm^-1 K^-1)
kb = 0.7280

Temp_range = np.array([4, 25,50,75,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1200, 1300])
# Temp_range = np.array([4, 25, 50, 75, 100, 125, 200, 250, 300, 350])
beta_range = 1/(kb * Temp_range)

for beta in beta_range:
    detunings = -(omegas - np.max(omegas)/2) # Setting the peak at the ZPL omega
    sideband_spectrum = phonon_sideband_spectrum(detunings, omegas, spectral_data, beta, 2 *  gamma_opt, gamma_deph)
    spectrum_normalized = sideband_spectrum/np.max(sideband_spectrum*np.abs(detunings[0]-detunings[1]))

    
    # More trickery
    boo_array = (detunings > -omega_ZPL)&(detunings < omega_ZPL)
    spectrum_normalized = sideband_spectrum[boo_array]
    detunings = detunings[boo_array] + omega_ZPL
    spectrum_normalized = spectrum_normalized*detunings**3
    spectrum_normalized /= np.max(spectrum_normalized*np.abs(detunings[0] - detunings[1]))


    # Saving data
    final_data_frame["T = {b:0.0f} K".format(b = 1/(kb*beta))] = (spectrum_normalized/np.max(spectrum_normalized)).real



fig = plt.figure(figsize = (12, 6))
for beta in beta_range:
    plt.plot(detunings/(2*np.pi)*0.00012, final_data_frame["T = {b:0.0f} K".format(b = 1/(kb*beta))], label = "{b:0.0f} K".format(b = 1/(kb*beta)))


plt.plot(detunings/(2*np.pi)*0.00012, final_data_frame["Zero temperature spectrum"], label = "0 K", linestyle = "--")


plt.title("Temperature dependent PL spectrum from the CbVn defect in hBN")
plt.xlabel("Frequency $[eV]$")
plt.ylabel("Normalized luminesence function")

plt.grid()
plt.legend(fontsize = 14, ncol = 2)
plt.xlim([0, (omega_ZPL/(2*np.pi) + 17000)*0.00012])

plt.tight_layout()
fig.savefig("Temperature_specs.png")


final_data_frame.to_csv("Temperature_PL_data.csv", index = False)
# final_data_frame.to_excel("Temperature_PL_data.xlsx")