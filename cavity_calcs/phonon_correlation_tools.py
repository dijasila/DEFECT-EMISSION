# Code written by Jake Iles-Smith
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, rfft, fftshift, ifftshift

def broadening(w, w0, sigma):
    # Replacement for the dirac delta functions
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((w-w0)/sigma)**2)

def gen_spectral_dens(file, phon_width, SD_sampling, w_max,  plot=None):
    #Function to generate the spectral density from data


    # Setting up the domains
    N_max = SD_sampling#SD_sampling #number of frequency points to sample
    dt = 2 * np.pi / w_max# extract the interval required to hit the max frequency.

    times = np.arange(0, N_max, dtype = "complex128")*dt
    omegas = times/dt*(2*np.pi/(N_max*dt)) # Centered around zero!
    dw = abs(omegas[1] - omegas[0])


    # First we need to load in the relevant data
    df = pd.read_excel(file, engine='openpyxl')

    df = df[df["Freq"] > 0] # Removing negative phonon frequencies
    frequencies = 2*np.pi*df["Freq"].values # Turning into angular frequencies
  #  hc_cmtoev = 1.2408E-4 #eVcm
    hbarc_cmtoev = 1.9746e-05  # eVcm
    
    frequencies = hbarc_cmtoev * frequencies  # convert to electron volts
    huang_rhys_factors = df["Huang RhysFactor Si=λi/hʋi"].values
    
   
    # extract the spectral density, S(omega)
    spectral_function = np.zeros_like(omegas, dtype = "complex128")
    for index in range(len(huang_rhys_factors)):
        sk = huang_rhys_factors[index]*broadening(omegas, frequencies[index] ,phon_width)
        spectral_function += sk
    

    if plot == True:
        fig = plt.figure()
        plt.plot(omegas.real, spectral_function.real,'.')
        #plt.xlim(0,10000)
        plt.tight_layout()
        #fig.savefig("test_sk_new.png")
        plt.show()
    
    return omegas, spectral_function


def nocc(w,beta):
    return 1/(np.exp(beta * w) - 1)



def phonon_propagator(omegas, spectral_data, beta):

    #here we calculate the phonon propagator, commonly reffered to as \phi(t)
    # using a spectral density calculated througb DFT. This is done through a FFT.

    dw = np.abs(omegas[0] - omegas[1])


    if beta==None:
        spectral_density =np.zeros_like(omegas)
        boo_array = omegas==0
        spectral_density[boo_array]=0

        boo_array= omegas > 0 
        spectral_density[boo_array] = spectral_data[boo_array]
        corr_func = np.fft.rfft(spectral_density.real)#np.conj(np.fft.rfft(spectral_data.real))
        
       
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


def polaron_correlation_functions(omegas, spectral_function, beta):
    # function builds the bath correlation function
    # find the phonon propagator
    trange, phi = phonon_propagator(omegas, spectral_function, beta)

    #calculate the phonon renormalisation
    Bren =np.exp(-0.5 * phi[0])    

    #build the correlation functions
    Lxx = 0.5 * (Bren ** 2) * (np.exp(phi) + np.exp(-phi) - 2)
    Lyy = 0.5 * (Bren ** 2) * (np.exp(phi) - np.exp(-phi))

    return trange, phi, Lxx, Lyy


def brute_correlation_function(trange, omegas, spectral_data, beta):

    dw = np.abs(omegas[0] - omegas[1])
    corr_data = np.array([])
    if beta==None:

        corr_func = np.fft.ifft(spectral_data)
       
    elif beta >0:
        absorption = np.zeros_like(omegas)
        emission = np.zeros_like(omegas)


        boo_array = omegas==0
        absorption[boo_array]=0
        emission[boo_array] = 0 

        boo_array= omegas > 0 
        absorption[boo_array] = spectral_data[boo_array] * nocc(omegas[boo_array], beta)
        emission[boo_array] = spectral_data[boo_array] * (nocc(omegas[boo_array], beta)+1)
        for t in trange:
            factor1 = np.exp(1j * t * omegas)
            factor2 = np.conj(factor1)

            corr_func = (np.sum(absorption * factor1) + np.sum(emission * factor2)) * dw
            corr_data = np.append(corr_data, corr_func)
    return corr_data



def spectra_zpl(w, Gamma_opt, Gamma_deph):
    spec= (Gamma_opt + Gamma_deph)/(2 * Gamma_opt)
    spec /= (Gamma_opt + Gamma_deph)**2 + w**2 
    return spec


def phonon_sideband_spectrum(omegas, spectral_data, beta, Gamma_opt, Gamma_deph):

    #find the phonon correlation function:
    tcorr, phon_corr = phonon_correlation_function(omegas, spectral_data, beta)
    
    #tfull = np.append(np.delete(np.flip(tcorr),-1),tcorr)
    #exponentiate into polaron form:
    Gt = np.exp(phon_corr)
    G_data = np.append(np.delete(np.flip(Gt),-1), Gt)

    #The Frank-Condon factor is simply the correlation function evaluated at t=0:
    FC = Gt[0]

    # we also want the steady state of this function
    phon_steady = Gt[-1]

    # find the relevant optical contribution: 
    g1bin = np.exp(0.5 *( Gamma_opt + Gamma_deph) * tcorr)
    g1bin /= Gamma_opt

    # the signal that will be Fourier transformed is then:
    signal = Gt-phon_steady
    signal *= FC * g1bin


    # Fourier transform this to find the sidband contribution to the spectra:
    sideband_spec = np.fft.fftshift(np.fft.fft(signal))
    plt.plot(sideband_spec.real)
    plt.show()



def brute_phonon_sideband_spectrum(detunings, omegas, spectral_data, beta, Gamma_opt, Gamma_deph):

    #find the phonon correlation function:
    tcorr, phon_corr = phonon_correlation_function(omegas, spectral_data, beta)

    #tfull = np.append(np.delete(np.flip(tcorr),-1),tcorr)
    #exponentiate into polaron form:
    Gt = np.exp(phon_corr)
   
    #The Frank-Condon factor is simply the correlation function evaluated at t=0:
    FC = 1/Gt[0]
    
    # we also want the steady state of this function
    phon_steady = Gt[-1]

    # find the relevant optical contribution: 
    g1bin = np.exp(-0.5 *( Gamma_opt + Gamma_deph) * tcorr)
    g1bin /= Gamma_opt
    

    # the signal that will be Fourier transformed is then:
    signal = Gt * FC * g1bin

    dt = np.abs(tcorr[0]-tcorr[1])
 
    spec_list = np.array([])
 
    for w in detunings:
        four_factor = np.exp(-1j* w * tcorr)
        arr = four_factor * signal * dt
        spec_list = np.append(spec_list, np.sum(arr))

    return spec_list



def phonon_sideband_spectrum(detunings,omegas, spectral_data, beta, Gamma_opt, Gamma_deph):
       #find the phonon correlation function:
    tcorr, phon_corr = phonon_correlation_function(omegas, spectral_data, beta)

    #tfull = np.append(np.delete(np.flip(tcorr),-1),tcorr)
    #exponentiate into polaron form:
    Gt = np.exp(phon_corr)
   
    #The Frank-Condon factor is simply the correlation function evaluated at t=0:
    FC = 1/Gt[0]
    
    # we also want the steady state of this function
    phon_steady = Gt[-1]

    # find the relevant optical contribution: 
    g1bin = np.exp(-0.5 *( Gamma_opt + Gamma_deph) * tcorr)
    g1bin /= Gamma_opt
    

    # the signal that will be Fourier transformed is then:
    signal = Gt * FC * g1bin

    dt = np.abs(tcorr[0]-tcorr[1])
 
    spec_list = np.array([]) 

    sideband_spectrum = ifftshift(np.fft.irfft(signal))
    return sideband_spectrum




if __name__=='__main__':


    file = "data.xlsx"

    dt = 0.00001#2 * np.pi / w_max# extract the interval required to hit the max frequency.

    omegas, spectral_data = gen_spectral_dens(file, 200, 120000, 2* np.pi/dt)
    
    
    Gamma_opt = 250
    Gamma_deph = 0
    # detunings = np.linspace(-6000,2000,800)*(2 * np.pi)
    detunings = -(omegas - np.max(omegas)/2)
    # detunings = np.linspace(-np.max(omegas)/2, np.max(omegas)/2, len(omegas)/4 + 1)
    # spec_list = brute_phonon_sideband_spectrum(detunings, omegas, spectral_data, None,2 *  Gamma_opt, Gamma_deph)
    spec_list = phonon_sideband_spectrum(detunings, omegas, spectral_data, None,2 *  Gamma_opt, Gamma_deph)
    
    
    
    spec_norm = spec_list/np.max(spec_list*np.abs(detunings[0]-detunings[1]))
    
    
    kb =69.36#(cm^-1 K^-1)
    Temp_range = np.array([4, 25,50,75,100])
    beta_range = 1/(kb * Temp_range)

    plt.rcParams.update({'font.size':20,
                         'lines.linewidth':2})


    fig, ax = plt.subplots(2,2, figsize=(20,10),sharey=False)
    zero_temp = phonon_correlation_function(omegas, spectral_data,None )
    ax[0][0].plot(omegas, spectral_data)
    ax[0][0].set_xlim(0,12000)
    ax[0][1].plot( detunings/(2 * np.pi), spec_norm.real, 'r-')
    ax[0][1].set_xlim([-6000, 2000])
  
    for beta in beta_range:
        print("Calculating for beta = {}".format(beta))
        temp = phonon_correlation_function(omegas, spectral_data, beta)

        ax[1][0].plot(temp[0], temp[1].real,'--')
        # detunings_HT = np.linspace(-6000,4000,1000)*(2 * np.pi)

        detunings_HT = detunings.copy()


        spectrum_HT = phonon_sideband_spectrum(detunings_HT, omegas, spectral_data, beta,2 *  Gamma_opt, Gamma_deph)
        df = np.abs(detunings_HT[0]-detunings_HT[1]) 
        # print(spectrum_HT)   
        norm_spec = spectrum_HT/np.max(spectrum_HT * df)
        ax[1][1].plot(detunings_HT/(2 * np.pi), norm_spec, label = r"$\beta = {b:0.1e}$".format(b = beta))


    ax[1][1].set_xlim([-12000, 8000])
    ax[1][1].legend(fontsize = 14)


    ax[1][0].set_xlim(0,0.025)
    # for beta in beta_range:
    #     sp_temp = brute_phonon_sideband_spectrum(detunings, omegas, spectral_data, beta,2 *  Gamma_opt, Gamma_deph)
    #     ax[1].plot(detunings,sp_temp)
    ax[1][0].plot(zero_temp[0], zero_temp[1].real)
    
    ax[0][0].set_ylabel('Spectral density')
    ax[0][1].set_ylabel('Spectrum (arb. units)')
    ax[1][1].set_ylabel('Spectrum (arb. units)')
    ax[0][0].set_xlabel(r'Detuning (cm$^{-1}$)')
    ax[0][1].set_xlabel(r'Detuning (cm$^{-1}$)')
    ax[1][1].set_xlabel(r'Detuning (cm$^{-1}$)')
   

    ax[0][1].set_xlabel('Time (inv. wavenumbers)')
    ax[1][0].set_ylabel('Phonon Correlation function')
  #ax[0].set_yscale('log')
    fig.suptitle("Results from 1st FFT based finite temperature version", y = 1.0)


    plt.tight_layout()
    plt.show()
    fig.savefig('Temperature_dependent_spectra.png')
