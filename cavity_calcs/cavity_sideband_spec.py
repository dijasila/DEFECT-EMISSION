import numpy as np
import scipy as sc
import scipy.signal
import qutip as qt
import phonon_correlation_tools as phonon
import matplotlib.pyplot as plt
import ind_tools as ind
import pandas as pd
import pickle
import cavity_lio as lio


def initial_operators(dim):
    rho0 = qt.tensor(qt.create(2) * qt.destroy(2),
                     qt.fock_dm(dim, 0))  # tls in excited state

    # unpack to numpy vector
    rho0np = qt.operator_to_vector(rho0).full()

    #find system operators in vectorised form, for use in regression
    op = [qt.spre(qt.tensor(qt.destroy(2), qt.qeye(dim))).full(),
          qt.spre(qt.tensor(qt.qeye(2), qt.destroy(dim))).full()]


    #find the projector, which acts as a trace in lioville space
    proj = [qt.tensor(qt.create(2), qt.qeye(dim)).full().reshape([op[0].shape[0], 1]),
            qt.tensor(qt.qeye(2), qt.create(dim)).full().reshape([op[1].shape[0], 1])]

    fullproj = [qt.tensor(qt.create(2) * qt.destroy(2), qt.qeye(dim)).full().reshape([op[0].shape[0], 1]),
                qt.tensor(qt.qeye(2), qt.create(dim)*qt.destroy(dim)).full().reshape([op[0].shape[0], 1])]

    return rho0np, op, proj, fullproj


def brute_force_corr(trange, vals, amat):
    val_vec = np.nan_to_num(1/vals)
    rhs = amat.T @ val_vec

    g1 = np.array([np.sum(np.exp(vals * t) * rhs) for t in trange])

    return g1



def one_colour(w, vals, rhs):
    den = -1j * w - vals
    dat = - np.real(np.sum(rhs/den))
    return dat


def cavity_filter(w, center, width):
    h = (0.5 * width)
    h /= 1j*(w-width) + 0.5 * kappa
    return h


def emission_spectrum(dim, eps, g, wcav, driving, kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max, beta=None,method=None, wrange=None):
    
    #find the spectral density from the data.
    omegas, spectral_data = phonon.gen_spectral_dens(
        sd_file, phon_width, SD_sampling, w_max)

    #caclulate the phonon correlation functions
    trange, phi, Lxx, Lyy = phonon.polaron_correlation_functions(
        omegas, spectral_data, beta)
    
    
    #construct the Liouvillian:
    Liouvillian = lio.build_meq(dim, eps, g, wcav, driving, kappa, Gam_opt,
                    sd_file, phon_width, SD_sampling, w_max, beta) #could be streamlined...
    nplio = Liouvillian.full()
    
    # diagonalise the liouvillian
    vals, vecs = np.linalg.eig(nplio)

    #define some initial operators
    rho0np, op, proj, fullproj = initial_operators(dim)


    # construct the elements for the regression overlap matrices
    Amats = [ind.a_mat_build(vals, vecs, rho0np, op[nn], proj[nn])
             for nn in range(len(op))]

    detunings = omegas-np.max(omegas)/2

    if method == None or method=='conv':

        # calculate the rhs for the spectra calculations
        val_vec = np.nan_to_num(1/vals)
        rhs = [A.T @ val_vec for A in Amats]

        #calculate the zero phonon line contribution
        spec_dat_em = np.array([one_colour(w, vals, rhs[0])
                                for w in detunings]).real
        spec_dat_cav = np.array([one_colour(w, vals, rhs[1])
                                 for w in detunings])

        #calculate the sideband spectrum:
        sideband_contribution = np.exp(phi)
        sideband_spectrum = np.fft.fftshift(
            np.fft.irfft(sideband_contribution))
        
        #convolve the two signals:
        list_conv = sc.signal.fftconvolve(
            sideband_spectrum, spec_dat_em, mode='same')
        # notice the flip to make sure the detunings run in the right direction.

        #pack results into dictionary
        output = dict()
        output['markov'] = {'wrange':detunings,'emitter':spec_dat_em, 'cavity':spec_dat_cav}

        cav_fil = cavity_filter(detunings, wcav, kappa)
        non_mark_cav = np.abs(cav_fil)**2 
        non_mark_cav *= list_conv

        output['non_markov'] = {'wrange': detunings, 'emitter': list_conv, 'cavity': non_mark_cav}

    elif method == 'brute' and wrange != None:


        sideband_contribution = np.exp(phi)
        #caclulate the first order correlation functions
        g1_em, g1_cav = [brute_force_corr(trange, vals, A) for A in Amats]
        
        #calculate the non_markovian g1:
        g1_non_mark = sideband_contribution * g1_em
        
        s_dat_em = np.array([])
        s_dat_cav = np.array([])
        s_non_mark = np.array([])

        for w in wrange:
            expfactor = np.exp(1j * w * trange)

            #full non_markov:
            sdat = -np.sum(g1_non_mark * expfactor)*(trange[1]-trange[0])
            s_non_mark = np.append(s_non_mark, sdat)

            #emitter spectrum
            sdat = -np.sum(g1_em*expfactor)*(trange[1]-trange[0])
            s_dat_em = np.append(s_dat_em, sdat)

            #cavity spectrum
            sdat = -np.sum(g1_cav*expfactor)*(trange[1]-trange[0])
            s_dat_cav = np.append(s_dat_cav, sdat)

        output = dict()
        output['markov'] = {'wrange': wrange,
                         'emitter': s_dat_em.real, 'cavity': s_dat_cav.real}

        cav_fil = cavity_filter(wrange, wcav, kappa)
        non_mark_cav = (np.abs(cav_fil)**2)*s_non_mark
      

        output['non_markov'] = {'wrange': wrange,
                             'emitter': s_non_mark.real, 'cavity': non_mark_cav.real}

    return output


        



        
    






   
    


if __name__=='__main__':

    hbarc_cmtoev = 1.9746e-05  # eVcm
    sd_file = '../data.xlsx'
    phon_width = 200 * hbarc_cmtoev
    SD_sampling = 240000
    dt = 0.000001
    w_max = (2 * np.pi/dt)*hbarc_cmtoev

    kb = 8.617E-5  # eV K^-1
    Temp_range = np.array([4, 25, 50, 75, 100])
    beta_range = 1/(kb * Temp_range)
    beta = beta_range[0]

    #units are in eV
    dim = 2
    eps = 0
    g = 0.1
    omega_c = 0
    kappa = 0.1
    Gam_opt = 0.001
    driving = 0

    out = emission_spectrum(dim, eps, g, omega_c, driving, kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max)
    
    
    
    # detuning = np.linspace(-1,1, 400)
    # out_brute = emission_spectrum(dim, eps, g, omega_c, driving,
    #                               kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max, method='brute', wrange=detuning)
    # pickle.dump(out_brute, open('brute_comparison.p', 'wb'))
    
    #out_brute = pickle.load(open('brute_comparison.p', 'rb'))


    fig, ax = plt.subplots(1,2)

    ax[0].plot(out['markov']['wrange'], out['markov']
               ['cavity'])# / np.max(out['markov']['cavity']))

    ax[0].plot(out_brute['markov']['wrange'], out_brute['markov']['cavity'] -
               out_brute['markov']['cavity'][-1])  # /np.max(out_brute['markov']['cavity']))

    ax[1].plot(out['non_markov']['wrange'], out['non_markov']
             ['cavity'] )
    ax[1].plot(out_brute['non_markov']['wrange'], out_brute['non_markov']
               ['cavity']-out_brute['non_markov']
               ['cavity'][0])

    for a in ax:
        a.set_xlim(-1,1)
    plt.show()
  
