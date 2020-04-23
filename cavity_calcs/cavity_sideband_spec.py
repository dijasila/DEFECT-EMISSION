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
    h /= 1j*(w-center) + 0.5 * width
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
    SD_sampling = 120000
    dt = 0.00001
    w_max = (2 * np.pi/dt)*hbarc_cmtoev

    kb = 8.617E-5  # eV K^-1
    Temp_range = np.array([4, 25, 50, 75, 100])
    beta_range = 1/(kb * Temp_range)
    beta = beta_range[0]

    #units are in eV
    dim = 2
    eps = 0
    g = 0.01
    omega_c = 0
    kappa = 0.1
    Gam_opt = 0.001
    driving = 0
    fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharey=False)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    kappa_range = 0.1 * np.array([1E-2, 1E-1, 0.25,1])
    for nn, kappa in enumerate(kappa_range):
        out = emission_spectrum(dim, eps, g, omega_c, driving, kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max)

        norm = np.max(out['markov']['cavity'])
        ax[0][0].plot(2.1-out['markov']['wrange'],
                      out['markov']['cavity']/norm)
        
        norm = np.max(out['non_markov']['cavity'])
        ax[0][1].plot(2.1-out['non_markov']['wrange'],
                      out['non_markov']['cavity']/norm, color=colors[nn])

        cav_fil = np.abs(cavity_filter(out['markov']['wrange'], 2.1, kappa))**2/np.abs(
            cavity_filter(2.1, 2.1, kappa))**2
        ax[0][1].plot(out['markov']['wrange'],cav_fil,'--',color = colors[nn])
    ax[0][0].legend(kappa_range, title=r'$\kappa$ (eV)')

    kappa = 0.01
    grange = [1E-3, 1E-2, 1E-1]
    for g in grange:
        out = emission_spectrum(dim, eps, g, omega_c, driving, kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max)
        
        norm = np.max(out['markov']['cavity'])
        ax[1][0].plot(2.1-out['markov']['wrange'], out['markov']['cavity']/norm)

        norm = np.max(out['non_markov']['cavity'])
        ax[1][1].plot(2.1-out['non_markov']['wrange'],
                      out['non_markov']['cavity']/norm)
    cav_fil = np.abs(cavity_filter(out['non_markov']['cavity'],2.1, kappa))**2
    ax[1][1].plot(out['non_markov']['cavity'],cav_fil/np.max(cav_fil),'--')
    ax[1][0].legend(grange, title = 'g (eV)')
    titles = ['(a) Markovian', '(b) Non-Markovian', '(c) Markovian', '(d) Non-Markovian']
    for index, a in enumerate(ax.flat):
        a.set_xlim(1.5,2.15)
        a.set_ylim(-0.1,1.1)
        a.set_xlabel(r'Energy, $\hbar\omega$ (eV)')
        a.set_title(titles[index])

    ax[0][0].set_xlim(2.085,2.115)
    ax[1][0].set_xlim(1.9, 2.2)

    ax[0][0].set_yscale('linear')
    ax[0][1].set_yscale('log')
    ax[1][1].set_yscale('log')
    ax[0][1].set_ylim(1E-5,1.5)
    ax[1][1].set_ylim(1E-5,1.5)
    plt.tight_layout()
    plt.savefig('cavity_emission_small_kappa.pdf')
  
