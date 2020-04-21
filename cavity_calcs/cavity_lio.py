import numpy as np
import scipy as sc
import scipy.signal
import qutip as qt
import phonon_correlation_tools as phonon
import matplotlib.pyplot as plt
import ind_tools as ind
import pandas as pd
import pickle


def lind_func(A):
    """
    Vectorised Lindblad superoperator.
    """
    L = 2 * qt.sprepost(A, A.dag())
    L += -qt.spre(A.dag() * A) - qt.spost(A.dag() * A)
    return L

def system_hamiltonian(dim, eps, g, omega_c, driving):

    H = eps * qt.tensor(qt.create(2) * qt.destroy(2), qt.qeye(dim)) #self-energy of the emitter
    H += g* qt.tensor(qt.create(2), qt.destroy(dim)) #interaction term
    H += g * qt.tensor(qt.destroy(2), qt.create(dim)) #conjugate of the above
    H += omega_c * qt.tensor(qt.qeye(2), qt.create(dim) * qt.destroy(dim)) #self-energy of the cavity mode
    H += driving * qt.tensor(qt.qeye(2), qt.create(dim) + qt.destroy(dim))
    return H


def rate_functions(sd_file, phon_width, SD_sampling, w_max, beta):

    # construct the spectral density:
    # 2 * np.pi / w_max# extract the interval required to hit the max frequency.
    dt = 0.00001
    omegas, spectral_data = phonon.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)

    #construct the phonon correlation functions:
    trange, phi, Lxx, Lyy = phonon.polaron_correlation_functions(
        omegas, spectral_data, beta)

    # extract the phonon renormalisation factor:
    Bren = np.exp(-0.5 * phi[0])

    #build the correlation functions for iterating over:
    freqs = np.fft.fftshift(np.fft.fftfreq(Lxx.shape[0], dt))
    four_xx = np.fft.fftshift(np.fft.fft(Lxx))
    four_yy = np.fft.fftshift(np.fft.fft(Lyy))

    f_xx = sc.interpolate.interp1d(freqs, four_xx, kind='nearest')
    f_yy = sc.interpolate.interp1d(freqs, four_yy, kind='nearest')
    return Bren, f_xx, f_yy

    
def phonon_dissipator(H, Bren, f_xx, f_yy):


    dim = int(H.shape[0]/2)
    
    #diagonalise the system Hamiltonian:
    vals, vecs = H.eigenstates()

    x_coupop = qt.tensor(qt.create(2), qt.destroy(dim)) 
    x_coupop += qt.tensor(qt.destroy(2), qt.create(dim))

    y_coupop = 1j*qt.tensor(qt.create(2), qt.destroy(dim)) 
    y_coupop -= 1j * qt.tensor(qt.destroy(2), qt.create(dim))

    x_rate_op = 0
    y_rate_op = 0

    for ni, vi in enumerate(vecs):
        for nj, vj in enumerate(vecs):
            lamb_ij = vals[ni] - vals[nj]

            x_ij = x_coupop.matrix_element(vi.dag(),vj)
            y_ij = y_coupop.matrix_element(vi.dag(),vj)

            if np.abs(x_ij)>0:
                x_rate_op += (f_xx(lamb_ij)) * x_ij * vi * vj.dag()

            if np.abs(y_ij)>0:
                y_rate_op += (f_yy(lamb_ij) )* y_ij * vi * vj.dag()
    
    # add the x components to the dissipator
    K_therm = qt.spre(x_coupop * x_rate_op)
    K_therm -= qt.sprepost(x_coupop,x_rate_op)
    K_therm += qt.spost(x_rate_op.dag() * x_coupop)
    K_therm -= qt.sprepost(x_coupop, x_rate_op.dag())

    # and now the y:
    K_therm += qt.spre(y_coupop * y_rate_op)
    K_therm -= qt.sprepost(y_coupop, y_rate_op)
    K_therm += qt.spost(y_rate_op.dag() * y_coupop)
    K_therm -= qt.sprepost(y_coupop, y_rate_op.dag())


    return -K_therm


def master_equation(dim, eps, g, wc,driving, kappa, Gam_opt, bren, f_xx, f_yy):

    H = system_hamiltonian(dim, eps, g * bren, wc, driving)

    unit = -1j * qt.spre(H) + 1j * qt.spost(H)

    cav_diss = 0.5 * kappa * lind_func(qt.tensor(qt.qeye(2), qt.destroy(dim)))
    leakage = 0.5 * Gam_opt * lind_func(qt.tensor(qt.destroy(2), qt.qeye(dim)))
    K_therm = phonon_dissipator(H, bren, f_xx, f_yy)

    return unit + cav_diss + leakage + K_therm * (g * bren)**2


def build_meq(dim, eps, g, wc, driving,  kappa,Gam_opt, sd_file, phon_width, SD_sampling, w_max, beta):
    
    Bren, f_xx, f_yy = rate_functions(sd_file, phon_width, SD_sampling, w_max, beta)
    print('Phonon renormalisation is ', Bren)

    meq = master_equation(dim, eps, g, wc,driving, kappa, Gam_opt, Bren, f_xx, f_yy)

    return meq

def initial_operators(dim):
    rho0 = qt.tensor(qt.create(2) * qt.destroy(2), qt.fock_dm(dim, 0))  # tls in excited state
    
    # unpack to numpy vector
    rho0np = qt.operator_to_vector(rho0).full()

    #find system operators in vectorised form, for use in regression
    op = [qt.spre(qt.tensor(qt.destroy(2), qt.qeye(dim))).full(),
          qt.spre(qt.tensor(qt.qeye(2), qt.destroy(dim))).full()]

    #op_dag = [qt.spre(qt.tensor(qt.displace(dim, eta/nu),e0 * g0.dag())).full(),
    #qt.spre(qt.tensor(qt.qeye(dim), g1 * g0.dag())).full()]

    #find the projector, which acts as a trace in lioville space
    proj = [qt.tensor(qt.create(2), qt.qeye(dim)).full().reshape([op[0].shape[0], 1]),
   qt.tensor(qt.qeye(2), qt.create(dim)).full().reshape([op[1].shape[0],1])]

    fullproj = [qt.tensor(qt.create(2) * qt.destroy(2),qt.qeye(dim)).full().reshape([op[0].shape[0],1]),
    qt.tensor(qt.qeye(2),qt.create(dim)*qt.destroy(dim)).full().reshape([op[0].shape[0],1])]

    return rho0np, op, proj, fullproj


def one_colour(w, vals, rhs):
    den = 1j * w - vals
    dat = - np.real(np.sum(rhs/den))
    return dat

def brute_force_corr(trange, vals, amat):
    val_vec = np.nan_to_num(1/vals)
    rhs = amat.T @ val_vec

    g1 = np.array([np.sum(np.exp(vals * t) * rhs) for t in trange])

    return g1





if __name__ =='__main__':
    hbarc_cmtoev = 1.9746e-05  # eVcm
    sd_file = '../data.xlsx'
    phon_width = 200 * hbarc_cmtoev
    SD_sampling = 120000
    dt = 0.00001
    w_max = (2 * np.pi/dt)*hbarc_cmtoev
    
    kb = 8.617E-5#eV K^-1
    Temp_range = np.array([4, 25, 50, 75, 100])
    beta_range = 1/(kb * Temp_range)
    beta = beta_range[0]
    

    omegas, spectral_data = phonon.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)
   
    trange, phi, Lxx, Lyy = phonon.polaron_correlation_functions(omegas, spectral_data, beta)
    sideband_contribution = np.exp(phi)#Lxx + Lyy

    sideband_spectrum = np.fft.fftshift(np.fft.irfft(sideband_contribution))
    d_omega = np.abs(omegas[0]-omegas[1]) 
    # plt.plot(omegas - np.max(omegas)/2, sideband_spectrum)
    # plt.show()

    #units are in eV
    dim=2
    eps=0
    g=0.1
    omega_c=0
    kappa=0.1
    Gam_opt = 0.001 
    driving = 0

    lio  = build_meq(dim, eps, g, omega_c, driving, kappa, Gam_opt, sd_file, phon_width, SD_sampling, w_max, beta)
    nplio = lio.full()

    vals, vecs = np.linalg.eig(nplio)
    rho0np, op, proj, fullproj = initial_operators(dim)

    Amats = [ind.a_mat_build(vals, vecs, rho0np, op[nn],proj[nn])
    for nn in range(len(op))]
    # a_list = [np.sum(np.nan_to_num(A/vals), axis=1) for A in a_mats]

    wrange_zpl = np.arange(-5, 5, d_omega)
   

    # calculate the ZPL contributions
    val_vec = np.nan_to_num(1/vals)
    rhs = [A.T @ val_vec for A in Amats]
    
    spec_dat_em = np.array([one_colour(w, vals, rhs[0])
                            for w in omegas-np.max(omegas)/2]).real
    spec_dat_cav = np.array([one_colour(w, vals, rhs[1]) for w in wrange_zpl])


    #calculate the spectra brute force to compare.
    g1 = brute_force_corr(trange, vals, Amats[0])
    g1_full = sideband_contribution * g1
    # s_arr = np.array([])
    # for w in wrange_zpl:
    #     expfactor = np.exp(1j * w * trange)
    #     sdat = -np.sum(g1_full * expfactor)*(trange[1]-trange[0])
    #     s_arr = np.append(s_arr, sdat)

   # pickle.dump(s_arr, open('brute_spec.p','wb') )

    s_arr = pickle.load(open('brute_spec.p','rb'))
    fig, ax = plt.subplots(1,2)

    ax[1].plot(trange,np.abs(g1_full).real,'.')
    ax[1].set_xscale('log')
    ax[0].plot(np.arange(-5, 5, d_omega), s_arr.real/np.max(s_arr.real))
    #plt.plot(omegas-np.max(omegas)/2, spec_dat_em.real, '--')
    #plt.plot((omegas-np.max(omegas)/2), sideband_spectrum/np.max(sideband_spectrum))
    ax[0].set_xlim(-5,5)
#    plt.plot(wrange_zpl, spec_dat_cav.real/np.max(spec_dat_cav.real))
    #plt.plot(wrange_zpl, spec_dat_em.real/np.max(spec_dat_em.real),'.')
    #plt.show()
    list_conv = sc.signal.fftconvolve(
        sideband_spectrum, np.flip(spec_dat_em,axis=0), mode='same') 
    #sc.ndimage.convolve1d(sideband_spectrum.real,spec_dat_em, mode='nearest')
    # print(list_conv)omegas - np.max(omegas)/2,
    ax[0].plot(omegas-np.max(omegas)/2, list_conv.real/np.max(list_conv.real), '-.')
 # plt.plot(-omegas+np.max(omegas)/2, sideband_spectrum.real/0.007 , '.')
    #plt.ylim(0,1)
    plt.show()

    # plt.plot(wrange_zpl, spec_dat_em)
    # plt.plot(omegas- np.max(omegas)/2, sideband_spectrum)
    # plt.show()

