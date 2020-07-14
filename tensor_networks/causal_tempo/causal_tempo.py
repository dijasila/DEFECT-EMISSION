import numpy as np
import scipy as sc
from scipy import integrate
from mpmath import coth
import matplotlib.pyplot as plt


def spre(op):
    return np.kron(op, np.eye(op.shape[0]))


def spost(op):
    return np.kron(np.eye(op.shape[0]), op.T)


def sprepost(A, B):
    #the Liouville space representation of A.R.B
    return np.kron(A, B.T)


class make_bath(object):

    def __init__(self, J_data, T=0):

        #initialise class with the sepctral density data.
        # default uses the J(w) = J_normal(w)/w**2, where J_normal
        # is the more standard open systems definition of the spectral density
        self.wrange, self.J = J_data

        #the list for the eta function:
        self.eta = []
        self.C = []

        self.J_func = None
        self.Temp = T

        self.kpoints = 1
        self.dt = 1
        self.trange = np.array([])

        self.norm = 'physics'

    def eta_num(self, t, norm='physics'):

        if norm == 'physics':
            J_func = self.wrange**(-2)*self.J

        elif norm == 'chemistry':
            J_func = self.J

        #construct data from the numerical spectral density
        if self.Temp == 0:

            re_data = J_func * (1-np.cos(self.wrange * t))
            im_data = J_func * (np.sin(self.wrange * t) - self.wrange * t)

        else:
            re_data = J_func * (1-np.cos(self.wrange * t)) * \
                coth(0.5 * self.wrange/self.Temp)
            im_data = J_func * (np.sin(self.wrange * t) - self.wrange * t)

        #remove the (bullshit) divergence at zero frequency:
        re_data[0] = 0
        im_data[0] = 0

        # set the frequency increment
        dw = self.wrange[1]-self.wrange[0]
        #do the integral using a trapezoidal rule
        re_eta = np.trapz(re_data, dx=dw)
        im_eta = np.trapz(im_data, dx=dw)
        #eta_point = np.sum(data) * dw#np.trapz(data, dx = dw)

        return re_eta + 1j * im_eta

    def construct_eta(self, points, dt):

        # establish the list of data points we wish to calculate the influence tensor:
        self.trange = np.arange(0, (points+3)*dt, dt)
        if self.norm == 'physics':
            print('Using the Physicists normalisation of the spectral density.')

        elif self.norm == 'chemistry':
            print('Using the Chemists normalisation of the spectral density.')
        # iterate over the time range to calculate eta:
        for t in self.trange:
            self.eta.append(self.eta_num(t, norm=self.norm))

    def corr_func(self, t):
        #calcualate the regular correlation function:
        # Int[J(w)(cos(wt)Coth(0.5 * w/T) - I * sin(wt))
        #construct data from the numerical spectral density
        if self.norm == 'physics':
            J_func = self.wrange**(-2)*self.J

        elif self.norm == 'chemistry':
            J_func = self.J

        if self.Temp == 0:

            data = J_func * (np.cos(self.wrange * t) -
                             1j * (np.sin(self.wrange * t)))

        else:

            data = J_func * ((np.cos(self.wrange * t)) *
                             coth(0.5 * self.wrange/self.Temp) - 1j * np.sin(self.wrange * t))

        # set the frequency increment
        dw = self.wrange[1]-self.wrange[0]
        #do the integral using a trapezoidal rule
        corr_point = np.trapz(self.wrange, data, dx=dw)
        return corr_point

    def check_sample_rate(self, file_name=None):
        # first calculate the regular correlation function
        for t in self.trange:
            self.C.append(self.corr_func(t))

        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(self.trange, np.real(self.C), '.-')
        ax[1].plot(self.trange, np.imag(self.C), '.-')
        if file_name == None:
            plt.show()
        else:
            plt.savefig(file_name)
        plt.close()


class causal_tempo_prop(object):

    def __init__(self, prec=1, hilbert_dim=2, coupop=np.array([[1, 0], [0, 0]])):

        #the dimension of the system Hilbert space:
        self.dim = hilbert_dim
        self.d = self.dim**2  # shortcut for the Lioville space dimension

        #set the initial system state
        self.initial = np.array(self.d)
        #the instaneous state of the reduced system:
        self.state = np.array(self.d)

        #initialise the system Hamiltonian
        self.system_Ham = np.zeros((self.dim, self.dim), dtype=np.complex128)
        #Its corresponding evolution (super)operator
        self.free_evolve = np.array((self.d, self.d))

        #define the bath object:
        self.bath = None

        #keep track of the instaneous timestep:
        self.point = 0
        #the size of the timestep in question:
        self.dt = 1
        #number of points iterated:
        self.kpoints = 1
        #the precision of the to keep singular values at.
        self.precision = prec
        #the max number of memory steps before the cut-off
        self.dkmax = 0

        # define the system-environment coupling operator
        self.coupop = coupop

        #define the commutator and anticommutator in Lioville space
        self.Ocomm = spre(self.coupop)-spost(self.coupop)
        self.Oanti = spre(self.coupop) + spost(self.coupop)

        # store the state and corresponding time step
        self.state_list = []
        self.trange = []
        self.expect_data = []

    def free_evolution(self):
        L0 = (spre(self.system_Ham) - spost(self.system_Ham))
        self.free_evolve = sc.linalg.expm(- 1j * L0 * self.dt/2).T

    def lind_vec(self, rate, op):
        O = op.conj().T @ op
        L = 2 * sprepost(op, op.conj().T) - spre(O) - spost(O)
        L *= 0.5 * rate
        return L

    def add_dissipation(self, rate_and_op):
        #function to add a dissipator to the free evolution of the system
        # takes a list of rate and operator, builds a lindblad dissipator
        # and adds it to the Liouvillian.

        #start with the free evolution:
        L0 = -1j * (spre(self.system_Ham) - spost(self.system_Ham))

        #append the dissipators contained in lind_list
        L = L0
        for rate, op in rate_and_op:
            L += self.lind_vec(rate, op)

        #redefine the free evolution operator:
        self.free_evolve = sc.linalg.expm(L * self.dt * 0.5).T

    def construct_bath_data(self, J_data, T=0, norm='physics'):
        #construct the bath information using the spectral density data
        # and temperature info as input.
        self.bath = make_bath(J_data, T)
        self.bath.norm = norm
        if self.dkmax == 0:
            self.bath.construct_eta(self.kpoints, self.dt)
        else:
            self.bath.construct_eta(self.dkmax+3, self.dt)

    def I_dk(self, dk):

        Op = self.Oanti.diagonal()
        Om = self.Ocomm.diagonal()

        if dk == 0:
            eta_dk = self.bath.eta[1]
            Idk = np.exp(-Om * (eta_dk.real * Om + 1j * eta_dk.imag * Op))
        else:
            eta_dk = self.bath.eta[dk + 1] - 2 * self.bath.eta[dk] + self.bath.eta[dk-1]
            Idk = np.exp(-np.outer((eta_dk.real * Om +
                                    1j * eta_dk.imag * Op), Om))

        return Idk

    def get_state(self):
        #out put the instantaneous reduced state:
        rvec = self.mps.readout() @self.free_evolve
        self.state = rvec.reshape(self.dim, self.dim)

    def get_expect(self, expec):

        for O in expec:
            O_dat = []
            for r in self.state_list:
                O_dat.append(np.trace(O@r))

            self.expect_data.append(O_dat)

    def b_tensor(self, dk):
        # construct the 4-leg tensors for the MPO 
        # indices for the MPS go as {S, N, W, E}
        if dk == 0:
            #dot in the free evolution operator and the zero time difference operator
            Ifree = self.I_dk(0) * (self.free_evolve @ self.free_evolve)

            #add the dummy indices and reshape into the indices [NSWE]
            tens = np.diag(Ifree.reshape(self.d**2)).reshape(4 * [self.d])
            tens = np.swapaxes(tens, 1, 2)

        # if dk == 1:
        #     #dot in the free evolution operator and the zero time difference operator
        #     Ifree = self.I_dk(1) * self.I_dk(0) * \
        #         (self.free_evolve @ self.free_evolve)

        #     #add the dummy indices and reshape into the indices [NSWE]
        #     tens = np.diag(Ifree.reshape(self.d**2)).reshape(4 * [self.d])
        #     tens = np.swapaxes(tens, 1, 2)


        else:
            # find the influence for step dk
            tens = self.I_dk(dk)
            #add the dummy indices and reshape into the indices [NSWE]
            tens = np.diag(tens.reshape(self.d**2)).reshape(4 * [self.d])
            tens = np.swapaxes(tens, 1, 2)

        if dk == self.point or (dk == self.dkmax and self.dkmax > 0):
            # if the mpo is at an end-site then sum over eastern leg and replace with
            # a 1D dummy index.
            return np.expand_dims(np.dot(tens, np.ones(tens.shape[3])), -1)
        else:
            return tens


