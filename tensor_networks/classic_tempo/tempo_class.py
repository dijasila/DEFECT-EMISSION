import numpy as np
import scipy as sc
from scipy import integrate
from mps_mpo_funcs import mps_block, mpo_block


def spre(op):
    return np.kron(op, np.eye(op.shape[0]))


def spost(op):
    return np.kron(np.eye(op.shape[0]), op.T)



class tempo_prop(object):

    def __init__(self, prec=1, hilbert_dim=2):
        self.dim = hilbert_dim

        self.d = self.dim**2
        self.initial = np.array(self.d) 
        self.state = np.array(self.d)

        
        self.system_Ham = np.zeros((self.dim, self.dim), dtype = np.complex128)
        self.free_evolve = np.array((self.d, self.d))
        
        self.point = 0
        self.dt = 1
        self.precision = prec
        self.dkmax = 0

        self.eta = np.array([])
        self.b_ten = None

        self.coupop = np.array([[1,0],[0,-1]])
        
        self.Ocomm = spre(self.coupop)-spost(self.coupop)
        self.Oanti = spre(self.coupop) + spost(self.coupop)
       
        self.state_list = []
        self.trange = []



    def free_evolution(self):
        L0 = (spre(self.system_Ham) - spost(self.system_Ham))
        self.free_evolve = sc.linalg.expm(- 1j * L0 *self.dt/2).T

    def I_dk(self, dk):


        Op = self.Oanti.diagonal()
        Om = self.Ocomm.diagonal()

        if dk == 0:
            eta_dk = self.eta[1]
            Idk = np.exp(-Om * (eta_dk.real * Om + 1j * eta_dk.imag * Op))
        else:
            eta_dk = self.eta[dk + 1] - 2 * self.eta[dk] + self.eta[dk-1]
            Idk = np.exp(-np.outer((eta_dk.real * Om + 1j * eta_dk.imag * Op), Om))

        return Idk


    def get_state(self):
        #out put the instantaneous reduced state:
        rvec = self.mps.readout() @self.free_evolve
        self.state = rvec.reshape(self.dim, self.dim)




    def b_tensor(self, dk):
        # construct the 4-leg tensors for the MPO
        if dk == 1:
            #dot in the free evolution operator and the zero time difference operator
            Ifree = self.I_dk(1) * self.I_dk(0) * (self.free_evolve @ self.free_evolve)

            #add the dummy indices and reshape into the indices [NSWE]
            tens = np.diag(Ifree.reshape(self.d**2)).reshape(4 *[self.d])
            tens = np.swapaxes(tens, 1, 2)
           
        else:
            # find the influence for step dk
            tens = self.I_dk(dk)
            #add the dummy indices and reshape into the indices [NSWE]
            tens = np.diag(tens.reshape(self.d**2)).reshape(4 * [self.d])
            tens = np.swapaxes(tens, 1,2)
           
        if dk == self.point or (dk==self.dkmax and self.dkmax>0):
            # if the mpo is at an end-site then sum over eastern leg and replace with 
            # a 1D dummy index.
            return np.expand_dims(np.dot(tens, np.ones(tens.shape[3])), -1)
        else:
            return tens
        


    def prep(self):
        # initialise mps and mpo blocks
        self.mps = mps_block(prec=self.precision)
        self.mpo = mpo_block()

        #build free propagator
        self.free_evolution()

        #initialise read-out lists for the state and times
        self.state_list.append(self.initial.reshape(self.dim, self.dim))
        self.trange.append(0)

        #prepare the initial ADT
        r_del = self.initial @ self.free_evolve
        r_prep = self.I_dk(0) * r_del
        #turn into 3 leg tensor, two of which have 1D
        itens = np.expand_dims(np.expand_dims(r_prep, 1), 2)

        #insert ADT into the mps
        self.mps.insert_site(0, itens)

        #update point
        self.point = 1
        #insert first mpo
        self.mpo.insert_site(0, self.b_tensor(1))
        
        #output reduced state at point 1
        self.get_state()
        # append state to the data lists
        self.state_list.append(self.state)
        self.trange.append(self.dt * self.point)


    def prop(self, kpoints=1):
        

        for k in range(kpoints):
            
            #contract mps state with mpo doing svd and truncating 
            self.mps.contract_with_mpo(self.mpo)

            #insert a new site at the first place in the mps
            self.mps.insert_site(0, np.expand_dims(np.eye(self.d), 1))
            
            # update the point
            self.point = self.point + 1
            #calculate the reduced state
            self.get_state()

            # this needs to change when have finite memory time:
            if self.point< self.dkmax+1 or self.dkmax==0:
                # while growing update the last site from three legs, to 
                # the 4-leg version we desire for the next propagation step
                self.mpo.sites[-1].update(self.b_tensor(self.point - 1))

                #insert the new three leg end mpo.
                self.mpo.insert_site(self.point - 1, self.b_tensor(self.point))
            else:
                # after this growth stage the mpo remains the same, but now we can
                # sum over the last site in the mps
                self.mps.contract_end()
                
            self.state_list.append(self.state)
            self.trange.append(self.dt * self.point)
            #print(self.state.shape)
