import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import causal_tempo as tempo
import spectral_density_prep as SD

#important conversion parameters:
hbarc_cmtoev = 1.9746e-05  # eVcm
hbar_meVps = 6.582E-7
kb = 8.617E-5  # eV K^-1

# import and construct the spectral data for HBN
sd_file = 'data.xlsx'
phon_width = 200 * hbarc_cmtoev
SD_sampling = 120000
dt = 0.00001
w_max = (2 * np.pi/dt)*hbarc_cmtoev

#construct a spectral density from this data:
omegas, spectral_data = SD.gen_spectral_dens(sd_file, phon_width, SD_sampling, w_max)
J_data = [omegas, spectral_data]

#define the  desired system raising and lowering operators:
sp = np.array([[0, 1], [0, 0]])
sm = sp.T


#Find the reorganisation energy from the spectral density:
reorg = np.trapz(omegas * spectral_data, dx=(omegas[1]-omegas[0]))
print('The reorgansiation energy is: ', reorg)

#set the system  parameters and Hamiltonian:
eps = 0#reorg
V = 0
#choose a spontaneous emission rate of 100 ps:
T1 = 1
Gamma = nv.GamOpt

Hsys = eps * sp @ sm + V * (sp + sm)

#initialise TEMPO:
sys = tempo.causal_tempo_prop()

#set the number of points and time steps
sys.kpoints = 7
sys.dt = 0.001

# #set a memory cut-off
sys.dkmax = 0

#add the main bath
sys.construct_bath_data(J_data, norm='chemistry')
#sys.bath.check_sample_rate() #uncomment to check how well the correlation function has been sampled


#define the free Hamiltonian:
sys.system_Ham = Hsys  # Uncomment if there is no dissipation
sys.free_evolution()
sys.point =sys.kpoints


class mps_site(object):

    def __init__(self, tens=None):
        #set up such that West is the local time.
        self.Sdims, self.Ndims, self.Wdims = tens.shape

        self.m = tens

    def update(self, tens=None):
        self.Sdims, self.Ndims, self.Wdims = tens.shape

        self.m = tens

    def contract_with_mpo_site(self, mpo):
        #contract 4 leg mpo tensor with mps site.
        #need to contract West leg of mps with East leg
        # of the mpo:
        tens0 = self.m.transpose(1,0,2).reshape(self.Wdims,-1)

        #rearrange the mpo into a matrix:
        mpo_mat = mpo.m.reshape(-1, mpo.Edims)

        #Contract West and East legs:
        tens0 = mpo_mat @ tens0 # now have indices [SNW, SN]

        #split indices:
        tens0 = tens0.reshape(mpo.Sdims, mpo.Ndims, mpo.Wdims, self.Sdims, self.Ndims)

        # get in the right order [SSNNW]:
        tens0 = tens0.transpose(0,3,1,4,2)

        #lump the South and North legs together:
        tens0 = tens0.reshape(mpo.Sdims * self.Sdims, mpo.Ndims * self.Ndims, mpo.Wdims)

        #and update the mps site
        self.update(tens0)




class mps_block(object):

    def __init__(self, prec=1):

        self.Nsites = 0

        self.sites = []
        self.precision = prec

    def insert_site(self, position, tensor):

        self.sites.insert(position, mps_site(tens=tensor))

        self.Nsites += 1

    def print_shape(self):
        print('Indices are laid out as [S,W,E]')
        print('mps dimensions are:')
        for site in self.sites:

            print(site.m.shape)


class mpo_site(object):
    def __init__(self, tens=None):
        self.Sdims, self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens

    def update(self, tens=None):

        self.Sdims, self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens

    def reduce_dims(self, index_to_reduce = 'North'):

        if index_to_reduce == 'North':
            #swap North axis to the right and contract with vector
            int_tens = np.dot(self.m.transpose(0,2,3,1), np.ones(self.Ndims))
            #expand the North dimensions
            int_tens = np.expand_dims(int_tens, 1)

        elif index_to_reduce== 'South':
            #contract with vector:
            int_tens = np.dot(self.m.transpose(1,2,3,0), np.ones(self.Sdims))
            #expand South dimensions:
            int_tens = np.expand_dims(int_tens, 0)

        #update the mpo
        self.update(int_tens)

    # def readout(self, state, dim_trace):

    #     #iteratively save





class mpo_block(object):
    def __init__(self):

        self.Nsites = 0
        self.sites = []

    def insert_site(self, position, tensor):

        self.sites.insert(position, mpo_site(tens=tensor))

        self.Nsites += 1

    def print_shape(self):
        print('MPO indices are ordered [S,N,W,E].')
        print('mpo dimensions are:')
        for site in self.sites:

            print(site.m.shape)

    def reduce_mpo(self):

        #remove final site in mpo
        del self.sites[-1]

        #reduce the North leg of the last site:
        self.sites[-1].reduce_dims(index_to_reduce='North')
        self.Nsites -= 1

# The MPS boundary (i.e. the density matrix index) is taken to be the
# western leg. The MPS will initially look like:
#                (West)
#                  |     |     |            |
#       (South) -- b1 -- b2 -- b3 -- ... -- bk (North -> this would be the TEMPO mps boundary)
#


#initialise the mps block.
mps = mps_block()

# build a list of tensors to the desired time point:
# indices go as [SNWE].
tensors = [sys.b_tensor(dk) for dk in range(sys.point)]



# want to remove the Eastern leg of the initial mps:
for n, tens in enumerate(tensors):
    #get the Eastern leg on the right, then dot with a vector of the same dimension.
    #It already is!
    int_tens = np.dot(tens, np.ones(tens.shape[1]))

    #add into the mps:
    mps.insert_site(n, int_tens)


# We can remove the Northern leg of the end_site, since it is redundant.
# Get the northern leg on the right using transpose and then contract with vector of ones
int_tens = np.dot(mps.sites[-1].m.transpose(0,2,1), np.ones(mps.sites[-1].Ndims))
#add a dummy leg on the Northern index.
mps.sites[-1].update( np.expand_dims(int_tens, 1))


# build the mpo;
mpo = mpo_block()

# MPO block has one less site for each time step.
#drop the last site in the mpo block:
del tensors[-1]

for n, tens in enumerate(tensors):
    mpo.insert_site(n, tens)

#function to remove a leg of the MPO site and replace with a dummy index.
#can discard the Southern leg of the first site, and Northern leg of Northern site.

mpo.sites[0].reduce_dims(index_to_reduce = 'South')
mpo.sites[-1].reduce_dims(index_to_reduce='North')
# ^^^ Note that we have to do this for each time step.

#set the current time step.
current_time = 1

for k in range(sys.kpoints-2):
    # print('timestep =', k)
    for nn in range(mpo.Nsites):

        #contract in the east leg of each MPO with West leg of MPS.
        #starting with the current timestep:
        mps.sites[nn + current_time].contract_with_mpo_site(mpo.sites[nn])

    #sort out the mpo for the next time step by discarding the last site
    # and reducing it to a three leg tensor (by discarding the Northern leg):
    mpo.reduce_mpo()

    # propoagate forward one timestep.
    current_time += 1

r0 = 0.5 * np.ones(4)
sys.initial = r0
# #extract the prepared initial state:
# r_del = sys.initial @ sys.free_evolve
# r_prep = sys.I_dk(0) * r_del
state_list = [r0]


#contract the South leg of the first site with the initial condition:

ADT = mps.sites[0].m.transpose(1,2,0) @ r0 #this tensor now has the dimension [N,W]

#remove the Northern bond to get the latest time:
state_list.append(ADT.sum(0))


#now repeat and contract the North leg of the current site with South of the next
for ii in range(1,mps.Nsites):
    # for the next time step, sum over the western leg, keeping
    # the Northern index of the ADT:
    reduced_ADT = ADT.sum(1)

    #get the South leg on the right, and turn into a matrix with indices [NxW, S]:
    mat = mps.sites[ii].m.transpose(1, 2, 0).reshape(-1, mps.sites[ii].Sdims)

    # contract S and N indices:
    ADT = mat @ reduced_ADT  # has shape [NW]

    #reshape into a matrix:
    ADT = ADT.reshape(mps.sites[ii].Ndims, -1)

    #sum over Northern index to get current state:
    state_list.append(ADT.sum(0))

    # # remove Western leg in preparation of the next time step:
    # reduced_ADT = ADT.sum(1)


for r in state_list: print(r)

#### SADLY I THINK THIS IS WRONG.... ####
#repeat procedure but trace over the Western leg rather than sum:
trace_state_list = [r0]

#trace projection:
proj = np.array([1,0,0,1])

# initial ADT:
ADT = mps.sites[0].m.transpose(1, 2, 0) @ r0

#remove the Northern bond to get the latest time:
trace_state_list.append(ADT.sum(0))


#now repeat and contract the North leg of the current site with South of the next
for ii in range(1, mps.Nsites):
    # for the next time step, sum over the western leg, keeping
    # the Northern index of the ADT:
    reduced_ADT = ADT @ proj


    #get the South leg on the right, and turn into a matrix with indices [NxW, S]:
    mat = mps.sites[ii].m.transpose(1, 2, 0).reshape(-1, mps.sites[ii].Sdims)

    # contract S and N indices:
    ADT = mat @ reduced_ADT  # has shape [NW]

    #reshape into a matrix:
    ADT = ADT.reshape(mps.sites[ii].Ndims, -1)

    #sum over Northern index to get current state:
    trace_state_list.append(ADT.sum(0))

    # # remove Western leg in preparation of the next time step:
    # reduced_ADT = ADT.sum(1)

# for r in trace_state_list: print(r)
