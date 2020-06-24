import numpy as np
import scipy as sc


class mpo_site(object):
    def __init__(self, tens=None):
        self.Sdims, self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens

    def update(self, tens=None):

        self.Sdims, self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens



class mps_site(object):

    def __init__(self, tens=None):

        self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens

    def update(self, tens=None):
        self.Ndims, self.Wdims, self.Edims = tens.shape

        self.m = tens

    def contract_with_mpo_site(self, mposite):
        # contract a four legged with a three mps.
        #first get the NOrthern leg of the mps on the left and turn into a matrix
        tens0 = self.m.reshape(self.Ndims, -1)
        # then rearrange the mpo in a similar fashion (south leg on right)
        mpo_mat = np.swapaxes(mposite.m, 1, 3).reshape(-1, mposite.Ndims)
        #contract to obtain with indices [NW2E2,W1E1]
        tens0 = mpo_mat @ tens0
        #Want the MPS in the form [N, W1xW2, E1xE2]:
        tens0 = np.swapaxes(tens0.reshape(-1, mposite.Edims, mposite.Wdims*self.Wdims, self.Edims), 1, 2)
        tens0 = tens0.reshape(-1, mposite.Wdims*self.Wdims,mposite.Edims*self.Edims)

        #update the mps site:
        self.update(tens0)


class mpo_block(object):
    def __init__(self):

        self.Nsites = 0
        self.sites = []

    def insert_site(self, position, tensor):

        self.sites.insert(position, mpo_site(tens=tensor))

        self.Nsites += 1

    def print_shape(self):
        print('mpo dimensions are:')
        for site in self.sites:

            print(site.m.shape)
    
    def reverse_mpo(self):
        self.sites.reverse()

        for site in self.sites:
            site.update(tens = np.swapaxes(site.m, 2,3))


class mps_block(object):

    def __init__(self, prec=1):

        self.Nsites = 0

        self.sites = []
        self.precision =prec
        #print(self.precision)


    def insert_site(self, position, tensor):

        self.sites.insert(position, mps_site(tens=tensor))

        self.Nsites += 1



    def reverse_mps(self):
        self.sites.reverse()

        for site in self.sites:
            site.update(tens = np.swapaxes(site.m, 1,2))



    def truncate_bond(self, k):
        # only allow truncation if enough sites exhist.
        if k<1 or k > self.Nsites-1: return 0
        
        # rearrange the site at k-1 into a matrix and do a singular value decomposition 
        theta = self.sites[k -1].m.reshape(-1, self.sites[k-1].Edims)
        U, S, _ = sc.linalg.svd(theta, full_matrices=False, lapack_driver='gesvd')
        #print(S)
        #keep only singular values above the desired precision.
        # chi = (S/np.max(S)) > self.precision
        # extract = sum(chi) #sums all the Trues and gives the number of elements to keep.
        # print(extract)
        try: chi = next(i for i in range(len(S)) if S[i]/max(S) < self.precision)
        #If no singular values are small enough then keep em all
        except(StopIteration):chi = len(S)
        # extract the desired colummns in the unitary
        Uprime = U[:,0:chi]
        # now we update the orininal matrix in the truncated basis:
        theta = Uprime.T.conj() @ theta

        #unitary becomes the site at k-1
        self.sites[k-1].update(Uprime.reshape(-1,self.sites[k-1].Wdims, chi))
        

        
        #multipy the site a k with the truncated theta
        theta = theta @ (np.swapaxes(self.sites[k].m,0,1).reshape(self.sites[k].Wdims,-1))
        theta = np.swapaxes(theta.reshape(chi, self.sites[k].Ndims, -1),0,1)

        #update the kth site:
        self.sites[k].update(theta)


    def trunc_sweep(self, k, mpo=None):
       
        if type(mpo) == mpo_block:
            self.sites[0].contract_with_mpo_site(mpo.sites[0])
            for jj in range(1, k+1):
                self.sites[jj].contract_with_mpo_site(mpo.sites[jj])
                self.trunc_sweep(jj)
        
        else:
            for jj in range(1, k+1):
                self.truncate_bond(jj)


    def contract_with_mpo(self, mpoblock, orth_centre = None):
        #no truncations for a single site.
        if self.Nsites == 1:
            self.sites[0].contract_with_mpo_site(mpoblock.sites[0])
            return 0 
        
        if self.precision == np.inf:
        # if we don't want to do any truncations:
            for jj in range(1, self.Nsites):
                self.sites[jj].contract_with_mpo_site(mpoblock.sites[jj])
        
        
        else:
        #truncation is done in two halves divided by the centre of orthogonality.
        #define an centre of orthogonality or use the default centre of the chain.
            if orth_centre==None: orth_centre = int(np.ceil(0.5 * self.Nsites))
            
            # sweet up to the the OC contracting in mpos
            self.trunc_sweep(orth_centre - 1, mpoblock)

            #flip the mps and mpos
            self.reverse_mps()
            mpoblock.reverse_mpo()

            #sweep up to the OC contracting in mpos:        
            self.trunc_sweep(self.Nsites - orth_centre-1, mpoblock)

            # truncate the bond linking the two halves
            self.truncate_bond(self.Nsites-orth_centre)

            #run a sweep across the whole chain
            self.trunc_sweep(self.Nsites)
            
            #flip back
            self.reverse_mps()
            mpoblock.reverse_mpo()
            
            #one final sweep.
            self.trunc_sweep(self.Nsites)

        # self.sites[0].contract_with_mpo_site(mpoblock.sites[0])
        # for jj in range(1, self.Nsites):
        #     self.sites[jj].contract_with_mpo_site(mpoblock.sites[jj])


    def readout(self):

        if self.Nsites == 1:
            # sum the east and south legs.
            return np.sum(self.sites[0].m, (1, 2))
        else:
            #prepare vector at end site by summing over North and East legs:
            out = np.sum(self.sites[self.Nsites-1].m,(0,-1))

            #iteratively contract in the sites summing over the Northern legs:
            for jj in range(self.Nsites-2):
                out = np.sum(self.sites[self.Nsites-2-jj].m,0) @ out
            # at the final site keep the Northern index and sum over the eastern.
            # Northern index becomes the index of the reduced density operator.
            out = np.sum(self.sites[0].m,1) @ out
            return out

    def contract_end(self):
        #sum over the Northern and eastern legs of the final site:
        end_vec = np.sum(self.sites[-1].m,(0,2))
        #contract with next site along
        end_up_date = np.expand_dims(self.sites[-2].m @ end_vec,-1)
        #and add eastern dummy index.

        #update site -2:
        self.sites[-2].update(end_up_date)

        #delete old last site and awdjust number of sites:
        del self.sites[-1]
        self.Nsites = self.Nsites-1

    def bonddims(self):
    #returns a list of the bond dimensions along the mps
        bond=[]
        for site in self.sites:
            bond.append([site.Ndims, site.Wdims, site.Edims])
        return bond

    def totsize(self):
        #returns the total number of elements (i.e. complex numbers) which make up the mps
        size=0
        for site in self.sites: size = size + site.SNdim*site.Wdim*site.Edim
        return size

    def print_shape(self):
        print('mps dimensions are:')
        for site in self.sites:
            
            print(site.m.transpose(1,0,2).shape)
