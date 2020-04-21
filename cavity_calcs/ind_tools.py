import numpy as np
import scipy as sc
import qutip as qt
import time
import os


def a_elem_opt(jj,kk, eigproj, eigop, eigR0):
    """
    code that extracts caclulates the elements of the 'A' matrix, that is therm
    eigenspace overlaps.
    jj, kk = the elements of the a matrix
    eigproj = the trace projector in the liovillian eigenspace
    eigop = superopertor in eigenspace
    eigR0 = initial vectorised density operator in lio eigenspace
    """

    element = (eigproj)[jj][0]*eigop[jj][kk]*eigR0[kk]

    return element[0]


def a_mat_build(vals, P, rho0, op, proj):
    """
    Build the eigen elements matrix. This requires the inputs:

    vals = eigenvalues
    P = eigenvectors matrix
    Pinv = the corresponding inverse
    rho0 = vectorised initial density operator
    op = regression annihilation operator in super-operator space
    proj = the projector which calculates the trace in density operator space.

    """

    #Transform initial, regression operator, and trace projector to eig basis
    eigR0 = np.linalg.solve(P, rho0)
    op_prime = op @ P
    eigop =np.linalg.solve(P, op_prime)
    eigproj = P.T @ proj

    #construct the overlap matrix
    dim = vals.shape[0]
  # st = time.time()
    mat_opt = np.array([[a_elem_opt(jj, kk, eigproj, eigop, eigR0) for jj in range(dim)]
        for kk in range(dim)])
  # en = time.time()
    #print('Optimised matrix build',en - st)
    return mat_opt


def alist_elem(jj, fulleigproj,eigR0):

    element = fulleigproj[jj][0]*eigR0[jj]
    return element

def a_list_build(vals, P, rho0, op,fullproj):

    #Transform initial, regression operator, and trace projector to eig basis
    eigR0 = np.linalg.solve(P,rho0)
    fulleigproj = P.T @ fullproj

    #construct the overlap matrix
    dim = vals.shape[0]
   # st = time.time()
    mat_opt = np.array([alist_elem(jj, fulleigproj, eigR0) for jj in range(dim)])
   # en = time.time()
    #print('Optimised matrix build',en - st)
    return mat_opt



def Time_dom_Indist(vals, A, Alist):
    """
    A function for calculating the indistinguishability and power for an arbitrary
    Liovillian. The required input is:
    vals = eignevalues for a Liovillian
    A = the eigen-overlap matrix
    """
    #conjugate the eigenvalues for convenience
    cvals = np.conj(vals)

    # build caclulation matrices:
    Lam = np.nan_to_num(np.array([[1/(jj + kk) for jj in vals]
    for kk in cvals]))

    # calculate the numerator
    numerator = np.trace(
    np.linalg.multi_dot([A.T.conj(), Lam, A,Lam.T]))

    Lam = np.nan_to_num(np.array([[1/(jj * (ll +jj)) for jj in vals]
    for ll in vals]))

    #Power calculation:
    val_vec = np.nan_to_num(1/vals)
    power = np.sum(Alist*val_vec)

    Umat = np.outer(Alist,Alist)
    denominator = np.trace(Umat @ Lam)
    #print('The estimated numerator is ', numerator)
    
    # power = Alist.dot(val_vec.reshape([val_vec.shape[0],1]).T)
    # denominator = (power)**2

    power = -np.sum(A.T @ val_vec)

    #find the indistinguishability
    indist = numerator/denominator#(power**2)

    return [power, indist]









######### LEGACY! ##########
# def Indist(vals, A):
#     """
#     A function for calculating the indistinguishability and power for an arbitrary
#     Liovillian. The required input is:
#     vals = eignevalues for a Liovillian
#     A = the eigen-overlap matrix
#     """
#     #conjugate the eigenvalues for convenience
#     cvals = np.conj(vals)

#     # build caclulation matrices:
#     Lam = np.nan_to_num(np.array([[1/(jj + kk) for jj in vals] for kk in cvals]))

#     # calculate the required matrix:
#     mats = np.linalg.multi_dot([A, Lam, A.T.conj(), Lam.T])

#     #find the numerator:
#     numerator = 2 * (np.pi**2) * np.real(np.trace(mats))

#     #Power calculation:
#     val_vec = np.nan_to_num(1/vals)
#     power = np.real(-np.pi * np.sum(A.T @ val_vec))

#     #find the indistinguishability
#     indist = numerator/(power**2)

#     return [power, indist]

