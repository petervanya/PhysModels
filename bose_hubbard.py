#!/usr/bin/env python
"""Usage:
    main.py <M>

Bose-Hubbard model, according to Zhang et al (2001)
H = -J \sum_<ij> a_i^+ a_j + a_j^+ a_i + U/2 \sum_i n_i (n_i-1)

Arguments:
    <M>      Number of sites [default: 3]

06/09/15
"""
from docopt import docopt
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from math import sqrt
import logging


def gen_new_vec(old):
    M = len(old)
    N = sum(old)
    ln = max(np.nonzero(old[:-1])[0])  # last non-zero position for 0..M-1
    new = np.zeros(M)
    new[:ln] = old[:ln]
    new[ln] = old[ln] - 1
    new[ln+1] = N - sum(new[:ln+1])
    return new

def gen_basis(M, filling):
    """Function to generate all basis states"""
    vec = np.array([N] + [0]*(M-1))
    logging.info("Start:" + str(vec))
    basis = []
    basis.append(vec)
    while vec[-1] != N:
        newvec = gen_new_vec(vec)
        basis.append(newvec)
        vec = newvec
    return basis

def int_H(basis, U=1):
    """Interaction part of the Hamiltonian"""
    D = len(basis)
#    return [(i, i, U/2*sum([n*(n-1) for n in basis[i]])) for i in range(D)]
    return np.diag([U/2*sum([n*(n-1) for n in basis[i]]) for i in range(D)])

def tag(vec):
    """Tag function"""
    T = 0
    for i in range(len(vec)):
        T += sqrt(100*i+3)*vec[i]
    return T

def kin_H(basis, J=1):
    """Kinetic matrix elements of the Hamiltonian"""
    D = len(basis)
    #H_kin = []
    H_kin = np.zeros((D, D))
    tags = [tag(vec) for vec in basis]
    sorted_tags, ind = zip(*[(i[1], i[0]) \
                       for i in sorted(enumerate(tags), key=lambda x: x[1])])
    
    for col in range(D):
        vec = basis[col]
        matel = 0
        logging.info("col:" + str(col) + "|ket>:" + str(vec))
        coeffs = []
        for i in range(M):     # SHOULD USE PBC? USING NOW
            coeff = sqrt((vec[(i+1)%M] + 1)*vec[i])  # forward hopping
            coeffs.append(coeff)
            if coeff != 0:
                hop_vec = np.copy(vec)
                hop_vec[(i+1)%N] += 1
                hop_vec[i] -= 1
                logging.info("H|ket>:" + str(coeff) + str(hop_vec))
                index = sorted_tags.index(tag(hop_vec))
                H_kin[index, col] = -J*coeff

            coeff = sqrt((vec[(i-1)%N] + 1)*vec[i])  # backward hopping
            coeffs.append(coeff)
            if coeff != 0:
                hop_vec = np.copy(vec)
                hop_vec[(i-1)%N] += 1
                hop_vec[i] -= 1
                logging.info("H|ket>:" + str(coeff) + str(hop_vec))
                index = sorted_tags.index(tag(hop_vec))
                H_kin[index, col] = -J*coeff
        logging.info(str(coeffs))
#        H_kin.append((i, j, -J*matel))
    return H_kin

def diag_H(H):
    """Diagonalise the Hamiltonian"""
    vals, vecs = eigh(H)
    return vals[0], vecs[0]



if __name__ == "__main__":
    args = docopt(__doc__)
    logging.basicConfig(filename="BH.log", filemode="w", level=logging.DEBUG)
    M = int(args["<M>"])
    filling = 1     # N/M
    N = M*filling   # number of atoms
    U = 1.0
    J = 1.0
    
    basis = gen_basis(M, filling)
    logging.info("Basis size: " + str(len(basis)))
    
    H_int = int_H(basis, U)
    H_kin = kin_H(basis, J)
    logging.info("H_kin:\n" + str(H_kin))
    logging.info("H_pot:\n" + str(H_int))
#    plt.spy(H_kin)
#    plt.show()
    vals, vecs = diag_H(H_kin + H_int)
    print vals
    print vecs
    print np.dot(vecs[0], vecs[0])


    
    
