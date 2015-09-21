#!/usr/bin/env python
"""Usage:
    heisenberg.py <N> [-J <J>] [--nup <nup>] [--lanczos [-e <evals>]]

Exact diagonalisation of Heisenberg model of N spins on open chain
H = -J\sum_<ij> S_i*S_j (1 = spin up; 0 = spin down)

Arguments:
    <N>             Number of spins in the chain [default: 3]

Options:
    -J <J>          Coupling parameter J [default: 1]
    --nup <nup>     Number of spins up [default: 2]
    --lanczos       Use Lanczos diag'n method 
    -e <evals>      Specify number of Lanczos eigenvalues [default: 1]

pv278@cam.ac.uk, 07/09/15
"""
from docopt import docopt
import numpy as np
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh as lanczos
from math import sqrt
from itertools import permutations
import time


def tag(vec):
    """Hash a state"""
    return sum([sqrt(100*i+3)*vec[i] for i in range(len(vec))])

def apply_H(basis):
    """Create matrix elements"""
    D, M = np.array(basis).shape
    tags = [tag(vec) for vec in basis]
    H = np.zeros((D, D))
    for col in range(D):
        vec = basis[col]
        c = 0
        for i in range(M-1):
            c += Sz(vec[i])*Sz(vec[i+1])
            H[col, col] = c
        for i in range(M-1):
            new_vec = np.copy(vec)
            # S_i^+ * S_{i+1}^-
            new_vec[i] += 1
            new_vec[i+1] -= 1
            if tag(new_vec) in tags:
                ind = tags.index(tag(new_vec))
                H[ind, col] += 0.5
            # S_i^- * S_{i+1}^+
            new_vec = np.copy(vec)
            new_vec[i] -= 1
            new_vec[i+1] += 1
            if tag(new_vec) in tags:
                ind = tags.index(tag(new_vec))
                H[ind, col] += 0.5
    return H

def filter(vec):
    """get states that passed through S+ * S-"""
    vec = np.array(vec)
    if (vec > 1).any() or (vec < 0).any():
        return None
    else:
        return list(vec)

def Sz(i):
    if i == 0:
        return 0.5
    if i == 1:
        return -0.5


if __name__ == "__main__":
    args = docopt(__doc__)
#    print args
    N = int(args["<N>"])
    Nup = int(args["--nup"])
    J = float(args["-J"])

    # ===== construct basis
#    start = [0, 0, 1]                     # g.s. = -1.0
#    start = [0, 0, 1, 1]                  # g.s. = -1.61602540378
#    start = [0, 0, 0, 1, 1, 1]
#    start = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # g.s. = -4.25803520728    
    firstvec = [0]*(N-Nup) + [1]*Nup
    basis = [firstvec]
    for i in permutations(firstvec):
        if list(i) not in basis:
            basis.append(list(i))
    print "Basis:\n", basis
    print "Basis size:", len(basis)
    
    # ===== construct H
    H = apply_H(basis)
    print "H:\n", H
    
    # ===== diagonalise H
    if args["--lanczos"]:
        Nevals = int(args["-e"])
        t1 = time.time()
        vals, vecs = lanczos(H, Nevals)
        t2 = time.time()
        dt = t2 - t1
    else:
        t1 = time.time()
        vals, vecs = eigh(H)
        t2 = time.time()
        dt = t2 - t1
    print "energies:\n", vals
    print "states:\n", vecs
    print "Diag time: %.4f s" % dt
    
