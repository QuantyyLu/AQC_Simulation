import sys
import numpy
import ctypes
from pyscf import lib 
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf import scf, gto, mcscf,fci
from scipy.sparse.linalg import eigs
import scipy.linalg as LA
from itertools import combinations
from cmath import exp
from numpy.linalg import norm as mynorm
from scipy.special import comb
import time
import os
import socket
import openfermion
import numpy as np
import scipy as sp
import datetime
import time
import cirq
import itertools

from openfermion.config import *
from openfermion.ops import *
from openfermion.transforms import *
from openfermion.utils import *
from openfermion.linalg import *
from cirq.ops import *
from cirq.circuits import *
from cirq import Simulator
from cirq.study import *
from numpy import dot, conjugate, zeros
from math import pi, sqrt, sin, ceil
from random import randrange
from utils import *



#******************************************************************************
#           parameter setting
#******************************************************************************
#setting of the mol
# mol_str = 'O 0 0 0.5488; O 0 0 -0.5488'


mol_str = 'N 0 0 0; N 0 0 1.134'
FCI_basis_str = 'sto-3g'# (6,(3,3))
# FCI_basis_str = 'sto-3g'
# FCI_basis_str = 'ccpv-dz'# (24,(3,3))
# FCI_basis_str = 'sto-3g'
# FCI_basis_str = '6-31g'#  (14,(3,3))
ASCI_basis_str = FCI_basis_str
symmetry = True
# ncas, nelecas = (14,(3,3))
ncas, nelecas = (6,(3,3))
spin = 0





"""
Quantum circuit simulation program for adiabatic state preparation
"""

# Simulation parameters
n_qubits          = 12
n_electrons       = 6
mapping_method    = "JWT"
trotter_term_ordering = "Magnitude"

# Simulation parameters
# Wave function configuration
integral_filename = ''

# ASP conditions
# Weight strategy should be "Lin", "Sin", "Squ", "SinCub", or "Cub"
evolution_time       = 20
weight_strategy      = "Lin"
trotter_order        = 1
num_steps            = ceil(evolution_time*2.0)

#
host = socket.gethostname()
start_time = time.time()
current_datetime = datetime.datetime.now()
print(" Quantum simulation on {}".format(host)," starts at {}".format(current_datetime))

nmo = n_qubits // 2

strategies = ['Lin', 'Sin', 'Squ', 'SinCub', 'Cub', 'Const']
if weight_strategy not in strategies:
    raise ValueError(' Unidentified strategy for nonzero terms')


##########################################################################################
# set occupations
initial_occupation = [0.0]*n_qubits
for i in range(n_electrons):
    initial_occupation[i] = 1

# Hamiltonian term classification
# nuclear_repulsion, oneint, twoint, = get_integrals(integral_filename, n_qubits)
mc, nuclear_repulsion, oneint, twoint = pyscf_get_integrals(mol_str,ASCI_basis_str,spin,symmetry,ncas,nelecas)
fock_ops, corr_ops = get_initial_hamiltonian(oneint, twoint, initial_occupation,
                                             n_qubits, n_electrons)


# Define time-independent (initial) and time-dependent Hamiltonians
h_fock_jw = jordan_wigner(fock_ops)
h_corr_jw = jordan_wigner(corr_ops)
h_fock_sparse = jordan_wigner_sparse(fock_ops, n_qubits = n_qubits)
h_corr_sparse = jordan_wigner_sparse(corr_ops, n_qubits = n_qubits)

full_hamiltonian_sparse = h_fock_sparse + h_corr_sparse

# Hartree-Fock energy
hf_pointer = 0
for i in range(n_qubits):
    if initial_occupation[i] == 1:
        hf_pointer += 2 ** (n_qubits - i-1)
hmat_dim = 2 ** (n_qubits)
hf_state = zeros(hmat_dim, dtype=np.complex64)
hf_state[hf_pointer] = 1.0+0.0j
hf_energy = expectation(full_hamiltonian_sparse, hf_state).real
print("\nE(HF) = {:.10f}".format(hf_energy.real)," Hartree")
fci_energy, fci_state = jw_get_target_state_at_particle_number( full_hamiltonian_sparse, n_electrons, hf_state)
# fci_info = sp.sparse.linalg.eigs(full_hamiltonian_sparse,k=1)
# fci_energy, fci_state  =  fci_info[0][0], fci_info[1][:,0]
print("E(FCI_final) = {:.10f}".format(fci_energy.real)," Hartree")


#----------------------------------------------------------------------------------------------------#
# Generate quantum circuit for the time-independent Hamiltonian
time_for_single_trotter = evolution_time / num_steps


# Preliminary steps: Use time-independent Hamiltonian only
print("\n   Time  s(t)   E(ASP)/Hartree    E(Exact)/Hartree   Tro-error  total-error  ad-error")
print("-----------------------------------------------------------------------------------------------------")

asp_wf_exact_curr = hf_state
asp_wf_sim_curr = hf_state

h_ins_jw = h_fock_jw
h_ins_sparse = h_fock_sparse

compare_with_ins_ground_state = False
# T_list = [10,20,40,80,150,300,500,700,1000]
T_list = [100]
# n_list = [10, 20,40,80,100,150,200,250,300,350,400,450,500,700,1000]
# n_list = [200]
dt = 0.2
# n_list = [200]
alpha = 12
len_power = 1
power = [1.5**(i-len_power//2) for i in range(len_power)]
# power = [1]
ASP_error = []
Exact_error = []
Trotter_error = []
with open("Data/GQSP_mole.txt","w") as File1:
    for evolution_time in T_list:
        K = 4 * int(evolution_time)
    if True:
            # tmp = 4 * sqrt(num_steps)
            num_steps = 5 * evolution_time
            # T_list = [tmp * power[i] for i in range(len_power)]
            # for evolution_time in T_list:
            if True:
                # File1.write(f"{evolution_time}   {num_steps}\n")
                # File1.write(f"{evolution_time}\n")
                time_for_single_trotter = evolution_time / num_steps
                asp_wf_exact_curr = hf_state
                asp_wf_sim_curr = hf_state
                overlap_sim_fci = dot(fci_state, conjugate(asp_wf_sim_curr))
                sq_overlap_sim_fci_ini = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real

                # Time-dependent steps: Use both time-independent and time-dependent Hamiltonians
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                for i_step in range(num_steps):
                    td_hamiltonian_weight = get_hamiltonian_weight(i_step+1, num_steps, weight_strategy)

                    # Calculate instantaneous eigenstate
                    h_ins_jw = h_fock_jw + td_hamiltonian_weight * h_corr_jw
                    h_ins_sparse = h_fock_sparse + td_hamiltonian_weight * h_corr_sparse
                    if compare_with_ins_ground_state:
                        fci_state = eigs(h_ins_sparse,k=1)[1][:,0]

                    U = GQSP_sparse(K, h_ins_sparse, time_for_single_trotter, alpha)
                    asp_sim_result = U @ asp_wf_sim_curr
                    asp_wf_sim_next = normalize_wave_function(asp_sim_result)[0]
                    # Exact ASP time evolution
                    asp_wf_exact_next = sp.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse, asp_wf_exact_curr)
                    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)[0]
                    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
                    e_asp = expectation(h_ins_sparse, asp_wf_sim_next).real 
                    e_asp_exact = expectation(h_ins_sparse, asp_wf_exact_next).real
                    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
                    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
                    overlap_sim_fci = dot(asp_wf_sim_next, conjugate(fci_state))
                    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
                    overlap_exact_fci = dot(fci_state, conjugate(asp_wf_exact_next))
                    sq_overlap_exact_fci = dot(overlap_exact_fci,conjugate(overlap_exact_fci)).real
                    current_time = time_for_single_trotter * (i_step+1)
                    if i_step % 10 == 0:
                        print("  {:>.3f}".format(evolution_time)," {:.3f}".format(td_hamiltonian_weight),
                        "   {:.10f}".format(e_asp),
                        "  {:.10f}".format(e_asp_exact),\
                        "      {:.6f}".format(1-sq_overlap_sim_exact),"       {:.6f}".format(1-sq_overlap_sim_fci),\
                        "       {:.6f}".format(1-sq_overlap_exact_fci) )
                    asp_wf_exact_curr = asp_wf_exact_next
                    asp_wf_sim_curr = asp_wf_sim_next

                sq_overlap_sim_fci_fin = sq_overlap_sim_fci

                print("")
                print(" SUMMARY OF THE QUANTUM CIRCUIT SIMULATION")
                print("")
                print("  E(ASP,Ini) = {:.10f}".format(hf_energy),"Hartree")
                print("  E(ASP,Fin) = {:.10f}".format(e_asp),"Hartree")
                print("  E(Full-CI) = {:.10f}".format(fci_energy),"Hartree")
                print("")
                print("1-|<ASP,Ini|ins_Full-CI>|^2 = {:.6f}".format(1-sq_overlap_sim_fci_ini))
                print("1-|<ASP,Fin|ins_Full-CI>|^2 = {:.6f}".format(1-sq_overlap_sim_fci_fin))

                elapsed_time = time.time() - start_time
                print("\nNormal termination. Wall clock time is {}".format(elapsed_time) + "[sec]")
                ASP_error.append(1-sq_overlap_sim_fci)
                Exact_error.append(1-sq_overlap_exact_fci)
                Trotter_error.append(1-sq_overlap_sim_exact)
            File1.write(f"{time_for_single_trotter}  {td_hamiltonian_weight}   {1-sq_overlap_sim_fci}   {1-sq_overlap_exact_fci}   {1-sq_overlap_sim_exact}\n")