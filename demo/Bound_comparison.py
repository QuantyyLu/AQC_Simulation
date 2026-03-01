import multiprocessing
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
import io
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
from cmath import sqrt
import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix, random as sparse_random
import scipy.sparse as sps
from math import sqrt
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.special import jn
from scipy.integrate import quad
from scipy.sparse.linalg import eigs
import scipy.sparse.linalg as sp_la

from itertools import product
from functools import reduce
from openfermion.ops import QubitOperator
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh
from utils import *
from generate_hamiltonian import Molecular


def run_and_optimize_for_g(g, n_qubits=5, h=2, J=1, initial_T=200.0, initial_step_inverse=30.0):
    """
    Runs the entire two-phase optimization for a given g and returns the final ratios.
    """
    print(f"\n{'='*25} RUNNING FOR g = {g} {'='*25}")
    
    # --- Return values initialized to NaN ---
    energy_gap_ratio = np.nan
    trotter_bound_ratio = np.nan

    # --- Initial physical setup ---
    qubits = cirq.LineQubit.range(n_qubits)
    mol_op = create_hamiltonian(n_qubits, h, g, J, periodic=False)
    z_op = QubitOperator()
    for i in range(n_qubits): z_op += QubitOperator(f'X{i}', -1.0)
    fock_ops = z_op
    mol_sparse = get_sparse_operator(mol_op, n_qubits=n_qubits)
    fock_sparse = get_sparse_operator(fock_ops, n_qubits=n_qubits)

    _, state0 = eigsh(fock_sparse, k=1, which='SA')
    state0 = state0[:, 0]
    fci_energy, fci_state = eigsh(mol_sparse, k=1, which='SA')
    fci_energy, fci_state = fci_energy[0], fci_state[:,0]

    print(f"E(HF) = {np.vdot(state0, mol_sparse @ state0).real:.10f} Hartree")
    print(f"E(FCI_final) = {fci_energy.real:.10f} Hartree")
    print("-" * 80)

    # --- ONE-TIME PRE-CALCULATION ---
    whether_bound = True
    calculate_alpha = 'compute'
    precomputed_coeffs, alpha_i, alpha_f, g_i, g_f, final_alpha = None, None, None, None, None, None

    if whether_bound:
        if calculate_alpha == 'compute':
            precomputed_coeffs = precompute_alpha_coeffs_direct(fock_ops, mol_op, "Data/alpha.txt")
        elif calculate_alpha == 'read':
            precomputed_coeffs = load_alpha("Data/alpha.txt")
        if precomputed_coeffs:
            alpha_i = get_alpha_at_time_t(0, precomputed_coeffs)
            alpha_f = get_alpha_at_time_t(1, precomputed_coeffs)
        g_i, _, _, _, _ = calculate_energy_gap(fock_sparse)
        g_f, _, _, _, _ = calculate_energy_gap(mol_sparse)
        print("Pre-calculation complete.")
        print("-" * 80)

    # --- Optimization Parameters ---
    T = initial_T
    step_inverse = 2
    target_error = 0.001
    tolerance = 0.00001
    max_adjust_steps = 10

    # --- Phase 1: Tune T ---
    print("PHASE 1: Adjusting T to target ad_error = 0.001")
    T_low, T_high, binary_search_mode = 0.0, float('inf'), False
    i = 0
    while i < max_adjust_steps or binary_search_mode:
        num_steps = int(T * step_inverse)
        if num_steps < 1: T *= 2; continue
        ad_error, sim_error, _, _ = run_asp_simulation(T, num_steps, fock_ops, mol_op, fock_sparse, mol_sparse, state0, fci_state, qubits, precomputed_coeffs)
        
        # Print the status of the current step
        print(f"Step {i+1}: T = {T:.4f}, step_inverse = {step_inverse:.2f} -> ad_error = {ad_error:.6f}, sim_error = {sim_error:.6f}") # <-- ADDED LINE

        if abs(ad_error - target_error) < tolerance: break
        T, T_low, T_high, binary_search_mode = renew_T(ad_error, target_error, T, T_low, T_high, binary_search_mode, i, max_adjust_steps, tolerance)
        i += 1
    T_final = T
    
    T = 20
    step_inverse = initial_step_inverse  # Reset step_inverse for Phase 2
    # --- Phase 2: Tune step_inverse ---
    print("\nPHASE 2: Adjusting step_inverse to target sim_error = 0.001")
    step_inverse_low, step_inverse_high, binary_search_mode_s = 0.0, float('inf'), False
    i = 0
    while i < max_adjust_steps or binary_search_mode_s:
        num_steps = int(T * step_inverse)
        if num_steps < 1: step_inverse *= 2; continue
        ad_error, sim_error, gap_history, final_alpha = run_asp_simulation(T, num_steps, fock_ops, mol_op, fock_sparse, mol_sparse, state0, fci_state, qubits, precomputed_coeffs)
        
        # Print the status of the current step
        print(f"Step {i+1}: T = {T:.4f}, step_inverse = {step_inverse:.2f} -> ad_error = {ad_error:.6f}, sim_error = {sim_error:.6f}") # <-- ADDED LINE
        
        if abs(sim_error - target_error) < tolerance: break
        step_inverse, step_inverse_low, step_inverse_high, binary_search_mode_s = renew_step_inverse(sim_error, target_error, step_inverse, step_inverse_low, step_inverse_high, binary_search_mode_s, i, max_adjust_steps, tolerance)
        i += 1 # Don't forget to increment the counter
    step_inverse_final = step_inverse

    num_steps = int(T_final * step_inverse_final)
    ad_error, sim_error, gap_history, final_alpha = run_asp_simulation(T_final, num_steps, fock_ops, mol_op, fock_sparse, mol_sparse, state0, fci_state, qubits, precomputed_coeffs)

    # --- Final Calculations for this g ---
    if gap_history:
        a = min(gap_history[0], gap_history[-1])
        b = min(gap_history)
        if b > 1e-9: energy_gap_ratio = a**2 / b**3
    
    if whether_bound and final_alpha is not None and g_i is not None and g_f is not None:
        num_steps_final = int(T_final * step_inverse_final)
        time_for_single_trotter_final = T_final / num_steps_final
        former_bound = final_alpha**2 * time_for_single_trotter_final**4
        our_bound = (alpha_i/g_i)**2 + (alpha_f/g_f)**2 * time_for_single_trotter_final**2
        if our_bound > 1e-12: trotter_bound_ratio = former_bound / our_bound
    
    energy_gap_ratio = a**2 * T_final**2

    return energy_gap_ratio, trotter_bound_ratio, T_final, a, b

# ==============================================================================
#                      NEW MAIN SCRIPT TO RUN THE EXPERIMENT
# ==============================================================================

if __name__ == "__main__":
    # Define the list of g values you want to test
    # g_values_to_test = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    g_values_to_test = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # g_values_to_test = [0.4]
    # 1. Create a Pool of worker processes.
    #    By default, multiprocessing.Pool() uses all available CPU cores.
    #    The 'with' statement ensures the pool is properly managed and closed.
    print(f"Starting parallel computation for {len(g_values_to_test)} g-values on {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        # 2. Use pool.map to apply the function to each g-value in parallel.
        #    - The first argument is the function to run.
        #    - The second argument is the list of inputs to iterate over.
        #    - pool.map blocks until all processes are finished and returns a list
        #      of results in the same order as the input list.
        result_list = pool.map(run_and_optimize_for_g, g_values_to_test)
    print(result_list)
    # Store results in a dictionary
    results = {}



    for g_val, (eg_ratio, tb_ratio, T_final, a, b) in zip(g_values_to_test, result_list):
        results[g_val] = {'energy_gap_ratio': eg_ratio, 'trotter_bound_ratio': tb_ratio, 'meta_ratio':tb_ratio/eg_ratio, 'T_final':T_final, 'a':a, 'b':b}

    File = open("Data/bound.txt", "w")
    # --- Print a final summary table of all results ---
    print(f"\n\n{'='*20} FINAL SUMMARY {'='*20}")
    print(f"{'g Value':<10} | {'Energy Gap Ratio (a/b)':<25} | {'Trotter Bound Ratio':<25}| {'Meta Ratio':<25}" + " | {'T_final':<25} | {'a':<25} | {'b':<25}")
    print(f"{'-'*10} | {'-'*25} | {'-'*25}")
    for g_val, res in results.items():
        egr_str = f"{res['energy_gap_ratio']:.6f}" if not np.isnan(res['energy_gap_ratio']) else "N/A"
        tbr_str = f"{res['trotter_bound_ratio']:.6f}" if not np.isnan(res['trotter_bound_ratio']) else "N/A"
        meta_ratio_str = f"{res['meta_ratio']:.6f}" if not np.isnan(res['meta_ratio']) else "N/A"
        T_final_str = f"{res['T_final']:.6f}" if not np.isnan(res['T_final']) else "N/A"
        a_str = f"{res['a']:.6f}" if not np.isnan(res['a']) else "N/A"
        b_str = f"{res['b']:.6f}" if not np.isnan(res['b']) else "N/A"  

        print(f"{g_val:<10.2f} | {egr_str:<25} | {tbr_str:<25}| {meta_ratio_str:<25} | {T_final_str:<25} | {b_str:<25} | {a_str:<25}")
        File.write(f"{g_val:<10.2f}  {egr_str:<25}  {tbr_str:<25} {meta_ratio_str:<25}  {T_final_str:<25}  {b_str:<25}  {a_str:<25}\n")
    File.close()

    

# if __name__ == "__main__":
#     # Define the list of g values you want to test
#     g_values_to_test = [0.001]
    
#     # Store results in a dictionary
#     results = {}

#     for g_val in g_values_to_test:
#         # Call the main function for each g and store the returned ratios
#         eg_ratio, tb_ratio = run_and_optimize_for_g(g_val, n_qubits=5, h=2, J=1, initial_T=200, initial_step_inverse=37.5)
#         results[g_val] = {'energy_gap_ratio': eg_ratio, 'trotter_bound_ratio': tb_ratio, 'meta_ratio':tb_ratio/eg_ratio**2}

#     # --- Print a final summary table of all results ---
#     print(f"\n\n{'='*20} FINAL SUMMARY {'='*20}")
#     print(f"{'g Value':<10} | {'Energy Gap Ratio (a/b)':<25} | {'Trotter Bound Ratio':<25}| {'Meta Ratio':<25}")
#     print(f"{'-'*10} | {'-'*25} | {'-'*25}")
#     for g_val, res in results.items():
#         egr_str = f"{res['energy_gap_ratio']:.6f}" if not np.isnan(res['energy_gap_ratio']) else "N/A"
#         tbr_str = f"{res['trotter_bound_ratio']:.6f}" if not np.isnan(res['trotter_bound_ratio']) else "N/A"
#         meta_ratio_str = f"{res['meta_ratio']:.6f}" if not np.isnan(res['meta_ratio']) else "N/A"
#         print(f"{g_val:<10.2f} | {egr_str:<25} | {tbr_str:<25}| {meta_ratio_str:<25}") 