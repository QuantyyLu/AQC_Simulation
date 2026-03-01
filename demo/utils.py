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
from scipy.linalg import eigh



#---------- FUNCTION GET_INTEGRALS ----------#
def get_integrals(filename, n_qubits):
    nlines = 0
    intdata = []
    for line in open(filename):
        items = line.split()
        intdata.append(items)
        nlines +=1

    nuclear_repulsion = 0
    oneint = zeros((n_qubits, n_qubits))
    twoint = zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for i in range(nlines):
        if len(intdata[i]) == 1:
            # Nuclear repulsion or frozen core energy
            nuclear_repulsion = float(intdata[i][0])
        elif len(intdata[i]) == 3:
            # One electron integrals
            iocc = int(intdata[i][0])
            avir = int(intdata[i][1])
            intvalue = float(intdata[i][2])
            oneint[iocc, avir] = intvalue
        elif len(intdata[i]) == 5:
            # Two electron integrals
            iocc = int(intdata[i][0])
            jocc = int(intdata[i][1])
            bvir = int(intdata[i][2])
            avir = int(intdata[i][3])
            intvalue = float(intdata[i][4])
            twoint[iocc, avir, jocc, bvir] = intvalue*2
    return nuclear_repulsion, oneint, twoint


#---------- FUNCTION GET_CLASSIFIED_HAMILTONIAN
def get_initial_hamiltonian(oneint, twoint, ini_occ, n_qubits, nelec):
    #
    fock_ini = FermionOperator()
    corr_ini = FermionOperator()
    #
    for i in range(n_qubits):
        for a in range(n_qubits):
            if i == a and ini_occ[i] == 1:
                fock_ini += FermionOperator(((a, 1), (i, 0)), oneint[i, a])
            else:
                corr_ini += FermionOperator(((a, 1), (i, 0)), oneint[i, a])
    # Two electron terms:
    for i in range(n_qubits):
        for j in range(n_qubits):
            for a in range(n_qubits):
                for b in range(n_qubits):
                    if ini_occ[i] == 1 and ini_occ[j] == 1 and i!= j:
                        if (i == a and j == b) or (i == b and j == a and (i-j)%2 == 0):
                            fock_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                        0.5*twoint[i, a, j, b])
                        else:
                            corr_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                        0.5*twoint[i, a, j, b])
                    else:
                        corr_ini += FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)),
                                                    0.5*twoint[i, a, j, b])

    return fock_ini, corr_ini


#---------- FUNCTION S_SQUARED_FERMION_DM ----------#
def s_squared_fermion_dm(n_qubits):
    # generate S^2 Fermionic operator in DM.
    """
    Notes:
    S(i,j)^2 = S_z(i)*S_z(j) + (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2
    """
    n_molorb = int(n_qubits / 2)
    s_squared_operator = FermionOperator()

    for iorb in range(n_molorb):
        ia = 2 * iorb
        ib  = 2 * iorb + 1
        for jorb in range(n_molorb):
            ja = 2 * jorb
            jb  = 2 * jorb + 1

            # S_z(i) * S_z(j) terms
            s_squared_operator +=  0.25 * FermionOperator(((ia, 1), (ia, 0), (ja, 1), (ja, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ia, 1), (ia, 0), (jb, 1), (jb, 0)))
            s_squared_operator += -0.25 * FermionOperator(((ib, 1), (ib, 0), (ja, 1), (ja, 0)))
            s_squared_operator +=  0.25 * FermionOperator(((ib, 1), (ib, 0), (jb, 1), (jb, 0)))
            # (S_+(i) * S_-(j) + S_-(i) * S_+(j))/2 terms
            s_squared_operator +=  0.50 * FermionOperator(((ia, 1), (ib, 0), (jb, 1), (ja, 0)))
            s_squared_operator +=  0.50 * FermionOperator(((ib, 1), (ia, 0), (ja, 1), (jb, 0)))

    return s_squared_operator

#---------- IMPORTED FUNCTION: JW_NUMBER_INDICES ----------#
def jw_number_indices(n_electrons, n_qubits):
    """Return the indices for n_electrons in n_qubits under JW encoding

    Calculates the indices for all possible arrangements of n-electrons
        within n-qubit orbitals when a Jordan-Wigner encoding is used.
        Useful for restricting generic operators or vectors to a particular
        particle number space when desired

    Args:
        n_electrons(int): Number of particles to restrict the operator to
        n_qubits(int): Number of qubits defining the total state

    Returns:
        indices(list): List of indices in a 2^n length array that indicate
            the indices of constant particle number within n_qubits
            in a Jordan-Wigner encoding.
    """
    occupations = itertools.combinations(range(n_qubits), n_electrons)
    indices = [sum([2 ** n for n in occupation])
               for occupation in occupations]
    return indices

#--------- IMPORTED FUNCTION:
def jw_get_target_state_at_particle_number(sparse_operator, particle_number, ref_wf):
    """Compute ground energy and state at a specified particle number.

    Assumes the Jordan-Wigner transform. The input operator should be Hermitian
    and particle-number-conserving.

    Args:
        sparse_operator(sparse): A Jordan-Wigner encoded sparse matrix.
        particle_number(int): The particle number at which to compute the ground
            energy and states

    Returns:
        ground_energy(float): The lowest eigenvalue of sparse_operator within
            the eigenspace of the number operator corresponding to
            particle_number.
        ground_state(ndarray): The ground state at the particle number
    """
    num_states = 4
    n_qubits = int(np.log2(sparse_operator.shape[0]))
    # Get the operator restricted to the subspace of the desired particle number
    restricted_operator = jw_number_restrict_operator(sparse_operator, particle_number, n_qubits)
    # Compute eigenvalues and eigenvectors
    if restricted_operator.shape[0] - 1 <= 1:
        # Restricted operator too small for sparse eigensolver
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    else:
        eigvals, eigvecs = sp.sparse.linalg.eigsh(restricted_operator,
                                                     k=num_states,
                                                     which='SA')
    # Expand the state
    target_so = 0
    target_st = 0
    for istate in range(num_states):
        curr_state = eigvecs[:, istate]
        expanded_state = zeros(2 ** n_qubits, dtype=complex)
        expanded_state[jw_number_indices(particle_number, n_qubits)] = curr_state
        overlap = dot(expanded_state, conjugate(ref_wf))
        sq_overlap = dot(overlap, conjugate(overlap)).real
        if sq_overlap > target_so:
            target_st = istate
            target_so = sq_overlap
    target_ene = eigvals[target_st]
    target_state = eigvecs[:, target_st]
    target_wf = zeros(2 ** n_qubits, dtype=complex)
    target_wf[jw_number_indices(particle_number, n_qubits)] = target_state
    return target_ene, target_wf


#---------- FUNCTION TRANSFORM_QUBOP_TO_CIRQ ----------#
def transform_qubop_to_cirq(operator, theta,qubits):
    """
    Transform qubit operator to cirq circuit
    """
    len_op = len(operator)
    if len_op != 0:
        for iop in range(len_op):
            pauli_op = operator[iop]
            if pauli_op[1] == "X":
                yield H(qubits[pauli_op[0]])
            elif pauli_op[1] == "Y":
                yield rx(-pi/2).on(qubits[pauli_op[0]])
        #
        for iop in range(len_op-1):
            control_qubit = operator[iop][0]
            target_qubit = operator[iop + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])
        #
        yield rz(theta*2).on(qubits[operator[len_op-1][0]])
        #
        for iop in reversed(range(len_op-1)):
            control_qubit = operator[iop][0]
            target_qubit = operator[iop + 1][0]
            yield CNOT(qubits[control_qubit],qubits[target_qubit])

        for term in range(len_op):
            pauli_op = operator[term]
            if pauli_op[1] == "X":
                yield H(qubits[pauli_op[0]])
            elif pauli_op[1] == "Y":
                yield rx(pi/2).on(qubits[pauli_op[0]])


#---------- FUNCTION DISCARD_ZERO_IMAGINARY ----------#
def discard_zero_imaginary(qubit_operator):
    for key in qubit_operator.terms:
        qubit_operator.terms[key] = float(qubit_operator.terms[key].real)
    qubit_operator.compress()
    return qubit_operator


#---------- FUNCTION SUB_CIRCUITS ----------#
def sub_circuit(qubit_operator, trotter_order, trotter_term_ordering, time_for_single_trotter,qubits):   
    qubit_operator = discard_zero_imaginary(qubit_operator)
    if trotter_term_ordering == "Magnitude":
        qubit_operator_sorted = sorted(list(qubit_operator.terms.items()),
                                       key=lambda x:abs(x[1]), reverse=True)
        num_qubit_terms = len(qubit_operator_sorted)
        if trotter_order == 2:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta,qubits)
            for iterm in reversed(range(num_qubit_terms)):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta,qubits)
        else:
            for iterm in range(num_qubit_terms):
                op = qubit_operator_sorted[iterm][0]
                theta = qubit_operator_sorted[iterm][1] * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta,qubits)
    else:
        term_ordering = sorted(list(qubit_operator.terms.keys()))
        if trotter_order == 2:
            for op in term_ordering:
                theta = qubit_operator.terms[op] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta,qubits)
            for op2 in reversed(term_ordering):
                theta = qubit_operator.terms[op2] * 0.5 * time_for_single_trotter
                yield transform_qubop_to_cirq(op2, theta)
        else:
            for op in term_ordering:
                theta = qubit_operator.terms[op] * time_for_single_trotter
                yield transform_qubop_to_cirq(op, theta,qubits)
    # yield X(qubits[-1])
    # yield X(qubits[-1])




#---------- FUNCTION GET_HAMILTONIAN_WEIGHT ----------#
def get_hamiltonian_weight(i_step, num_steps, weight_strategy):
    curr_position = i_step / num_steps
    # Default is a linear function
    h_weight = curr_position
    #
    if weight_strategy == "Sin":
        h_weight = sin(pi * curr_position / 2.0)
    elif weight_strategy == "Squ":
        h_weight = 3 * curr_position**2 - 2 * curr_position**3
    elif weight_strategy == "SinCub":
        h_weight = (sin(pi * curr_position / 2.0))**3
    elif weight_strategy == "Cub":
        h_weight = 6 * curr_position**5 - 15 * curr_position**4 + 10 * curr_position**3
    elif weight_strategy == 'Const':
        h_weight = 1
    return h_weight


#---------- FUNCTION NORMALIZE_WAVE_FUNCTION ----------#
def normalize_wave_function(wave_function):
    wave_function_norm = dot(wave_function, conjugate(wave_function)).real
    wave_function = wave_function / sqrt(wave_function_norm)
    return wave_function, wave_function_norm


#---------- FUNCTION PRINT_WAVE_FUNCTION ----------#
def print_wave_function(wave_function):
    wfdim = len(wave_function)
    thresh = 0.0001
    for idet in range(wfdim):
        det = wave_function[idet]
        if dot(det, conjugate(det)) > thresh:
            print(det, '  {:13b}'.format(idet))


def get_mol_info(mol_str, basis, spin):
    mol = gto.M(
        atom=mol_str,
        basis=basis,
        spin=spin
    )
    num_orbitals_up = mol.nao_nr()
    num_orbitals_down = num_orbitals_up
    num_elec_up = (mol.nelectron + spin)//2
    num_elec_down = (mol.nelectron - spin)//2
    mol_info = [num_orbitals_up, num_orbitals_down, num_elec_up, num_elec_down]
    return mol_info


def build_CASCI_mol(mol_str, basis, symmetry, ncas, nelecas):
    # mol_str = 'H 0 0 0; F 0 0 1.1'
    mol = gto.M(
        atom=mol_str,  # in Angstrom
        basis=basis,
        symmetry=symmetry,
        verbose=3
    )
    myhf = scf.RHF(mol)
    myhf.kernel()
    mc = mcscf.CASCI(myhf, ncas, nelecas).run()
    e_CASCI = mc.e_tot
    e_HF = myhf.e_tot
    wf_CASCI = mc.ci
    wf_CASCI = wf_CASCI.ravel()
    print('CASCI mol built')
    return mc, e_HF, e_CASCI, wf_CASCI

def get_eri(mc):
        h1, h0 = mc.get_h1eff ()
        # h0 = 0
        # h1 = numpy.zeros((6,6))
        # h2 = mc.get_h2eff () 
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nelecas = mc.nelecas
        mo_coeff = mc.mo_coeff[:,ncore:nocc]
        # h2 = ao2mo.full(mc.mol, mo_coeff, max_memory=mc.max_memory, verbose=mc.verbose)
        h2 = ao2mo.full(mc.mol, mo_coeff, max_memory=mc.max_memory,verbose=mc.verbose, compact = False)
        return h0, h1, h2

def eri_to_int(n_qubits, h1, h2):
    oneint = zeros((n_qubits, n_qubits))
    twoint = zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for i in range(n_qubits):
        for a in range(n_qubits):
            if (i-a)%2 == 0:
                oneint[i][a] = h1[i//2][a//2]
                for j in range(n_qubits):
                     for b in range(n_qubits):
                        if (j-b)%2 == 0:
                            twoint[i][a][j][b] = h2[i//2*n_qubits//2+a//2][j//2*n_qubits//2+b//2]
    return oneint, twoint

def eri_to_int4(n_qubits, h1, h2):
    oneint = zeros((n_qubits, n_qubits))
    twoint = zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for i in range(n_qubits):
        for a in range(n_qubits):
            if (i-a)%2 == 0:
                oneint[i][a] = h1[a//2][i//2]
                for j in range(n_qubits):
                     for b in range(n_qubits):
                        if (j-b)%2 == 0:
                            twoint[i][a][j][b] = h2[a//2][b//2][j//2][i//2]
    return oneint, twoint

def pyscf_get_integrals(mol_str,ASCI_basis_str,spin,symmetry,ncas,nelecas):
    mol_info = get_mol_info(mol_str,ASCI_basis_str,spin)
    mc, e_HF, e_CASCI, wf_CASCI = build_CASCI_mol(mol_str,ASCI_basis_str,symmetry,ncas,nelecas)
    # mc1, e_HF1, e_CASCI1, wf_CASCI1= build_CASCI_mol(mol_str,ASCI_basis_str,symmetry,ncas1,nelecas1)
    # wf_FCI, e_FCI = build_FCI_mol(mol_str,ASCI_basis_str,symmetry,ncas,nelecas,frozen)
    n_qubits = 2*ncas
    h0 ,h1, h2 = get_eri(mc)
    oneint, twoint = eri_to_int(n_qubits, h1, h2)
    return mc, h0, oneint, twoint

def generate_initial_occupation(n_qubits, n_electrons, initial_config='RHF', num_bspair=0):
    initial_occupation = [0.0]*n_qubits
    if initial_config == "RHF":
        for i in range(n_electrons):
            initial_occupation[i] = 1
    if initial_config == "BS":
        n_doc = n_electrons - (num_bspair * 2)
        if n_doc != 0:
            for iorb in range(n_doc):
                initial_occupation[iorb] = 1
        for bspair in range(num_bspair):
            initial_occupation[n_doc + 2*bspair] = 1
            initial_occupation[n_electrons + 1 + 2*bspair] = 1
    return initial_occupation

def GQSP(K,H,dt,alpha):
    j = complex(0,1)
    n = H.shape[0]
    H = H/alpha
    H_gqsp = jn(0,-1*alpha*dt)* complex(1,0) * np.eye(n)
    chebyshev_T_list = [np.eye(n),H]
    H_gqsp += 2 * j * jn(1,-1*alpha*dt) * H
    for k in range(2,K+1):
        chebyshev_Tk = 2 * H @ chebyshev_T_list[k-1] - chebyshev_T_list[k-2]
        chebyshev_T_list.append(chebyshev_Tk)
        H_gqsp += 2 * j**k * jn(k,-1*dt*alpha) * chebyshev_Tk
    return H_gqsp

def GQSP_sparse(K, H, dt, alpha):
    """
    GQSP函数的稀疏优化版本。
    输入H应为scipy稀疏矩阵格式(csc, csr等)。
    输出为标准的numpy稠密矩阵。
    """
    # 确保输入是CSC或CSR格式，这两种格式对于算术运算效率较高
    # --- 修正 2: 将所有 sp. 替换为 sps. ---
    if not sps.isspmatrix_csc(H) and not sps.isspmatrix_csr(H):
        H = H.tocsc() # 转换为CSC格式

    j = complex(0, 1)
    n = H.shape[0]
    
    # 使用稀疏单位矩阵
    identity_sparse = sps.identity(n, format='csc', dtype=np.complex128)

    # H 仍然是一个稀疏矩阵
    H = H / alpha
    
    # H_gqsp 初始化也使用稀疏单位矩阵
    H_gqsp_sparse = jn(0, -1 * alpha * dt) * identity_sparse
    
    # 确保切比雪夫多项式列表全程稀疏
    chebyshev_T_list = [identity_sparse, H] 
    
    # 第一次的加法也是稀疏的
    H_gqsp_sparse += 2 * j * jn(1, -1 * alpha * dt) * H
    
    # 在这个循环中，所有运算都是 稀疏*稀疏 - 稀疏，结果保持稀疏
    for k in range(2, K + 1):
        chebyshev_Tk = 2 * H @ chebyshev_T_list[k-1] - chebyshev_T_list[k-2]
        chebyshev_T_list.append(chebyshev_Tk)
        
        H_gqsp_sparse += (2 * (j**k) * jn(k, -1 * dt * alpha)) * chebyshev_Tk

    # 循环结束后，H_gqsp_sparse 是最终的稀疏结果
    # 将其转换为标准的稠密numpy数组后返回
    return H_gqsp_sparse.toarray()



def _calculate_commutator_norm_sum_direct(terms_A: list[QubitOperator], terms_B: list[QubitOperator], num_qubits: int) -> float:
    """
    【直接计算版】
    计算两组 QubitOperator 项之间所有对易子的范数之和。
    [A, B] = A*B - B*A
    """
    total_norm_sum = 0.0
    
    num_total_pairs = len(terms_A) * len(terms_B)
    current_pair_count = 0
    # 设置一个打印间隔，避免过于频繁地刷新屏幕
    print_interval = max(1, num_total_pairs // 100) if num_total_pairs > 0 else 1

    # print(f"  > 开始计算 {num_total_pairs} 对算符的对易子范数...")
    for op_a in terms_A:
        for op_b in terms_B:
            current_pair_count += 1
            # if current_pair_count % print_interval == 0:
            #     print(f"    ...进度: {current_pair_count}/{num_total_pairs}")

            # 直接使用 QubitOperator 的乘法和减法计算对易子
            commutator_op = op_b * op_a - op_a * op_b
            
            # 如果对易子为零，其范数也为0，可以跳过后续步骤
            if not commutator_op.terms:
                continue

            # 将结果转换为稀疏矩阵
            sparse_matrix = get_sparse_operator(commutator_op, n_qubits=num_qubits)
            try:
                # 计算谱范数
                norm = svds(sparse_matrix, k=1, which='LM', return_singular_vectors=False)[0]
            except Exception:
                # 备用方案，适用于svds可能失败的小矩阵
                norm = np.linalg.norm(sparse_matrix.toarray(), ord=2)
            
            # 累加范数值
            total_norm_sum += norm
            
    return total_norm_sum

# --- 步骤 1: 预计算函数 ---

def precompute_alpha_coeffs_direct(z: QubitOperator, h_corr: QubitOperator, file) -> dict:
    """
    【直接计算版】
    预计算 alpha(t) = alpha_ff + s(t)^2*alpha_cc + s(t)*alpha_fc 的三个系数。
    """
    # print("--- 开始预计算 alpha 系数 (p=1, 范数求和法) ---")
    start_time = time.time()

    # 确定系统所需的量子比特总数
    combined_ham = z + h_corr
    max_qubit_index = 0
    if combined_ham.terms:
        for term in combined_ham.terms:
            if term: # 跳过单位算符
                max_qubit_index = max(max_qubit_index, max(op[0] for op in term))
    num_qubits = max_qubit_index + 1

    # 将 z 和 H_corr 分解为 QubitOperator 子项列表
    fock_terms = [QubitOperator(pstr, c) for pstr, c in z.terms.items()]
    corr_terms = [QubitOperator(pstr, c) for pstr, c in h_corr.terms.items()]
    # 转换为反厄米项 A_gamma = -i * H_gamma
    fock_anti_terms = [-1j * term for term in fock_terms]
    corr_anti_terms = [-1j * term for term in corr_terms]
    
    # 1. 计算 alpha_ff (Fock 内部)
    # print("正在计算 alpha_ff (fock-fock)...")
    alpha_ff = _calculate_commutator_norm_sum_direct(fock_anti_terms, fock_anti_terms, num_qubits)
    
    # 2. 计算 alpha_cc (Corr 内部)
    # print("正在计算 alpha_cc (corr-corr)...")
    alpha_cc = _calculate_commutator_norm_sum_direct(corr_anti_terms, corr_anti_terms, num_qubits)
    
    # 3. 计算 alpha_fc (Fock 和 Corr 之间)
    # print("正在计算 alpha_fc (fock-corr cross-term)...")
    # 注意公式中的因子 2:  2 * Σ ||[B_k, A_j]||
    cross_sum = _calculate_commutator_norm_sum_direct(fock_anti_terms, corr_anti_terms, num_qubits)
    alpha_fc = 2 * cross_sum
    
    end_time = time.time()
    # print(f"--- 预计算完成，耗时: {end_time - start_time:.4f} 秒 ---")
    
    coeffs = {
        'alpha_ff': alpha_ff,
        'alpha_cc': alpha_cc,
        'alpha_fc': alpha_fc
    }
    try:
        # 使用 'with' 语句安全地打开文件
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f'alpha_ff: {alpha_ff}\n')
            f.write(f'alpha_cc: {alpha_cc}\n')
            f.write(f'alpha_fc: {alpha_fc}\n')
    except FileNotFoundError:
        print(f"错误: 文件未找到: '{file}'")
    return coeffs

# --- 步骤 2: 快速求值函数 ---

def get_alpha_at_time_t(s_t: float, alpha_coeffs: dict) -> float:
    """根据预计算的系数和时间权重 s(t)，快速计算 alpha(t)。"""
    return (1-s_t)**2*alpha_coeffs['alpha_ff'] + (s_t**2 * alpha_coeffs['alpha_cc']) + (s_t*(1-s_t) * alpha_coeffs['alpha_fc'])

#******************************************************************************
#           parameter setting
#******************************************************************************
#setting of the mol
# mol_str = 'O 0 0 0.5488; O 0 0 -0.5488'
from random import randrange




def calculate_energy_gap(hamiltonian_sparse: sps.spmatrix) -> float:
    """
    计算一个稀疏厄米哈密顿量的基态能量、第一激发态能量以及它们之间的能隙。

    Args:
        hamiltonian_sparse (sps.spmatrix): 以 SciPy 稀疏矩阵格式表示的哈密顿量。

    Returns:
        tuple[float, float, float]: 一个元组，包含 (基态能量, 第一激发态能量, 能隙)。
                                    如果矩阵太小无法计算，则返回 (nan, nan, nan)。
    """    
    n = hamiltonian_sparse.shape[0]
    eigenvalues, vec = eigsh(hamiltonian_sparse, k=n-2, which='SA', return_eigenvectors=True)

    # eigsh 返回的特征值是按升序排列的
    ground_state_energy, ground_state = eigenvalues[0], vec[:,0]
    first_excited_state_energy, first_excited_state = eigenvalues[1], vec[:,1]
    
    energy_gap = first_excited_state_energy - ground_state_energy
    
    return energy_gap, ground_state_energy, first_excited_state_energy, ground_state, first_excited_state


def calculate_energy_gap_dense(hamiltonian_sparse: sps.spmatrix) -> tuple:
    """
    通过将稀疏矩阵转换为稠密矩阵，计算所有本征值，然后找出能隙。

    Args:
        hamiltonian_sparse (sps.spmatrix): 以 SciPy 稀疏矩阵格式表示的哈密顿量。

    Returns:
        tuple: (能隙, 基态能量, 第一激发态能量, 基态, 第一激发态)。
               如果矩阵太小，则返回 (nan, nan, nan, None, None)。
    """
    N = hamiltonian_sparse.shape[0]
    if N < 2:
        return np.nan, np.nan, np.nan, None, None
    hamiltonian_dense = hamiltonian_sparse.toarray()
    eigenvalues, eigenvectors = eigh(hamiltonian_dense)
    ground_state_energy = eigenvalues[0]
    first_excited_state_energy = eigenvalues[1]
    
    ground_state = eigenvectors[:, 0]
    first_excited_state = eigenvectors[:, 1]
    
    energy_gap = first_excited_state_energy - ground_state_energy
    
    return energy_gap, ground_state_energy, first_excited_state_energy, ground_state, first_excited_state

def cal_sq_overlap(fci_state, asp_wf_exact_next, asp_wf_sim_next):
    overlap_sim_exact = dot(asp_wf_exact_next, conjugate(asp_wf_sim_next))
    sq_overlap_sim_exact = dot(overlap_sim_exact, conjugate(overlap_sim_exact)).real
    overlap_sim_fci = dot(asp_wf_sim_next, conjugate(fci_state))
    sq_overlap_sim_fci = dot(overlap_sim_fci, conjugate(overlap_sim_fci)).real
    overlap_exact_fci = dot(fci_state, conjugate(asp_wf_exact_next))
    sq_overlap_exact_fci = dot(overlap_exact_fci,conjugate(overlap_exact_fci)).real
    total_error = 1-sq_overlap_sim_fci
    sim_error = 1-sq_overlap_sim_exact
    ad_error = 1-sq_overlap_exact_fci
    return total_error, sim_error, ad_error

def load_alpha(file_path):
    """
    从一个文件中读取 alpha 值。

    文件的格式应该是:
    key: value
    例如:
    alpha_ff: 0.00000000
    alpha_cc: 51.39646608
    alpha_fc: 26.04653622

    Args:
        file_path (str): 输入文件的路径。

    Returns:
        dict: 一个包含 alpha 键和其浮点数值的字典。
              如果文件未找到或发生解析错误，则返回一个空字典。
    """
    alpha_values = {}
    try:
        # 使用 'with' 语句安全地打开文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行首尾的空白字符
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # 跳过空行

                try:
                    # 通过冒号分割键和值
                    key, value = stripped_line.split(':', 1)
                    
                    # 去除键和值两边的空白，并将值转换为浮点数
                    key = key.strip()
                    float_value = float(value.strip())
                    
                    # 将键值对存入字典
                    alpha_values[key] = float_value
                except ValueError:
                    # 如果分割或转换失败，则打印错误信息
                    print(f"警告: 无法解析行: '{stripped_line}'")
    except FileNotFoundError:
        print(f"错误: 文件未找到: '{file_path}'")
    
    return alpha_values

def create_hamiltonian(n_qubits: int, h: float, g: float, J: float, periodic: bool = True) -> QubitOperator:
    """
    构建指定的量子比特哈密顿量。

    H = sum_i (-h*Z_i Z_{i+1} + g * Z_i + J * X_i)

    Args:
        n_qubits (int): 系统中的量子比特数 (N)。
        periodic (bool): 是否使用周期性边界条件。
                         如果为 True, Z_{N-1} 会与 Z_0 相互作用。
                         如果为 False, 相互作用项只到 Z_{N-2}Z_{N-1}。

    Returns:
        QubitOperator: 构建好的哈密顿量的 OpenFermion 对象。
    """
    if n_qubits <= 1 and periodic:
        print("Warning: Cannot use periodic boundary conditions with <= 1 qubit. Defaulting to open.")
        periodic = False

    hamiltonian = QubitOperator()

    # 遍历每一个量子比特 i
    for i in range(n_qubits):
        # 1. 添加 Z_i Z_{i+1} 项
        # 确定 j = i+1，并处理边界条件
        if periodic:
            j = (i + 1) % n_qubits
            # 防止单比特周期性情况下的重复计算
            if n_qubits > 1 or i < j:
                 hamiltonian += QubitOperator(f'Z{i} Z{j}', -1.0)
        else: # 开放边界条件
            if i < n_qubits - 1:
                j = i + 1
                hamiltonian += QubitOperator(f'Z{i} Z{j}', -1.0)
        
        # 2. 添加 0.4 * Z_i 项
        hamiltonian += QubitOperator(f'Z{i}', g)

        # 3. 添加 0.4 * X_i 项
        hamiltonian += QubitOperator(f'X{i}', J)

    return hamiltonian

def asp_step_simulation(h_ins_jw, trotter_order, trotter_term_ordering, time_for_single_trotter, qubits, asp_wf_sim_curr, asp_wf_exact_curr, h_ins_sparse, fci_state):
    h_ins_circuit = cirq.Circuit()
    circ_tmp = sub_circuit(h_ins_jw, trotter_order, trotter_term_ordering, time_for_single_trotter,qubits)
    h_ins_circuit.append(circ_tmp)
    simulator = Simulator()
    asp_sim_result = simulator.simulate(h_ins_circuit, initial_state = asp_wf_sim_curr)
    asp_wf_sim_next = normalize_wave_function(asp_sim_result.final_state_vector)[0]
    asp_wf_exact_next = sp.sparse.linalg.expm_multiply(-1.0j * time_for_single_trotter * h_ins_sparse, asp_wf_exact_curr)
    asp_wf_exact_next = normalize_wave_function(asp_wf_exact_next)[0]
    # Calculate E(sim), <S^2>(sim), and sim-exact and sim-fci overlaps
    e_asp = expectation(h_ins_sparse, asp_wf_sim_next).real 
    e_asp_exact = expectation(h_ins_sparse, asp_wf_exact_next).real
    return asp_wf_sim_next, asp_wf_exact_next, e_asp, e_asp_exact

def PrintResults(hf_energy, e_asp, fci_energy, sq_overlap_sim_fci_ini, total_error, ad_error, sim_error, whether_bound, time_for_single_trotter, start_time, evolution_time, num_steps, alpha=None, alpha_i=None, alpha_f=None, g_i=None, g_f=None):
    # print("")
    # print(" SUMMARY OF THE QUANTUM CIRCUIT SIMULATION")
    # print("")
    # print("  E(ASP,Ini) = {:.10f}".format(hf_energy),"Hartree")
    # print("  E(ASP,Fin) = {:.10f}".format(e_asp),"Hartree")
    # print("  E(Full-CI) = {:.10f}".format(fci_energy),"Hartree")
    # print("1-|<ASP,Ini|ins_Full-CI>|^2 = {:.6f}".format(1-sq_overlap_sim_fci_ini))
    # print("1-|<ASP,Fin|ins_Full-CI>|^2 = {:.6f}".format(1-total_error))

    elapsed_time = time.time() - start_time
    if whether_bound:
        print()  
        print('Evolution time: {:.2f}'.format(evolution_time))
        print('Time for single Trotter step: {:.2f}'.format(time_for_single_trotter))
        print('Former bound for Trotter error: {:.6f}'.format(alpha**2*time_for_single_trotter**4)) 
        print('Our bound for Trotter error: {:.6f}'.format((alpha_i/g_i)**2+(alpha_f/g_f)**2*time_for_single_trotter**2))
        print('Ratio of the two bounds: {:.6f}'.format(alpha**2/((alpha_i/g_i)**2+(alpha_f/g_f)**2)*time_for_single_trotter**2))
        print('Simulation error: {:.6f}'.format(sim_error))
        print('Adiabatic error: {:.6f}'.format(ad_error))
        print('Total error: {:.6f}'.format(total_error))
        # print(f"{evolution_time}  {evolution_time/num_steps}   {alpha**2*time_for_single_trotter**4}   {(alpha_i/g_i)**2+(alpha_f/g_f)**2*time_for_single_trotter**2}   {alpha**2/((alpha_i/g_i)**2+(alpha_f/g_f)**2)}  {sim_error}   {ad_error} \n")
    else:
        print(f"{evolution_time}  {evolution_time/num_steps}     {sim_error}  {total_error} {ad_error} \n")
    # print(f"{evolution_time}  {evolution_time/num_steps}     {sim_error}  {total_error} {ad_error} \n")
    print("\nNormal termination. Wall clock time is {}".format(elapsed_time) + "[sec]")


def run_asp_simulation(
    evolution_time, num_steps,
    fock_ops, mol_op, fock_sparse, mol_sparse,
    initial_state, final_fci_state,
    qubits, precomputed_coeffs, weight_strategy="Lin", trotter_order=1, trotter_term_ordering="Magnitude"
):
    """
    Runs a complete ASP simulation and returns errors, gap list, AND the integrated alpha.
    """
    time_for_single_trotter = evolution_time / num_steps
    asp_wf_sim_curr, asp_wf_exact_curr = initial_state, initial_state
    gap_list, alpha = [], 0.0

    for i_step in range(1, num_steps + 1):
        s_t = get_hamiltonian_weight(i_step, num_steps, weight_strategy)
        h_ins_jw = (1 - s_t) * fock_ops + s_t * mol_op
        h_ins_sparse = (1 - s_t) * fock_sparse + s_t * mol_sparse
        
        inst_gap, _, _, _, _ = calculate_energy_gap_dense(h_ins_sparse)
        gap_list.append(inst_gap)
        
        # Calculate and accumulate alpha_t for the current step
        if precomputed_coeffs is not None:
            alpha_t = get_alpha_at_time_t(s_t, precomputed_coeffs)
            alpha += alpha_t
        
        asp_wf_sim_next, asp_wf_exact_next, _, _ = asp_step_simulation(
            h_ins_jw, trotter_order, trotter_term_ordering, time_for_single_trotter, 
            qubits, asp_wf_sim_curr, asp_wf_exact_curr, h_ins_sparse, final_fci_state
        )
        asp_wf_exact_curr, asp_wf_sim_curr = asp_wf_exact_next, asp_wf_sim_next

    _, final_sim_error, final_ad_error = cal_sq_overlap(
        final_fci_state, asp_wf_exact_curr, asp_wf_sim_curr
    )
    
    # Return the integrated alpha along with other results
    return final_ad_error, final_sim_error, gap_list, alpha

def renew_T(ad_error, target_error, T, T_low, T_high, binary_search_mode, i, max_adjust_steps, tolerance=1e-5):
    # Update the search bounds based on the result
    if ad_error > target_error:
        # The current T is too small, so it's a new lower bound.
        T_low = max(T_low, T)
    else: # ad_error < target_error
        # The current T is too large, so it's a new upper bound.
        T_high = min(T_high, T)

    # Decide the next T value
    if T_high != float('inf') and T_low > 0:
        # If we have found both an upper and a lower bound, we can switch to binary search.
        if not binary_search_mode:
            print(f"  -> Bracketing successful. Switching to binary search in range [{T_low:.2f}, {T_high:.2f}].")
            binary_search_mode = True
        # The next guess is the midpoint of the current valid range.
        T = (T_low + T_high) / 2
    else:
        # Continue with the proportional adjustment until the target is bracketed.
        if ad_error > target_error:
            adjustment_ratio = 1.5
        else:
            adjustment_ratio = 0.67
        T *= adjustment_ratio
    
    if i == max_adjust_steps - 1 and not binary_search_mode:
        print("WARNING: Max adjustment steps reached for Phase 1. Using last values.")
    
    return T, T_low, T_high, binary_search_mode

def renew_step_inverse(sim_error, target_error, step_inverse, step_inverse_low, step_inverse_high, binary_search_mode, i, max_adjust_steps, tolerance=1e-5):
    """
    Updates the step_inverse parameter using a hybrid proportional/binary search method.
    """
    # Update the search bounds based on the result
    if sim_error > target_error:
        # The current step_inverse is too small (too few steps), so it's a new lower bound.
        step_inverse_low = max(step_inverse_low, step_inverse)
    else: # sim_error < target_error
        # The current step_inverse is too large (too many steps), so it's a new upper bound.
        step_inverse_high = min(step_inverse_high, step_inverse)

    # Decide the next step_inverse value
    if step_inverse_high != float('inf') and step_inverse_low > 0:
        # If we have found both an upper and a lower bound, we can switch to binary search.
        if not binary_search_mode:
            print(f"  -> Bracketing successful. Switching to binary search in range [{step_inverse_low:.2f}, {step_inverse_high:.2f}].")
            binary_search_mode = True
        # The next guess is the midpoint of the current valid range.
        step_inverse = (step_inverse_low + step_inverse_high) / 2
    else:
        # Continue with the proportional adjustment until the target is bracketed.
        # This mirrors the logic from your renew_T function.
        if sim_error > target_error:
            # Error is too high, need more steps -> increase step_inverse
            adjustment_ratio = 2
        else:
            # Error is too low, can use fewer steps -> decrease step_inverse
            adjustment_ratio = 0.5
        step_inverse *= adjustment_ratio
    
    if i == max_adjust_steps - 1 and not binary_search_mode:
        print("WARNING: Max adjustment steps reached for Phase 2. Using last values.")
    
    return step_inverse, step_inverse_low, step_inverse_high, binary_search_mode




