from cmath import sqrt
import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix, random as sparse_random
import scipy.sparse as sp
from math import sqrt
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.special import jn
from scipy.integrate import quad
from scipy.sparse.linalg import eigs

from itertools import product

def retain_random_elements(A, s):
    """
    For each row of the matrix A, randomly retain at most s elements and set the rest to zero.
    Parameters:
    A (np.ndarray): The input matrix
    s (int): The maximum number of elements to retain in each row
    Returns:
    np.ndarray: The modified matrix A with at most s non-zero elements per row
    """    
    mask = np.zeros_like(A, dtype=bool)
    
    for i in range(A.shape[0]):
        col_indices = np.arange(A.shape[1])
        
        selected_indices = np.random.choice(col_indices, min(s, A.shape[1]), replace=False)
        
        mask[i, selected_indices] = True
    
    A[~mask] = 0
    
    return A

def generate_one_sparse_matrix(kappa, N, s):
    # Step 1: Generating a diagonal matrix with eigenvalues between 1 and kappa
    eigenvalues = np.linspace(1, kappa, N)
    D = np.diag(eigenvalues)
    
    # Step 2: Generate a random orthogonal matrix
    Q, _ = qr(np.random.randn(N, N))
    
    # Step 3: Construct the matrix with the desired condition number
    A = Q @ D @ Q.T
     
    A_sparse = retain_random_elements(A, s)
    A_sparse = (A_sparse + A_sparse.T) / 2
    
    return csr_matrix(A_sparse)

def save_sparse_matrix_to_txt(A, filename):
    """
    Save a sparse matrix to a .txt file.
    Parameters:
    A (sp.csr_matrix): The sparse matrix to be saved
    filename (str): The name of the file to save the matrix to
    """
    # Transform the sparse matrix to COO format for easier iteration
    A_coo = A.tocoo()
    
    with open(filename, 'w') as f:
        for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
            f.write(f"{i} {j} {v}\n")

def read_sparse_matrix_from_txt(filename):
    """
    Read a sparse matrix from a .txt file.
    Parameters:
    filename (str): The name of the file to read the matrix from
    Returns:
    sp.coo_matrix: The sparse matrix read from the file
    """
    row = []
    col = []
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            i, j, v = map(float, line.split())
            row.append(int(i))
            col.append(int(j))
            data.append(v)
    
    # Create a COO format sparse matrix
    A_coo = sp.coo_matrix((data, (row, col)))
    
    return A_coo

def generate_sparse_matrix(kappa, N, s, condition):
    while True:
        # generate a sparse matrix with the specified condition number
        A = generate_one_sparse_matrix(kappa, N, s)
        condition_number = np.linalg.cond(A.toarray())
        if condition_number <= condition:
            break
    # Print the generated sparse matrix and its condition number
    print("Sparse matrix generated:")
    print(A.toarray())

    # Check the condition number
    condition_number = np.linalg.cond(A.toarray())
    print(f"Condition number: {condition_number}")
    save_sparse_matrix_to_txt(A,f'sparse_matrix_N={N}.txt')


def generate_sparse_normalized_vector(N):
    """
    Generate a random N-dimensional vector with at most s non-zero elements and normalize it.
    Parameters:
    N (int): Dimension of the vector
    s (int): Maximum number of non-zero elements
    Returns:
    np.ndarray: Normalized N-dimensional vector
    """    

    if N == 16:
        s = 4
    elif N == 32:
        s = 5
    b = np.zeros(N)
    non_zero_indices = np.random.choice(N, s, replace=False)
    b[non_zero_indices] = np.random.randn(s)
    norm = np.linalg.norm(b)
    if norm == 0:
        return b
    b_normalized = b / norm
    return b_normalized

def write_vector_to_txt(vector, filename):
    """
    Write a vector to a .txt file. 
    Parameters:
    vector (np.ndarray): The vector to be written to the file
    filename (str): The name of the file to save the vector to    
    """
    np.savetxt(filename, vector, delimiter=',')
    # print(f"Vector saved to {filename}")

def read_vector_from_txt(filename):
    """
    Read a vector from a .txt file.    
    Parameters:
    
    filename (str): file name to read the vector from
    
    Returns:
    np.ndarray: the vector read from the file
    """
    vector = np.loadtxt(filename, delimiter=',')
    # print(f"Vector read from {filename}: {vector}")
    return vector


def data_generating():
    generate_sparse_matrix(kappa=5, N=16, s=4, condition=10)
    generate_sparse_matrix(kappa=5, N=32, s=5, condition=50)
    b = generate_sparse_normalized_vector(N=16)
    write_vector_to_txt(b, filename='sparse_vec_N=16.txt')
    b = generate_sparse_normalized_vector(N=32)
    write_vector_to_txt(b, filename='sparse_vec_N=32.txt')


def data_loading(N):
    mat_filename = f'sparse_matrix_N={N}.txt'
    A = read_sparse_matrix_from_txt(mat_filename)
    vec_filename = f'sparse_vec_N={N}.txt'
    b = read_vector_from_txt(vec_filename)
    return A, b

def tensor_product(matrix1, matrix2):
    dim = len(matrix1.shape)
    if dim == 1:
        tensor_product = np.einsum('i,j->ij', matrix1, matrix2)

        result = tensor_product.reshape(matrix1.shape[0] * matrix2.shape[0])
        return result
    if dim == 2:
        tensor_product = np.einsum('ij,kl->ikjl', matrix1, matrix2)

        result = tensor_product.reshape(matrix1.shape[0] * matrix2.shape[0], matrix1.shape[1] * matrix2.shape[1])
        return result


def H(s, Z_IN, X_A, P_bar_b):
    As = (1-s)*Z_IN + s* X_A
    Op = tensor_product(Sigmap, As @ P_bar_b) 
    return Op + np.transpose(Op.conj())
    # return As @ P_bar_b @ As

def spnorm(x):
    modulus_squared = x.transpose().dot(x.conj()).real
    return sqrt(modulus_squared.sum())

    
def QSP(K,H,dt):
    j = complex(0,1)
    n = H.shape[0]
    H_qsp = jn(0,-1*dt)* complex(1,0) * np.eye(n)
    chebyshev_T_list = [np.eye(n),H]
    H_qsp += 2 * j * jn(1,-1*dt) * H
    for k in range(2,K+1):
        chebyshev_Tk = 2 * H @ chebyshev_T_list[k-1] - chebyshev_T_list[k-2]
        chebyshev_T_list.append(chebyshev_Tk)
        H_qsp += 2 * j**k * jn(k,-1*dt) * chebyshev_Tk
    return H_qsp

def GQSP(K,H,dt,alpha):
    j = complex(0,1)
    n = H.shape[0]
    H = H/alpha
    H_gqsp = jn(0,-1*dt)* complex(1,0) * np.eye(n)
    chebyshev_T_list = [np.eye(n),H]
    H_gqsp += 2 * j * jn(1,-1*dt) * H
    for k in range(2,K+1):
        chebyshev_Tk = 2 * H @ chebyshev_T_list[k-1] - chebyshev_T_list[k-2]
        chebyshev_T_list.append(chebyshev_Tk)
        H_gqsp += 2 * j**k * jn(k,-1*dt*alpha) * chebyshev_Tk
    return H_gqsp


def integrand(x_prime):
    return np.exp(-1 / (x_prime * (1 - x_prime)))

def calculate_integral(x, epsilon=1e-10):
    if x < epsilon :
        return 0
    # Avoid singularity at 0 and 1
    result, _ = quad(integrand, epsilon, x - epsilon)
    return result

def schedule(x, funcname = 'Linear',epsilon=1e-10):
    if funcname == 'Linear':
        return x
    elif funcname == 'Squ':
        return 3 * x**2 - 2 * x**3
    elif funcname == 'Cub':
        return 6 * x**5 -15 * x ** 4 + 10 * x**3
    elif funcname == 'Exp':
        C = calculate_integral(1, epsilon)
        integral_value = calculate_integral(x, epsilon)
        return integral_value / C


def pauli_string_to_matrix(pauli_string):
    """ Translate a Pauli string to a matrix"""
    result = 1
    for p in pauli_string:
        result = np.kron(result, pauli_matrices[p])
    return result

def decompose_to_pauli_strings(H):
    """Decompose a Hermitian matrix into a sum of Pauli strings"""
    n = int(np.log2(H.shape[0]))
    pauli_strings = list(product(range(4), repeat=n))
    coefficients = []

    for pauli_string in pauli_strings:
        P = pauli_string_to_matrix(pauli_string)
        coefficient = np.trace(H @ P) / (2 ** n)
        coefficients.append(coefficient)

    return coefficients, pauli_strings

def pauli_string_to_label(pauli_string):
    """Translate a Pauli string to a label"""
    return ''.join(pauli_labels[p] for p in pauli_string)

def trotterization(H,dt):
    N = H.shape[0]
    H_trotter = np.eye(N)
    coefficients, pauli_strings = decompose_to_pauli_strings(H)
    for coeff, pauli_string in zip(coefficients, pauli_strings):
        P = pauli_string_to_matrix(pauli_string)
        H_trotter =   H_trotter @ expm(-1*j*coeff*dt*P)
    return H_trotter

def trotterization(H,dt,reverse = False):
    N = H.shape[0]
    H_trotter = np.eye(N)
    coefficients, pauli_strings = decompose_to_pauli_strings(H)
    if not reverse:
        for coeff, pauli_string in zip(coefficients, pauli_strings):
            P = pauli_string_to_matrix(pauli_string)
            H_trotter =    expm(-1*j*coeff*dt*P) @H_trotter
    else:
        for coeff, pauli_string in zip(coefficients, pauli_strings):
            P = pauli_string_to_matrix(pauli_string)
            H_trotter =   H_trotter @ expm(-1*j*coeff*dt*P) 
    return H_trotter

def cosort(A, B, reverse = False):
    combined = list(zip(A, B))
    combined_sorted = sorted(combined, key=lambda x: x[0],reverse = reverse)
    A_sorted, B_sorted = zip(*combined_sorted)
    return A_sorted, B_sorted

def generate_unitary_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random n x n complex matrix
    random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    
    # QR decomposition 
    q, r = np.linalg.qr(random_matrix)
    
    # Modify R to ensure Q is unitary
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    
    return q

def schedule(x, funcname = 'Linear',epsilon=1e-10):
    if funcname == 'Linear':
        return x
    elif funcname == 'Squ':
        return 3 * x**2 - 2 * x**3
    elif funcname == 'Cub':
        return 6 * x**5 -15 * x ** 4 + 10 * x**3
    elif funcname == 'Exp':
        C = calculate_integral(1, epsilon)
        integral_value = calculate_integral(x, epsilon)
        return integral_value / C


j = complex(0,1)
# Define Pauli matrices
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1*j], [j, 0]])
Z = np.array([[1, 0], [0, -1]])
Sigmap = (X+j*Y)/2
Sigmam = (X-j*Y)/2
pauli_matrices = [I, X, Y, Z]
pauli_labels = ['I', 'X', 'Y', 'Z']
