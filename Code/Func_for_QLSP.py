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
    对于矩阵A的每一行，随机保留至多s个元素，其余元素置零。
    
    参数:
    A (np.ndarray): 输入矩阵
    s (int): 每行保留的最多元素数量
    
    返回:
    np.ndarray: 修改后的矩阵A
    """
    # 创建一个与A形状相同的掩码矩阵
    mask = np.zeros_like(A, dtype=bool)
    
    # 对每一行进行处理
    for i in range(A.shape[0]):
        # 获取当前行的所有列索引
        col_indices = np.arange(A.shape[1])
        
        # 随机选择至多s个列索引
        selected_indices = np.random.choice(col_indices, min(s, A.shape[1]), replace=False)
        
        # 将掩码矩阵中对应位置设置为True
        mask[i, selected_indices] = True
    
    # 应用掩码，将A中未被选择的元素置零
    A[~mask] = 0
    
    return A

def generate_one_sparse_matrix(kappa, N, s):
    # Step 1: 生成对角矩阵，其特征值在 [1, kappa] 之间
    eigenvalues = np.linspace(1, kappa, N)
    D = np.diag(eigenvalues)
    
    # Step 2: 生成随机正交矩阵
    Q, _ = qr(np.random.randn(N, N))
    
    # Step 3: 形成具有所需条件数的矩阵
    A = Q @ D @ Q.T
    
    # Step 4: 将矩阵稀疏化
    A_sparse = retain_random_elements(A, s)
    
    # 确保稀疏矩阵的对称性
    A_sparse = (A_sparse + A_sparse.T) / 2
    
    return csr_matrix(A_sparse)

def save_sparse_matrix_to_txt(A, filename):
    """
    将稀疏矩阵保存到.txt文件中。
    
    参数:
    A (sp.csr_matrix): 输入的稀疏矩阵
    filename (str): 保存的文件名
    """
    # 将稀疏矩阵转换为COO格式
    A_coo = A.tocoo()
    
    # 打开文件并写入数据
    with open(filename, 'w') as f:
        for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
            f.write(f"{i} {j} {v}\n")

def read_sparse_matrix_from_txt(filename):
    """
    从.txt文件中读取稀疏矩阵。
    
    参数:
    filename (str): 文件名
    
    返回:
    sp.coo_matrix: 读取的稀疏矩阵
    """
    # 初始化空的行、列和数据列表
    row = []
    col = []
    data = []
    
    # 打开文件并读取数据
    with open(filename, 'r') as f:
        for line in f:
            # 解析每一行，获取行索引、列索引和值
            i, j, v = map(float, line.split())
            row.append(int(i))
            col.append(int(j))
            data.append(v)
    
    # 创建COO格式的稀疏矩阵
    A_coo = sp.coo_matrix((data, (row, col)))
    
    return A_coo

def generate_sparse_matrix(kappa, N, s, condition):
    # 参数
    while True:
        # 生成矩阵
        A = generate_one_sparse_matrix(kappa, N, s)
        condition_number = np.linalg.cond(A.toarray())
        if condition_number <= condition:
            break
    # 打印结果
    print("生成的稀疏矩阵:")
    print(A.toarray())

    # 检查条件数
    condition_number = np.linalg.cond(A.toarray())
    print(f"条件数: {condition_number}")
    save_sparse_matrix_to_txt(A,f'tmp_sparse_matrix_N={N}.txt')


def generate_sparse_normalized_vector(N, s):
    """
    生成一个随机的、至多有s个非零元素的N维向量，并进行归一化。
    
    参数:
    N (int): 向量的维度
    s (int): 非零元素的最大数量
    
    返回:
    np.ndarray: 归一化后的N维向量
    """
    s = min(s, N)
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
    将向量写入到.txt文件中。
    
    参数:
    vector (np.ndarray): 要写入的向量
    filename (str): 保存的文件名
    """
    np.savetxt(filename, vector, delimiter=',')
    # print(f"向量已保存到 {filename} 文件中。")

def read_vector_from_txt(filename):
    """
    从.txt文件中读取向量。
    
    参数:
    filename (str): 文件名
    
    返回:
    np.ndarray: 读取的向量
    """
    vector = np.loadtxt(filename, delimiter=',')
    # print(f"从文件 {filename} 中读取的向量：", vector)
    return vector


def data_generating():
    generate_sparse_matrix(kappa=5, N=16, s=4, condition=10)
    generate_sparse_matrix(kappa=5, N=32, s=5, condition=50)
    b = generate_sparse_normalized_vector(N=16, s=5)
    write_vector_to_txt(b, filename='sparse_vec_N=16.txt')
    b = generate_sparse_normalized_vector(N=32, s=5)
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
        # 使用 np.einsum 计算张量积并保持原始维数
        tensor_product = np.einsum('i,j->ij', matrix1, matrix2)

        # 将结果重新排列为原始维数
        result = tensor_product.reshape(matrix1.shape[0] * matrix2.shape[0])
        return result
    if dim == 2:
        # 使用 np.einsum 计算张量积并保持原始维数
        tensor_product = np.einsum('ij,kl->ikjl', matrix1, matrix2)

        # 将结果重新排列为原始维数
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
    # 避免在0和1处的奇异性
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
    """将Pauli字符串转换为矩阵"""
    result = 1
    for p in pauli_string:
        result = np.kron(result, pauli_matrices[p])
    return result

def decompose_to_pauli_strings(H):
    """将厄密矩阵分解为Pauli字符串的和"""
    n = int(np.log2(H.shape[0]))
    pauli_strings = list(product(range(4), repeat=n))
    coefficients = []

    for pauli_string in pauli_strings:
        P = pauli_string_to_matrix(pauli_string)
        coefficient = np.trace(H @ P) / (2 ** n)
        coefficients.append(coefficient)

    return coefficients, pauli_strings

def pauli_string_to_label(pauli_string):
    """将Pauli字符串转换为标签"""
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
    # 使用 zip 函数将两个数组合并成一个列表
    combined = list(zip(A, B))
    # 按照 A 的值进行排序
    combined_sorted = sorted(combined, key=lambda x: x[0],reverse = reverse)
    # 拆分回两个数组
    A_sorted, B_sorted = zip(*combined_sorted)
    return A_sorted, B_sorted

def generate_unitary_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 随机生成一个 n x n 的复数矩阵
    random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    
    # QR 分解
    q, r = np.linalg.qr(random_matrix)
    
    # 调整 R 的对角线元素的相位，使得 Q 仍然是酉矩阵
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
# 定义Pauli矩阵
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1*j], [j, 0]])
Z = np.array([[1, 0], [0, -1]])
Sigmap = (X+j*Y)/2
Sigmam = (X-j*Y)/2
pauli_matrices = [I, X, Y, Z]
pauli_labels = ['I', 'X', 'Y', 'Z']
