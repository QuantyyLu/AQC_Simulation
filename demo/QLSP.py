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
from Func_for_QLSP import *
from math import log10
import time


# data_generating() # If you want to generate sparse matrix and sparse vector, uncomment this line

begin_time = time.time()

N = 16
A, b = data_loading(N)
A =  A.toarray()
x = np.matmul(np.linalg.inv(A),b)
x =  x/norm(x)
belln = np.array([1/sqrt(2),-1*1/sqrt(2)])
bellp = np.array([1/sqrt(2),1/sqrt(2)])
phi0 = np.array([1,0])
X = np.array([[0, 1],[1,0]])
Z = np.array([[1, 0],[0, -1]])
IN = np.eye(N)
I2 = np.eye(2)
Z_IN = tensor_product(Z,IN)
X_A = tensor_product(X, A) 
bar_b = tensor_product(bellp, b)
P_bar_b = np.eye(2*N)- np.einsum('i,j->ij', bar_b, bar_b)
xext = tensor_product(phi0,tensor_product(bellp,x))
xext = xext/norm(xext)


x0 = tensor_product(phi0,tensor_product(belln, b))
x0 = x0.transpose()
# xext = tensor_product(bellp,x)
# xext = xext/norm(xext)


# x0 = tensor_product(belln, b)
# x0 = x0.transpose()
dt = 0.05
alpha = 10
K = 3
Ts = np.logspace(2,3, 1)
# Ts = np.array([   100.,    215.,    464.,   1000.,   2154.,   4641.,  10000.,
#         21544.,  46415.])
# Ts = [10000]
# Ts = [100000]
errors = []
filename = 'Data/QLSP.txt'
file = open(filename, 'w')
methods = ['Linear']#, 'Squ', 'Cub', 'Exp']

print(f"{'T':>10} {'dt':>10} {'total error':>18} {'non-ad error':>18} {'gqsp error':>18}")

for method in methods:
    # Calculate the error of the GQSP algorithm for different evolution times
    errors = []
    file.write('K=3'+'\n')
    for p in range(len(Ts)):
        T = Ts[p]
        T = int(T)
        Ts[p] = T
        r = int(T/dt) 
        dt = T/r
        x0 = x0/norm(x0)
        xs = x0
        adxs = x0
        for i in range(r):
            s = schedule(i/r,method)
            Hs = H(s, Z_IN, X_A, P_bar_b)
            xtmp = np.dot(GQSP(K, Hs,dt,alpha), xs)
            adxtmp = np.dot(expm(-1* j * dt* Hs) , adxs)
            adxs = adxtmp/norm(adxtmp)
            # Normalize by columns
            xs = xtmp/norm(xtmp)
            gamma_tot = np.dot(xext,xs.conj())
            gamma_ad = np.dot(xext,adxs.conj())
            gamma_gqsp = np.dot(xs.transpose(),adxs.conj())
            # if (i+1) % 100 ==  0:
            #     print(method,p,i, r, 1-abs(gamma_tot)**2,1-abs(gamma_ad)**2,1-abs(gamma_lcu)**2)

        print(f"{T:>10.3f} {dt:>10.3f} {1-abs(gamma_tot)**2:>18.8e} {1-abs(gamma_ad)**2:>18.8e} {1-abs(gamma_gqsp)**2:>18.8e}")
        error = [T, 1-abs(gamma_tot)**2,1-abs(gamma_ad)**2,1-abs(gamma_gqsp)**2]
        file.write(' '.join(map(str,error))+'\n')
        errors.append(error)
file.close()

end_time = time.time()
print(f"Runtime: {end_time - begin_time:.2f} s")