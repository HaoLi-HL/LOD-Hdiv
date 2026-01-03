from firedrake import *
from collections import defaultdict
import numpy as np
import petsc4py.PETSc as PETSc
from mesh2D_manager import *
from lod_new import mixed_LOD
import time

def a(x, y):
    col = np.floor(x*(np.size(Adata,1))).astype(int)
    row = np.floor((1-y)*(np.size(Adata,0))).astype(int)
    return Adata[row,col]

def f(x, y):
    return 2 * np.pi**2 * cos(np.pi * x) * cos(np.pi * y)


coef = 'noise'
N_fine_level = 6
# Start program
if coef=='1':
    Adata = np.array([[1]])
elif coef=='noise':
    np.random.seed(1)
    # Adata = np.exp(10*np.random.rand(2**7,2**7))
    alpha = 0.01
    beta = 1
    N = 2**(N_fine_level-1)
    Adata = alpha+(beta-alpha)*np.random.randint(0, 2, (N, N))
    # print(f"Random coef with alpha={alpha} and beta={beta}.")
elif coef=='channels':
    Adata = netpbmfile.imread("../../darumalib-main/data/randomA.pgm")
    Adata = Adata+0.1
    # Adata = np.exp(10./255*Adata)
else:
    # print 'Invalid coefficient:', coef
    print('Invalid coefficient:', coef)
    sys.exit(2)


def main():
    k_min = 1
    k_max = 7
    n_min = 1
    n_max = N_fine_level - 1

    degree = 2
    
    u_eng_err_arr = np.zeros((n_max-n_min+1, k_max-k_min+1))
    u_l2_err_arr = np.zeros((n_max-n_min+1, k_max-k_min+1))
    pre_l2_err_arr = np.zeros((n_max-n_min+1, k_max-k_min+1))
    divu_l2_err_arr = np.zeros((n_max-n_min+1, k_max-k_min+1))

    for n in range(n_min, n_max+1):
        L = N_fine_level - n
        N = 2**n
        for k in np.arange(k_min, k_max+1):
            print("-----------------------------------------")
            startT = time.time()
            base_mesh = UnitSquareMesh(N, N)
            test_mixed_LOD = mixed_LOD(base_mesh, a, f, L, k, degree=degree, int_method="stable", printLevel=0, parallel_computation=True)
            flux_ref, pres_ref = test_mixed_LOD.ref(True)
            M_a = test_mixed_LOD.M_a_csr
            ref_ENorm = np.sqrt(flux_ref.dat.data.T @ M_a @ flux_ref.dat.data)
            Mq_csr = test_mixed_LOD.Mq_csr
            ref_Pres_Norm = np.sqrt(pres_ref.dat.data.T @ Mq_csr @ pres_ref.dat.data)
            print(f"Testing mixed LOD with 1/{N} coarse mesh, k={k}, degree={degree}. Reference solution energy norm: {ref_ENorm:8f}, pressure L2 norm: {ref_Pres_Norm:8f}")

            flux_lod, pres_lod = test_mixed_LOD.lod()
            solveT = time.time()-startT
            err_arr = flux_lod.dat.data - flux_ref.dat.data
            rel_err_ENorm = np.sqrt(err_arr.T @ M_a @ err_arr)/ref_ENorm
            err_pres_arr = pres_lod.dat.data - pres_ref.dat.data
            rel_err_Pres_L2Norm = np.sqrt(err_pres_arr.T @ Mq_csr @ err_pres_arr)/ref_Pres_Norm

            u_eng_err_arr[n-n_min, k-k_min] = rel_err_ENorm
            pre_l2_err_arr[n-n_min, k-k_min] = rel_err_Pres_L2Norm

            print(f"n={n}, k={k}, degree={degree}, error energy norm: {rel_err_ENorm:8f}, pressure L2 norm error: {rel_err_Pres_L2Norm:8f}, solveT={solveT:4f}")

        np.save(f'num_results/u_eng_err_arr_deg_{degree}_n_{n}.npy', u_eng_err_arr)
        np.save(f'num_results/pre_l2_err_arr_deg_{degree}_n_{n}.npy', pre_l2_err_arr)

if __name__ == "__main__":
    main()