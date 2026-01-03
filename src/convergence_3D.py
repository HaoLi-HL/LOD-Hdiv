from firedrake import *
from collections import defaultdict
import numpy as np
import petsc4py.PETSc as PETSc
from mesh3D_manager import *
from lod_new import mixed_LOD


def a(x, y, z):
    col = np.floor(x * (np.size(Adata, 1))).astype(int)
    row = np.floor((1 - y) * (np.size(Adata, 0))).astype(int)
    dep = np.floor(z * (np.size(Adata, 2))).astype(int)
    # Clamp indices to avoid out-of-bounds
    col = np.clip(col, 0, np.size(Adata, 1) - 1)
    row = np.clip(row, 0, np.size(Adata, 0) - 1)
    dep = np.clip(dep, 0, np.size(Adata, 2) - 1)
    return Adata[row, col, dep]

def f(x, y, z):
    return 3 * np.pi**2 * cos(np.pi * x) * cos(np.pi * y) * cos(np.pi * z)


coef = 'noise'
N_fine_level = 4
# Start program
if coef == '1':
    Adata = np.array([[[1]]])  # 3D array for constant
elif coef == 'noise':
    np.random.seed(1)
    alpha = 0.01
    beta = 1
    N = 2**(N_fine_level-1)  # Reduced size for 3D to manage computation
    Adata = alpha + (beta - alpha) * np.random.randint(0, 2, (N, N, N))
    # print(f"Random coef with alpha={alpha} and beta={beta}.")
elif coef == 'channels':
    # Note: netpbmfile.imread is for 2D PGM; for 3D, need a 3D data source (e.g., voxel data)
    # Simulate or load 3D equivalent; here, placeholder - extend to 3D by replication if needed
    # Adata_2d = netpbmfile.imread("../../darumalib-main/data/randomA.pgm")
    # Adata = np.repeat(Adata_2d[:, :, np.newaxis] + 0.1, N, axis=2)  # Replicate in z
    print("Channels coef not directly supported in 3D; using placeholder.")
    sys.exit(2)
else:
    print('Invalid coefficient:', coef)
    sys.exit(2)

# n = 2  # Reduced base resolution for 3D to manage memory/compute (original n=2)
# N_fine_level = 3  # Reduced fine level for 3D (original 7); adjust based on resources
# L = N_fine_level - n
# k = 2

# base_mesh = UnitCubeMesh(2**n, 2**n, 2**n)
# mesh_mgr = mesh3D_manager(base_mesh, L)  # Assuming 3D version of manager class exists
    
# test_mixed_LOD = mixed_LOD(base_mesh, a, f, L, k, int_method="canonical", printLevel=1, parallel_computation=True)
# # test_mixed_LOD = mixed_LOD(base_mesh, a, f, L, k, int_method="stable", printLevel=1, parallel_computation=True)

# flux_ref, pres_ref = test_mixed_LOD.ref(True)
# M_a = test_mixed_LOD.M_a_csr

# x, y, z = SpatialCoordinate(test_mixed_LOD.mesh_mgr.mesh_f)


# def u(x, y, z):
#     return cos(np.pi * x) * cos(np.pi * y) * cos(np.pi * z)

# print(norm(grad(u(x, y, z)) - flux_ref), norm(u(x, y, z) - pres_ref))
# print(norm(flux_ref), norm(pres_ref))

# For 3D visualization, use VTK output instead of 2D plots
# e.g., File("flux_ref.pvd").write(flux_ref)
# File("pres_ref.pvd").write(pres_ref)
# Comment out 2D-specific plots:
# plot_RT_func(test_mixed_LOD.mesh_mgr.mesh_f, flux_ref)
# plot_DG_func(test_mixed_LOD.mesh_mgr.mesh_f, pres_ref)

def main():
    k_list = [1, 2, 3, 3]
    for i, n in enumerate(range(1, N_fine_level)):
        L = N_fine_level - n
        N = 2**n
        k = k_list[i]
        print("-----------------------------------------")
        print(f"Testing mixed LOD with 1/{2**n} coarse mesh, L={L}, k={k}")
        base_mesh = UnitCubeMesh(N, N, N)
        test_mixed_LOD = mixed_LOD(base_mesh, a, f, L, k, int_method="stable", printLevel=0, parallel_computation=True)
        flux_lod, pres_lod = test_mixed_LOD.lod()
        flux_ref, pres_ref = test_mixed_LOD.ref(True)
        M_a = test_mixed_LOD.M_a_csr
        ref_ENorm = np.sqrt(flux_ref.dat.data.T @ M_a @ flux_ref.dat.data)

        Mq_csr = test_mixed_LOD.Mq_csr
        ref_Pres_Norm = np.sqrt(pres_ref.dat.data.T @ Mq_csr @ pres_ref.dat.data)

        print(f"Reference solution energy norm: {ref_ENorm}, pressure L2 norm: {ref_Pres_Norm}")
        err_arr = flux_lod.dat.data - flux_ref.dat.data
        rel_err_ENorm = np.sqrt(err_arr.T @ M_a @ err_arr)/ref_ENorm
        err_pres_arr = pres_lod.dat.data - pres_ref.dat.data
        rel_err_Pres_L2Norm = np.sqrt(err_pres_arr.T @ Mq_csr @ err_pres_arr)/ref_Pres_Norm

        print(f"1/{2**n}, error energy norm: {rel_err_ENorm}, ref energy norm: {ref_ENorm}, pressure L2 norm error: {rel_err_Pres_L2Norm}, ref pressure L2 norm: {ref_Pres_Norm}")

if __name__ == "__main__":
    main()