import numpy as np
import matplotlib.pyplot as plt
from firedrake import *

from mesh2D_manager import *
from mesh3D_manager import *
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, bmat
from scipy.sparse.linalg import spsolve



def _petsc_to_csr(petsc_mat):
    indptr, indices, data = petsc_mat.getValuesCSR()
    shape = petsc_mat.getSize()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)

def solve_linear_system(M, B, IRT, IQ, rhs):
    """
    Solve the linear system K @ x = rhs using block elimination after permutation.
    
    Parameters:
    - M: n x n sparse matrix
    - B: m x n sparse matrix
    - IRT: k x n sparse matrix
    - IQ: l x m sparse matrix
    - rhs: dense vector of size n + m + k + l
    
    Returns:
    - x: solution vector in original order
    """
    # Determine block sizes
    n = M.shape[0]
    m = B.shape[0]
    k = IRT.shape[0]
    l = IQ.shape[0]
    N = n + m + k + l

    # Form the submatrix K for the first three blocks (variables 1, 2, 4)
    K = sparse.bmat([[M, B.T, None],
                     [B, None, IQ.T],
                     [None, IQ, None]], format='csc')
    # K = quasi_int.symmetrize_and_clean_sparse_matrix(K, tol=1e-15)
    # Compute LU factorization of K
    solver = sparse.linalg.splu(K)
    solve = solver.solve

    # solve = factorized(K)

    # Define L: (n + m + l) x k matrix, L[:n, :] = IRT.T, L[n:, :] = 0
    L = sparse.vstack([IRT.T, sparse.csc_matrix((m + l, k))])

    # Compute Schur complement S = - M @ Z
    S = -L.T@solve(L.toarray())
    
    # Permute rhs to match K_permuted order: [0:n, n:n+m, n+m+k:n+m+k+l, n+m:n+m+k]
    rhs_permuted = np.hstack([rhs[:n], rhs[n:n+m], rhs[n+m+k:n+m+k+l], rhs[n+m:n+m+k]])
    rhs1 = rhs_permuted[:n + m + l]  # First three blocks
    rhs2 = rhs_permuted[n + m + l:]  # Last block

    # Solve K y = rhs1
    y = solve(rhs1)

    # Compute w = M @ y
    w = IRT @ y[:n]  # Since M_mat @ y = IRT @ y[:n]

    # Compute right-hand side for Schur complement
    rhs_S = rhs2 - w

    # Option 1: Dense inversion (suitable if k is small)
    # Solve S x2 = rhs_S
    x2_dense = np.linalg.solve(S, rhs_S)
    x2 = x2_dense

    # Option 2: Sparse inversion (if S were sparse, but here it's dense due to K^{-1})
    # For demonstration, convert S to sparse and use spsolve (though typically dense here)
    # S_sparse = sparse.csc_matrix(S)
    # S_sparse = quasi_int.symmetrize_and_clean_sparse_matrix(S_sparse)
    # x2_sparse = sparse.linalg.spsolve(S_sparse, rhs_S)
    # x2 = x2_sparse
    # print(f"# Nonzero: {S_sparse.data.size}, # Entries: {np.multiply(*S.shape)}, Matrix size: {S.shape}")
    
    # Compute L x2 = [IRT.T @ x2, 0, 0]
    L_x2 = np.hstack([IRT.T @ x2, np.zeros(m + l)])

    # Solve K temp = L x2
    temp = solve(L_x2)

    # Compute x1 = y - K^{-1} (L x2)
    x1 = y - temp

    # Form x_permuted
    x_permuted = np.hstack([x1, x2])

    # Rearrange to original order: [0:n, n:n+m, n+m:n+m+k, n+m+k:n+m+k+l]
    x = np.zeros(N)
    x[:n] = x_permuted[:n]
    x[n:n+m] = x_permuted[n:n+m]
    x[n+m:n+m+k] = x_permuted[n+m+l:n+m+l+k]
    x[n+m+k:n+m+k+l] = x_permuted[n+m:n+m+l]
    return x[:n]

def worker(input_tuple):
    M_list, B_list, IRT_list, IQ_list, rhs_list, flux_rows_list = input_tuple
    corrector_vals = []
    corrector_rows = []
    for i in range(len(M_list)):
        M_a = M_list[i]
        B = B_list[i]
        IRT = IRT_list[i]
        IQ = IQ_list[i]
        rhs = rhs_list[i]
        flux_rows = flux_rows_list[i]
        fluxVals = solve_linear_system(M_a, B, IRT, IQ, rhs)
        corrector_vals.extend(fluxVals)
        corrector_rows.extend(flux_rows)
    return corrector_vals, corrector_rows

def clean_sparse_matrix(K, tol=1e-12):
    # Ensure K is in CSC format
    if not sparse.isspmatrix_csc(K):
        K = K.tocsc()
    
    max_abs = np.max(np.abs(K.data))
    threshold = tol * max_abs
    # Eliminate small entries: set entries with abs(value) < tol to zero
    K.data[np.abs(K.data) < threshold] = 0
    
    # Remove explicit zero entries to maintain sparsity
    K.eliminate_zeros()
    return K

class mixed_LOD:
    def __init__(self, base_mesh, a, f, L=3, k=np.inf, ell=0, noCorrectors=0, noCorrections=1, degree=1, int_method="canonical", printLevel=0, parallel_computation=False):
        self.a = a
        self.f = f
        self.k = k
        self.ell = ell
        self.noCorrectors = noCorrectors
        self.noCorrections = noCorrections
        self.degree = degree
        self.dim = base_mesh.topological_dimension()
        self.printLevel = printLevel
        
        if self.dim == 2:
            if self.printLevel > 0:
                print(f"Building 2D mesh manager with refinement level L={L}")
            self.mesh_mgr = mesh2D_manager(base_mesh, L)
        elif self.dim == 3:
            if self.printLevel > 0:
                print(f"Building 3D mesh manager with refinement level L={L}")
            self.mesh_mgr = mesh3D_manager(base_mesh, L)

        self.V_c, self.Q_c, self.R_c, self.V_DRT_c = self._get_mesh_and_spaces(self.mesh_mgr.mesh_c)
        self.V_f, self.Q_f, self.R_f, self.V_DRT_f = self._get_mesh_and_spaces(self.mesh_mgr.mesh_f)

        self.V0_c, self.Q0_c, self.R0_c, self.V_DRT0_c = self._get_mesh_and_spaces(self.mesh_mgr.mesh_c, degree=1)
        self.V0_f, self.Q0_f, self.R0_f, self.V_DRT0_f = self._get_mesh_and_spaces(self.mesh_mgr.mesh_f, degree=1)

        self.M_a_csr, self.M_csr, self.B_csr, self.C_csr, self.f_q, self.Mq_csr, self.M_a_DRT_csr, self.M_DRT_csr, self.B_DRT_csr = self.assemble_systems(self.V_f, self.Q_f, self.R_f, self.V_DRT_f, mesh=self.mesh_mgr.mesh_f)
        self.M_a_csr_c, self.M_csr_c, self.B_csr_c, self.C_csr_c, self.f_q_c, self.Mq_csr_c, _, self.M_DRT_csr_c, self.B_DRT_csr_c = self.assemble_systems(self.V_c, self.Q_c, self.R_c, self.V_DRT_c, mesh=self.mesh_mgr.mesh_c)

        V_coeff = FunctionSpace(self.mesh_mgr.mesh_f, "DG", 0)
        W = VectorFunctionSpace(self.mesh_mgr.mesh_f, V_coeff.ufl_element())
        X = Function(W).interpolate(self.mesh_mgr.mesh_f.coordinates)
        self.coeff = Function(V_coeff)
        if self.dim == 2:
            self.coeff.dat.data[:] = self.a(X.dat.data_ro[:, 0], X.dat.data_ro[:, 1])
        elif self.dim == 3:
            self.coeff.dat.data[:] = self.a(X.dat.data_ro[:, 0], X.dat.data_ro[:, 1], X.dat.data_ro[:, 2])

        self.int_dofs_c = self._get_int_dofs(self.V_c)
        self.int_dofs_f = self._get_int_dofs(self.V_f)
        self.N_dof_f = self.V_f.dim()
        self.N_dof_c = self.V_c.dim()
        self.N_int_dof_f = np.size(self.int_dofs_f, 0)
        self.N_int_dof_c = np.size(self.int_dofs_c, 0)
        self.N_DG_dof_f = self.Q_f.dim()
        self.N_DG_dof_c = self.Q_c.dim()
        self.N_r_f = self.R_f.dim()
        self.N_r_c = self.R_c.dim()

        self.proj_RT_DRT_c = self._build_RT_DRT_proj_matrix(self.V_DRT_c, self.V_c)  # Projection matrix from RT to DRT on coarse mesh
        self.proj_RT_DRT_f = self._build_RT_DRT_proj_matrix(self.V_DRT_f, self.V_f)  # Projection matrix from RT to DRT on fine mesh
        
        self.parallel_computation = parallel_computation

        self.dofs_support_c = self._build_dof_support_connectivity(self.V_c)
        self.dofs_support_f = self._build_dof_support_connectivity(self.V_f)
        
        self.H = self._build_hierarchy_matrix(self.Q_c, self.Q_f)
        self.IQ = self.H.T @ self.Mq_csr
        # if self.degree > 1:
        #     self.H0 = self._build_hierarchy_matrix(self.Q0_c, self.Q_f)
        #     self.IQ = self.H0.T @ self.Mq_csr
        self.PRT = self._build_PRT_matrix(self.V_c, self.V_f)
        self.PRT_DRT = self._build_PRT_DRT_matrix(self.V_DRT_c, self.V_DRT_f)
        self.tau_H = self._build_tau_H_matrix()
        self.quasi_IRT = self._build_quasi_int_matrix(self.V_c, self.V_f, self.Q_c, self.Q_f, self.V_DRT_c, self.V_DRT_f)
        self.IRT_canonical = self.mesh_mgr.IRT
        if int_method == "canonical":
            self.IRT = self.IRT_canonical
        elif int_method == "stable":
            self.IRT = self.quasi_IRT
        else:
            raise ValueError("Unknown interpolation method: {}".format(int_method))
        self.IRT = clean_sparse_matrix(self.IRT, tol=1e-12)

        if self.printLevel > 0:
            print(f"V_f dim: {self.V_f.dim()}, V_c dim: {self.V_c.dim()}")

    
    def _get_mesh_and_spaces(self, mesh, degree=None):
        if degree is None:
            degree = self.degree
        V = FunctionSpace(mesh, "RT", degree)
        Q = FunctionSpace(mesh, "DG", degree-1)
        R = FunctionSpace(mesh, 'R', 0)
        V_DRT = FunctionSpace(mesh, "DRT", degree)
        return V, Q, R, V_DRT
    
    def _get_int_dofs(self, V):
        bc = DirichletBC(V, Constant((0.0,)*self.dim), "on_boundary")
        boundary_dofs = bc.nodes
        all_dofs = np.arange(V.dim())
        interior_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        return interior_dofs

    def assemble_systems(self, V, Q, R, V_DRT, mesh=None):
        # Define trial and test functions on subspaces
        sigma = TrialFunction(V)
        tau = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        r = TrialFunction(R)
        s = TestFunction(R)
        sigma_DRT = TrialFunction(V_DRT)
        tau_DRT = TestFunction(V_DRT)
        
        V_coeff = FunctionSpace(mesh, "DG", self.degree-1)
        W = VectorFunctionSpace(mesh, V_coeff.ufl_element())
        X = Function(W).interpolate(mesh.coordinates)
        coeff = Function(V_coeff)
        if self.dim == 2:
            x, y = SpatialCoordinate(mesh)
            coeff.dat.data[:] = self.a(X.dat.data_ro[:, 0], X.dat.data_ro[:, 1])
        elif self.dim == 3:
            x, y, z = SpatialCoordinate(mesh)
            coeff.dat.data[:] = self.a(X.dat.data_ro[:, 0], X.dat.data_ro[:, 1], X.dat.data_ro[:, 2])
        
        # Define each part of the bilinear form separately
        form_M_a = inner(1/coeff*sigma, tau) * dx
        form_M = inner(sigma, tau) * dx
        form_B = inner(div(sigma), q) * dx
        L_form = -self.f(x, y) * q * dx if self.dim == 2 else -self.f(x, y, z) * q * dx
        form_Mq = inner(p, q) * dx
        form_M_a_DRT = inner(1/coeff*sigma_DRT, tau_DRT) * dx
        form_M_DRT = inner(sigma_DRT, tau_DRT) * dx
        form_B_DRT = inner(div(sigma_DRT), q) * dx
        
        # Assemble vector for C and CTs
        vec = assemble(inner(Constant(1.0), q) * dx)
        data = vec.dat.data_ro
        C_csr = sparse.csr_matrix(data[:, None])
        
        # Assemble each part as a PETSc matrix
        M_a_petsc = assemble(form_M_a, mat_type="aij").M.handle
        M_petsc = assemble(form_M, mat_type="aij").M.handle
        B_petsc = assemble(form_B, mat_type="aij").M.handle
        Mq_petsc = assemble(form_Mq, mat_type="aij").M.handle
        M_a_DRT_petsc = assemble(form_M_a_DRT, mat_type="aij").M.handle
        M_DRT_petsc = assemble(form_M_DRT, mat_type="aij").M.handle
        B_DRT_petsc = assemble(form_B_DRT, mat_type="aij").M.handle

        # Convert each to CSR
        M_a_csr = _petsc_to_csr(M_a_petsc)
        M_csr = _petsc_to_csr(M_petsc)
        B_csr = _petsc_to_csr(B_petsc)
        Mq_csr = _petsc_to_csr(Mq_petsc)
        M_a_DRT_csr = _petsc_to_csr(M_a_DRT_petsc)
        M_DRT_csr = _petsc_to_csr(M_DRT_petsc)
        B_DRT_csr = _petsc_to_csr(B_DRT_petsc)

        # Assemble the RHS vector
        f_q = assemble(L_form).dat.data_ro
        return M_a_csr, M_csr, B_csr, C_csr, f_q, Mq_csr, M_a_DRT_csr, M_DRT_csr, B_DRT_csr

    def gen_localized_fine_problem(self, cell_patch_c, rhs_cell_c, f_func=None, g_func=None):
        """
        generate a localized fine-scale mixed problem on a patch of coarse cells.
        This function sets up and solves the fine mixed Poisson problem on a submesh (patch) 
        defined by a set of coarse cells. The equations solved are:
            a^{-1} w + ∇p = g  (in patch)
            -div(w) = f        (in patch)
        with boundary condition w·n = 0 on the patch boundary.
        Parameters:
            cell_patch_c (list or array): Indices of coarse cells defining the patch.
            rhs_cell_c (list or array): Indices of coarse cells where source terms apply.
                                            (For a corrector, this could be the support coarse cell(s) of a basis; 
                                            for source correction, this could be the coarse cell index itself.)
            mesh_mgr (mesh2D_manager): The mesh manager for coarse/fine hierarchy.
            f_func: Source term function f(x) for right-hand side (if solving source problem).
            g_func: Vector source for flux equation (if solving basis corrector problem). 
                    This should be a function giving the fine-scale flux to impose (e.g., coarse basis prolongation).
        Returns:
            w_patch (Function): Fine RT0 flux solution on the patch (extended to full fine mesh DOFs).
            p_patch (Function): Fine DG0 pressure solution on the patch (extended to full fine mesh DOFs).
        """
        # Generate fine submesh for the given patch of coarse cells
        patch_mesh_c, patch_mesh_f, cell_patch_f = self.mesh_mgr.gen_submesh_c_f(cell_patch_c, label_ind=rhs_cell_c, label_prefix='cell_patch')
        # plot_mesh_with_labels_fn(patch_mesh_c)

        V_patch_f, Q_patch_f, R_patch_f, _ = self._get_mesh_and_spaces(patch_mesh_f)
        V_patch_c, Q_patch_c, R_patch_c, _ = self._get_mesh_and_spaces(patch_mesh_c)

        RT_sub_in_par_f = self.mesh_mgr.gen_RT_dof_map_sub_to_par(patch_mesh_f, V_patch_f, self.V_f)
        DG_sub_in_par_f = self.mesh_mgr.gen_DG_dof_map_sub_to_par(patch_mesh_f, Q_patch_f, self.Q_f)

        RT_sub_in_par_c = self.mesh_mgr.gen_RT_dof_map_sub_to_par(patch_mesh_c, V_patch_c, self.V_c)
        DG_sub_in_par_c = self.mesh_mgr.gen_DG_dof_map_sub_to_par(patch_mesh_c, Q_patch_c, self.Q_c)
        
        patch_int_dofs_f = self._get_int_dofs(V_patch_f)
        patch_int_dofs_c = self._get_int_dofs(V_patch_c)
        RT_sub_int_in_par_f = RT_sub_in_par_f[patch_int_dofs_f]
        RT_sub_int_in_par_c = RT_sub_in_par_c[patch_int_dofs_c]

        M_a = self.M_a_csr[RT_sub_int_in_par_f][:, RT_sub_int_in_par_f]
        B = self.B_csr[DG_sub_in_par_f][:, RT_sub_int_in_par_f]

        Nf_patch_f = np.size(M_a, 0)
        Nc_patch_f = np.size(B, 0)

        if g_func is not None:
            # # Represent g_func on patch V (RT0). Assume g_func is given as a numpy array of fine RT dofs or a Function.
            # tau = TestFunction(self.V_f)
            # if self.dim == 2:
            #     x, y = SpatialCoordinate(self.mesh_mgr.mesh_f)
            # elif self.dim == 3:
            #     x, y, z = SpatialCoordinate(self.mesh_mgr.mesh_f)
            # g_form = assemble(inner(1/self.coeff*g_func, tau) * dx(domain=self.mesh_mgr.mesh_f, subdomain_id=rhs_cell_c+1))
            # gRhs1 = g_form.dat.data_ro[RT_sub_int_in_par_f]

            rhs_cells_f = self.mesh_mgr.find_fine_cells_in_coarse([rhs_cell_c])
            rhs_DRT_dofs_f = np.unique(self.V_DRT_f.cell_node_list[rhs_cells_f].flatten())
            rhs_RT_dofs_f = np.unique(self.V_f.cell_node_list[rhs_cells_f].flatten())
            rhs_DRT_arr_local = self.proj_RT_DRT_f[rhs_DRT_dofs_f, :][:, rhs_RT_dofs_f] @ g_func.dat.data[rhs_RT_dofs_f]
            gRhs = (self.proj_RT_DRT_f[rhs_DRT_dofs_f, :][:, RT_sub_int_in_par_f]).T @ (self.M_a_DRT_csr[rhs_DRT_dofs_f, :][:, rhs_DRT_dofs_f]) @ rhs_DRT_arr_local
            # print(f"gRhs: {gRhs}")
            # print(f"gRhs norm: {np.linalg.norm(gRhs)}, gRhs1 norm: {np.linalg.norm(gRhs1)}, gRhs diff norm: {np.linalg.norm(gRhs-gRhs1)}")
        else:
            gRhs = np.zeros(Nf_patch_f)
        if f_func is not None:
            q = TestFunction(self.Q_f)
            if self.dim == 2:
                x, y = SpatialCoordinate(self.mesh_mgr.mesh_f)
                f_eval = f_func(x, y)
            elif self.dim == 3:
                x, y, z = SpatialCoordinate(self.mesh_mgr.mesh_f)
                f_eval = f_func(x, y, z)
            f_form = assemble(f_eval * q * dx(domain=self.mesh_mgr.mesh_f, subdomain_id=rhs_cell_c+1))
            fRhs = f_form.dat.data_ro[DG_sub_in_par_f]
        else:
            fRhs = np.zeros(Nc_patch_f)
        
        if np.size(RT_sub_int_in_par_c, 0) != 0:
            IRT = self.IRT[RT_sub_int_in_par_c][:, RT_sub_int_in_par_f]  # Interpolation from fine RT0 to coarse RT0 on patch
            Nf_patch_c = np.size(IRT, 0)
        else:
            IRT = None
            Nf_patch_c = 0

        IQ = self.IQ[DG_sub_in_par_c][:, DG_sub_in_par_f]  # Interpolation from fine DG0 to coarse DG0 on patch
        Nc_patch_c = IQ.shape[0]

        rhs = np.hstack([gRhs, fRhs, np.zeros(Nf_patch_c), np.zeros(Nc_patch_c)])
        # K = sparse.bmat([[M_a,    B.T,  IRT.T, None],
        #                 [B,    None, None,  IQ.T],
        #                 [IRT,  None, None,  None],
        #                 [None, IQ,   None,  None]]).tocsc()
        # x = sparse.linalg.spsolve(K, rhs)

        # fluxVals = x[:Nf_patch_f]
        # fluxRows = RT_sub_int_in_par_f
        # return fluxVals, fluxRows
        return M_a, B, IRT, IQ, rhs, RT_sub_int_in_par_f


    def gen_corrector_problem(self, dof_ind):
        """
        generate the fine-scale flux corrector problem for a single coarse RT basis function (coarse facet).
        This solves a localized problem on a patch of coarse cells around the support of the given coarse facet 
        to obtain the fine-scale correction flux that removes oscillatory components of that basis.
        Parameters:
            dof_ind (int): Index of the coarse facet (RT0 basis DOF) for which to compute the corrector.
        """
        if self.printLevel > 1:
            print("Coarse basis function {0}".format(dof_ind))

        # Get the coarse cells (support) that share this facet (dofs_support_c gives two cells for interior facets)
        dofs_support_c = self.dofs_support_c  # shape (N_facets_c, 2)
        dof_support = dofs_support_c[dof_ind]
        dof_support = dof_support[dof_support >= 0]  # coarse cells supporting this facet
        if dof_support.size == 0:
            raise ValueError("Facet does not belong to any coarse cell.")
        # Determine patch of coarse cells around the support

        M_list = []
        B_list = []
        IRT_list = []
        IQ_list = []
        rhs_list = []
        flux_rows_list = []

        # Construct the coarse RT basis function as a fine-mesh vector (flux) to use as g_func in local solve.
        PRT = self.PRT
        phi_f = Function(self.V_f).assign(0.0)
        phi_f.dat.data[:] = PRT[:, dof_ind].toarray().squeeze()

        Nc_c = self.mesh_mgr.Nc_c

        for cell_c in dof_support:
            if self.k != np.inf:
                patch_mask_c = np.zeros(Nc_c, dtype=int)
                patch_mask_c[cell_c] = 1
                cells_to_cells_c = self.mesh_mgr.cells_to_cells_c_fn  # coarse adjacency matrix (sparse)
                for _ in range(self.k):
                    new_mask = cells_to_cells_c.dot(patch_mask_c)
                    patch_mask_c = (new_mask > 0).astype(int)            
            else:
                patch_mask_c = np.arange(Nc_c, dtype=int)
            cell_patch_c = np.where(patch_mask_c)[0]
            
            # Solve localized fine problem on the patch with g_func = phi_full (imposing coarse basis on patch), f_func = None
            M_a, B, IRT, IQ, rhs, RT_sub_int_in_par_f = self.gen_localized_fine_problem(cell_patch_c, cell_c, f_func=None, g_func=phi_f)
            
            M_list.append(M_a)
            B_list.append(B)
            IRT_list.append(IRT)
            IQ_list.append(IQ)
            rhs_list.append(rhs)
            flux_rows_list.append(RT_sub_int_in_par_f)
        if self.printLevel > 1:
            print("Generated corrector problem for dof {0}".format(dof_ind))
        return M_list, B_list, IRT_list, IQ_list, rhs_list, flux_rows_list


    def comp_correctors(self):
        """
        Compute fine-scale flux correctors for all interior coarse RT0 basis functions (facets).
        For each interior coarse facet (i.e., facet shared by two coarse cells), this computes a localized corrector flux 
        on a patch of radius k that cancels the fine-scale oscillatory part of that coarse basis.
        Returns:
            (correctors, free_edges_c):
            - correctors: list of fine-mesh RT0 Functions, one per interior coarse facet.
            - free_edges_c: list of indices of interior coarse facets corresponding to these correctors.
        """
        # Identify interior coarse facets (those with two neighboring coarse cells)
        corrector_vals_list = []
        corrector_rows_list = []

        for dof_ind in self.int_dofs_c:
            M_list, B_list, IRT_list, IQ_list, rhs_list, flux_rows_list = self.gen_corrector_problem(dof_ind)

            corrector_vals = []
            corrector_rows = []

            for i in range(len(M_list)):
                M_a = M_list[i]
                B = B_list[i]
                IRT = IRT_list[i]
                IQ = IQ_list[i]
                rhs = rhs_list[i]
                flux_rows = flux_rows_list[i]
                fluxVals = solve_linear_system(M_a, B, IRT, IQ, rhs)

                corrector_vals.extend(fluxVals)
                corrector_rows.extend(flux_rows)

            corrector_vals_list.append(corrector_vals)
            corrector_rows_list.append(corrector_rows)

            if self.printLevel>0:
                print(f"Solved corrector {dof_ind}")
        # Assemble the sparse matrix
        correctors_vals = np.hstack(corrector_vals_list)
        correctors_rows = np.hstack(corrector_rows_list)
        correctors_cols = np.repeat(np.arange(0, self.N_int_dof_c), [len(y) for y in corrector_vals_list])
        return sparse.csc_matrix((correctors_vals, (correctors_rows, correctors_cols)), shape=(self.N_dof_f, self.N_int_dof_c))


    def comp_correctors_parallel(self):
        """
        Compute fine-scale flux correctors for all interior coarse RT0 basis functions (facets) in parallel.
        For each interior coarse facet (i.e., facet shared by two coarse cells), this computes a localized corrector flux 
        on a patch of radius k that cancels the fine-scale oscillatory part of that coarse basis.
        Returns:
            (correctors, free_edges_c):
            - correctors: list of fine-mesh RT0 Functions, one per interior coarse facet.
            - free_edges_c: list of indices of interior coarse facets corresponding to these correctors.
        """
        from joblib import Parallel, delayed
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        # results = Parallel(n_jobs=num_cores)(delayed(self.comp_corrector)(dof_ind) for dof_ind in self.int_dofs_c)
        chunksize = min(int(self.N_int_dof_c / (2 * float(num_cores)) + 0.8), 10)
        

        M_list_all = []
        B_list_all = []
        IRT_list_all = []
        IQ_list_all = []
        rhs_list_all = []
        flux_rows_list_all = []
        for dof_ind in self.int_dofs_c:
            M_list, B_list, IRT_list, IQ_list, rhs_list, flux_rows_list = self.gen_corrector_problem(dof_ind)
            M_list_all.append(M_list)
            B_list_all.append(B_list)
            IRT_list_all.append(IRT_list)
            IQ_list_all.append(IQ_list)
            rhs_list_all.append(rhs_list)
            flux_rows_list_all.append(flux_rows_list)
        
        inputs = list(zip(M_list_all, B_list_all, IRT_list_all, IQ_list_all, rhs_list_all, flux_rows_list_all))

        with multiprocessing.Pool() as pool:
            results = pool.map(worker, inputs, chunksize=chunksize)

        corrector_vals_list = [res[0] for res in results if res is not None]
        corrector_rows_list = [res[1] for res in results if res is not None]

        # Assemble the sparse matrix
        correctors_vals = np.hstack(corrector_vals_list)
        correctors_rows = np.hstack(corrector_rows_list)
        correctors_cols = np.repeat(np.arange(0, self.N_int_dof_c), [len(y) for y in corrector_vals_list])
        return sparse.csc_matrix((correctors_vals, (correctors_rows, correctors_cols)), shape=(self.N_dof_f, self.N_int_dof_c))


    def assembleMSMatrices(self):

        int_dofs_f = self.int_dofs_f
        int_dofs_c = self.int_dofs_c

        self.P = self.PRT[int_dofs_f][:,int_dofs_c]

        self.M_a = self.M_a_csr[int_dofs_f][:, int_dofs_f]
        self.correctors = self.correctors_full[int_dofs_f,:]
        self.Mms = (self.P - self.correctors).T @ self.M_a @ (self.P - self.correctors)
        self.B_c = self.B_csr_c[:, int_dofs_c]
        self.C_c = self.C_csr_c


    def ref(self, return_pressure=False):
        """
        Compute the fine-scale reference solution of the mixed Poisson problem using RT0/DG0 on the fine mesh.
        Solves:
            a^{-1} u + ∇p = 0  (flux equation)
            -div(u) = f       (continuity equation)
        with homogeneous Neumann boundary (u·n = 0 on boundary).
        Returns:
            If return_pressure=False: flux_fine (RT0 Function on fine mesh).
            If return_pressure=True: (flux_fine, pressure_fine) with pressure DG0 Function (mean zero).
        """

        M_a = self.M_a_csr[self.int_dofs_f][:, self.int_dofs_f]
        B = self.B_csr[:, self.int_dofs_f]
        C = self.C_csr
        
        # Assemble the whole system matrix using bmat
        A_csc = bmat([[M_a, B.T, None],
                      [B, None, C],
                      [None, C.T, None]], format='csc')
        b_np = np.concatenate((np.zeros(self.N_int_dof_f), self.f_q, np.zeros(self.N_r_f)))
        print("Solving the reference solution by solving a linear system of size {0}".format(A_csc.shape[0]))
        x = spsolve(A_csc, b_np)

        nv = self.N_int_dof_f
        nq = self.N_DG_dof_f
        # Extract components
        flux_ref = Function(self.V_f).assign(0.0)
        flux_ref.dat.data[self.int_dofs_f] = x[:nv]
        pres_ref = Function(self.Q_f).assign(0.0)
        pres_ref.dat.data[:] = x[nv:nv + nq]
        r_sol = Function(self.R_f)
        r_sol.dat.data[:] = x[nv + nq:]
        if self.printLevel > 0:
            print("The reference solution has been computed.")
        if return_pressure:
            return flux_ref, pres_ref
        else:
            return flux_ref


    def lod(self):
        """
        Compute the Localized Orthogonal Decomposition (LOD) solution of the mixed Poisson problem.
        This uses coarse-scale (RT0) basis with fine-scale correctors and source corrections to construct 
        a multiscale solution efficiently.
        Steps:
        1. Compute fine-scale corrector fluxes for each interior coarse facet (if any, based on k).
        2. Compute fine-scale source correction flux (if ell > 0).
        3. Assemble the multiscale coarse system (M_ms and B_coarse) incorporating correctors.
        4. Solve the coarse saddle-point system for coarse flux coefficients (alpha) and coarse pressures (beta).
        5. Reconstruct the fine-mesh flux solution = (P * alpha - sum_j alpha_j * w_j) + u_correction, and fine pressure from coarse pressure.
        Returns:
            lod_flux (Function): Fine-mesh RT0 flux solution (LOD multiscale flux).
            lod_pressure (Function): Fine-mesh DG0 pressure solution (LOD multiscale pressure, mean zero).
        """
        # Determine if correctors are needed (if coarse mesh is same as fine or k==0, skip correctors)
        no_correctors = (self.mesh_mgr.Nf_c == self.mesh_mgr.Nf_f) or (self.k == 0)
        if no_correctors:
            self.correctors_full = []
        else:
            if self.parallel_computation:
                self.correctors_full = self.comp_correctors_parallel()
            else:
                self.correctors_full = self.comp_correctors()
        # Assemble multiscale matrices (coarse mass and divergence) using correctors
        self.assembleMSMatrices()

        # Compute source term correction on fine mesh
        no_corrections = (self.ell == 0 or self.ell is None)
        if no_corrections:
            self.u_correction_full = Function(self.V_f).assign(0.0).dat.data_ro
        else:
            self.computeSourceCorrections()
        self.u_correction = self.u_correction_full[self.int_dofs_f]
        gRhsC = -self.P.T*self.M_a*self.u_correction
        # print(f"gRhsC max: {np.max(np.abs(gRhsC))}")
        
        fRhsC = self.H.T @ self.f_q
        K = sparse.bmat([[self.Mms, self.B_c.T, None],
                        [self.B_c, None, self.C_c],
                        [None, self.C_c.T, None]]).tocsc()
        rhs = np.hstack([gRhsC, fRhsC, np.zeros(self.C_c.shape[1])])
        # print(K.dtype, rhs)
        x = sparse.linalg.spsolve(K, rhs)
        
        N_dof_lod = self.N_int_dof_c + self.N_DG_dof_c
        flux_lod = Function(self.V_f).assign(0.0)
        phiMod = self.PRT[:,self.int_dofs_c] - self.correctors_full
        flux_lod.dat.data[:] = phiMod*x[:self.N_int_dof_c] + self.u_correction_full

        pres_lod = Function(self.Q_f).assign(0.0)
        pressure = self.H @ x[self.N_int_dof_c:N_dof_lod]
        pres_lod.dat.data[:] = pressure
        return flux_lod, pres_lod


    def _build_tau_H_matrix(self, V_c=None, Q_c=None, v=None) -> csc_matrix:
        """Build the quasi-interpolation matrix from fine RT space to coarse RT space."""
        vals = []
        rows = []
        cols = []
        if V_c is None:
            V_c = self.V_c
            Q_c = self.Q_c
        num_cells_c = Q_c.mesh().num_cells()
        for cell in range(num_cells_c):
            cell_list = [cell]
            cells_in_c_f = self.mesh_mgr.find_fine_cells_in_coarse(cell_list)
            DRT_dofs_c = self.V_DRT_c.cell_node_list[cell_list].flatten()
            DRT_dofs_f = np.unique(self.V_DRT_f.cell_node_list[cells_in_c_f].flatten())
            if np.linalg.norm(np.sort(DRT_dofs_f)-np.sort(self.V_DRT_f.cell_node_list[cells_in_c_f].flatten())) > 1e-12:
                raise ValueError("DRT dof mapping mismatch in tau_H construction.")
            DG_dofs_c = self.Q_c.cell_node_list[cell].flatten()
            DG_dofs_f = self.Q_f.cell_node_list[cells_in_c_f].flatten()
            
            M_cell_c = self.M_DRT_csr_c[DRT_dofs_c][:, DRT_dofs_c]
            B_cell_c = self.B_DRT_csr_c[DG_dofs_c][:, DRT_dofs_c]
            # print(f"cell: {cell}, DRT_dofs_c: {DRT_dofs_c}")
            # Assemble local interpolation matrix
            K_cell = sparse.bmat([[M_cell_c, B_cell_c.T],
                                  [B_cell_c, None]]).toarray()
            # print(np.linalg.det(K_cell), K_cell)
            # inv_K_cell = np.linalg.inv(K_cell)
            PRT_DRT_cell = self.PRT_DRT[DRT_dofs_f][:, DRT_dofs_c]
            M_cell_f = self.M_DRT_csr[DRT_dofs_f][:, DRT_dofs_f]
            f_rhs_cell = PRT_DRT_cell.T @ M_cell_f

            H_cell = self.H[DG_dofs_f][:, DG_dofs_c]
            B_cell_f = self.B_DRT_csr[DG_dofs_f][:, DRT_dofs_f]
            g_rhs_cell = H_cell.T @ B_cell_f

            rhs_cell = np.vstack([f_rhs_cell.toarray(), g_rhs_cell.toarray()])
            if v is not None:
                rhs_cell = rhs_cell @ v[DRT_dofs_f]
            # print(f"K_cell: {K_cell}, rhs_cell: {rhs_cell}")
            x_cell = np.linalg.solve(K_cell, rhs_cell)
            quasi_IRT = x_cell[:len(DRT_dofs_c)]
            vals.extend(quasi_IRT.flatten())
            rows.extend(np.repeat(DRT_dofs_c, len(DRT_dofs_f)))
            cols.extend(np.tile(DRT_dofs_f, len(DRT_dofs_c)))
        quasi_IRT_DRT_DRT = sparse.csc_matrix((vals, (rows, cols)), shape=(self.V_DRT_c.dim(), self.V_DRT_f.dim()), dtype=np.float64)
        tau_H = quasi_IRT_DRT_DRT @ self.proj_RT_DRT_f
        return tau_H


    def _build_quasi_int_matrix(self, V_c, V_f, Q_c, Q_f, V_DRT_c, V_DRT_f,) -> csc_matrix:
        verts_cells_c = self.mesh_mgr._build_cell_vertex_matrix(self.mesh_mgr.mesh_c).T
        cols = []
        rows = []
        data = []
        for vert in range(verts_cells_c.shape[0]):
            vert_patch_c = np.where(verts_cells_c[vert].toarray().flatten())[0]
            
            patch_mesh_c, patch_mesh_f, vert_patch_f = self.mesh_mgr.gen_submesh_c_f(vert_patch_c, label_ind=vert, label_prefix='vert_patch')
            V_patch_c = FunctionSpace(patch_mesh_c, "RT", self.degree)
            V_patch_f = FunctionSpace(patch_mesh_f, "RT", self.degree)
            V_patch_CG_c = FunctionSpace(patch_mesh_c, "CG", 1)
            V_patch_CG_f = FunctionSpace(patch_mesh_f, "CG", 1)
            V_patch_DG_c = FunctionSpace(patch_mesh_c, "DG", self.degree-1)
            V_patch_DG_f = FunctionSpace(patch_mesh_f, "DG", self.degree-1)
            V_patch_DRT_c = FunctionSpace(patch_mesh_c, "DRT", self.degree)
            V_patch_DRT_f = FunctionSpace(patch_mesh_f, "DRT", self.degree)

            RT_sub_in_par_c = self.mesh_mgr.gen_RT_dof_map_sub_to_par(patch_mesh_c, V_patch_c, V_c)
            patch_int_dofs_c = self._get_int_dofs(V_patch_c)
            RT_sub_int_in_par_c = RT_sub_in_par_c[patch_int_dofs_c]
            RT_sub_in_par_f = self.mesh_mgr.gen_RT_dof_map_sub_to_par(patch_mesh_f, V_patch_f, V_f)

            DG_sub_in_par_c = self.mesh_mgr.gen_DG_dof_map_sub_to_par(patch_mesh_c, V_patch_DG_c, Q_c)

            M = self.M_csr_c[RT_sub_int_in_par_c][:, RT_sub_int_in_par_c]
            B = self.B_csr_c[DG_sub_in_par_c][:, RT_sub_int_in_par_c]
            C = self.C_csr_c[DG_sub_in_par_c]
            K_patch = sparse.bmat([[M,    B.T, None],
                                   [B,    None, C],
                                   [None, C.T, None]]).toarray()

            DRT_sub_in_par_c = self.mesh_mgr.gen_DRT_dof_map_sub_to_par(patch_mesh_c, V_patch_DRT_c, V_DRT_c)
            dofs_DRT_patch_c = np.sort(V_DRT_c.cell_node_list[vert_patch_c].flatten())
            if not np.array_equal(np.sort(DRT_sub_in_par_c), dofs_DRT_patch_c):
                raise ValueError("DRT dof mapping mismatch in quasi-interpolation construction.")
            mesh_c = V_c.mesh()
            V_CG_c = FunctionSpace(mesh_c, "CG", 1)
            psi_vert_c = Function(V_CG_c).assign(0.0)
            psi_vert_c.dat.data[vert] = 1.0
            IH_vals = []
            for dof_c in DRT_sub_in_par_c:
                f_c = Function(V_DRT_c).assign(0.0)
                f_c.dat.data[dof_c] = 1.0

                I_psi_f_c = Function(V_DRT_c).interpolate(psi_vert_c*f_c)
                IH_vals.append(I_psi_f_c.dat.data[DRT_sub_in_par_c])

            IH_patch = np.vstack(IH_vals).T  # shape (num_DRT_dofs_in_patch_c, num_dofs_in_patch_c)
            proj_RT_DRT_patch_c = self.proj_RT_DRT_c[DRT_sub_in_par_c][:, RT_sub_int_in_par_c]
            M_DRT_patch_c = self.M_DRT_csr_c[DRT_sub_in_par_c][:, DRT_sub_in_par_c]
            tau_H_patch = self.tau_H[DRT_sub_in_par_c][:, RT_sub_in_par_f]

            f_rhs = proj_RT_DRT_patch_c.T @ M_DRT_patch_c @ IH_patch @ tau_H_patch
            
            if self.dim == 2:
                x, y = SpatialCoordinate(patch_mesh_f)
            elif self.dim == 3:
                x, y, z = SpatialCoordinate(patch_mesh_f)

            sigma_f = TrialFunction(V_patch_DRT_f)
            q_f = TestFunction(V_patch_DG_f)
            psi_vert_patch_f = Function(V_patch_CG_f).interpolate(psi_vert_c)
            g1_form = assemble(inner(psi_vert_patch_f*div(sigma_f), q_f)*dx, mat_type="aij").M.handle
            g1_csr = _petsc_to_csr(g1_form)
            
            mu_vals = []
            for dof_c in range(V_patch_DG_c.dim()):
                mu_patch_c = Function(V_patch_DG_c).assign(0.0)
                mu_patch_c.dat.data[dof_c] = 1.0
                mu_patch_f = Function(V_patch_DG_f).interpolate(mu_patch_c)
                mu_vals.append(mu_patch_f.dat.data)
            mu_basis = np.vstack(mu_vals)  # shape (num_dg_dofs_c, num_dg_dofs_f)
            dofs_DRT_patch_f = self.mesh_mgr.gen_DRT_dof_map_sub_to_par(patch_mesh_f, V_patch_DRT_f, V_DRT_f)
            
            proj_RT_DRT_patch_f = self.proj_RT_DRT_f[dofs_DRT_patch_f][:, RT_sub_in_par_f]
            g1_rhs = mu_basis @ g1_csr.toarray() @ proj_RT_DRT_patch_f 

            psi_vert_patch_c = Function(V_patch_CG_c).interpolate(psi_vert_c)
            sigma_c = TrialFunction(V_patch_DRT_c)
            q_c = TestFunction(V_patch_DG_c)
            g2_form = assemble(inner(dot(grad(psi_vert_patch_c), sigma_c), q_c)*dx, mat_type="aij").M.handle
            g2_rhs = _petsc_to_csr(g2_form).toarray() @ tau_H_patch
            
            rhs = np.vstack([f_rhs, g1_rhs + g2_rhs, np.zeros((C.shape[1], f_rhs.shape[1]))])
            x_patch = np.linalg.solve(K_patch, rhs)
            quasi_IRT_patch = x_patch[:len(RT_sub_int_in_par_c)]

            rows.extend(np.repeat(RT_sub_int_in_par_c, len(RT_sub_in_par_f)))
            cols.extend(np.tile(RT_sub_in_par_f, len(RT_sub_int_in_par_c)))
            data.extend(quasi_IRT_patch.flatten())
        return sparse.csc_matrix((data, (rows, cols)), shape=(V_c.dim(), V_f.dim()), dtype=np.float64)
        

    def _build_RT_DRT_proj_matrix(self, V_DRT, V_RT) -> csc_matrix:
        """Build the projection matrix from RT space to DRT space on the coarse mesh."""
        DRT_map = V_DRT.cell_node_list
        RT_map = V_RT.cell_node_list
        num_cells = V_DRT.mesh().num_cells()
        dofs_per_cell = V_RT.finat_element.space_dimension()
        rows = []
        cols = []
        vals = []

        for cell in range(num_cells):
            for i in range(dofs_per_cell):
                row = DRT_map[cell, i]
                col = RT_map[cell, i]
                rows.append(row)
                cols.append(col)
                vals.append(1)
        proj_RT_DRT = sparse.csc_matrix((vals, (rows, cols)), shape=(V_DRT.dim(), V_RT.dim()), dtype=np.float64)
        return proj_RT_DRT


    def _build_PRT_DRT_matrix(self, V_DRT_c, V_DRT_f) -> csc_matrix:
        rows = []
        cols = []
        vals = []
        tm = TransferManager()
        for cell_c in range(V_DRT_c.mesh().num_cells()):
            cells_in_c_f = self.mesh_mgr.find_fine_cells_in_coarse([cell_c])
            DRT_dofs_f = np.unique(self.V_DRT_f.cell_node_list[cells_in_c_f].flatten())
            DRT_dofs_c = self.V_DRT_c.cell_node_list[cell_c].flatten()
            for dof_c in DRT_dofs_c:
                f_c = Function(self.V_DRT_c).assign(0.0)
                f_c.dat.data[dof_c] = 1.0
                f_f = Function(self.V_DRT_f).assign(0.0)
                tm.prolong(f_c, f_f)
                vals_f = f_f.dat.data[DRT_dofs_f]
                # print(DRT_dofs_f, np.where(f_f.dat.data[:]!=0))
                rows.extend(DRT_dofs_f)
                cols.extend([dof_c]*len(DRT_dofs_f))
                vals.extend(vals_f)
        PRT_DRT = sparse.csc_matrix((vals, (rows, cols)), shape=(V_DRT_f.dim(), V_DRT_c.dim()), dtype=np.float64)
        return PRT_DRT


    def _build_PRT_matrix(self, V_c, V_f) -> csc_matrix:
        rows = []
        cols = []
        vals = []
        tm = TransferManager()
        for dof_c in range(V_c.dim()):
            support_cells_c = self.dofs_support_c[dof_c]
            cells_in_c_f = self.mesh_mgr.find_fine_cells_in_coarse(support_cells_c[support_cells_c>=0])
            dofs_f = np.unique(V_f.cell_node_list[cells_in_c_f].flatten())
            f_c = Function(V_c).assign(0.0)
            f_c.dat.data[dof_c] = 1.0
            f_f = Function(V_f).assign(0.0)
            tm.prolong(f_c, f_f)
            vals_f = f_f.dat.data[dofs_f]
            rows.extend(dofs_f)
            cols.extend([dof_c]*len(dofs_f))
            vals.extend(vals_f)
        PRT = sparse.csc_matrix((vals, (rows, cols)), shape=(V_f.dim(), V_c.dim()), dtype=np.float64)
        return PRT


    def _build_hierarchy_matrix(self, Q_c, Q_f) -> csc_matrix:
        rows = []
        cols = []
        vals = []
        tm = TransferManager()
        num_cells_c = Q_c.mesh().num_cells()

        for cell in range(num_cells_c):
            cell_list = [cell]
            cells_in_c_f = self.mesh_mgr.find_fine_cells_in_coarse(cell_list)
            dofs_f = np.unique(Q_f.cell_node_list[cells_in_c_f].flatten())
            DG_dofs_in_cell = Q_c.cell_node_list[cell].flatten()

            for dof_c in DG_dofs_in_cell:
                f_c = Function(Q_c).assign(0.0)
                f_c.dat.data[dof_c] = 1.0
                f_f = Function(Q_f).assign(0.0)
                tm.prolong(f_c, f_f)
                vals_f = f_f.dat.data[dofs_f]
                rows.extend(dofs_f)
                cols.extend([dof_c]*len(dofs_f))
                vals.extend(vals_f)
        H = sparse.csc_matrix((vals, (rows, cols)), shape=(Q_f.dim(), Q_c.dim()), dtype=np.float64)
        return H


    def _build_dof_support_connectivity(self, V):
        """Build the DOF-to-Supporting-cells connectivity for a given function space V."""
        cell_dof = V.cell_node_map().values_with_halo
        N_dof = V.dim()
        N_cells = V.mesh().num_cells()
        N_dof_per_cell = cell_dof.shape[1]
        vals = np.ones(N_cells * N_dof_per_cell, dtype=np.int32)
        dofs_cells_mat = sparse.csc_matrix((vals, (cell_dof.flatten(), np.repeat(np.arange(N_cells), N_dof_per_cell))),
                                           shape=(N_dof, N_cells), dtype=np.int32)
        dofs_support = -1 * np.ones((N_dof, 2), dtype=np.int32)
        for dof in range(N_dof):
            supporting_cells = dofs_cells_mat.getrow(dof).nonzero()[1]
            if len(supporting_cells) == 1:
                dofs_support[dof] = np.array([-1, supporting_cells[0]], dtype=np.int32)
            elif len(supporting_cells) == 0:
                dofs_support[dof] = np.array([-1, -1], dtype=np.int32)
            else:
                dofs_support[dof] = np.array(supporting_cells[:2], dtype=np.int32)
        return dofs_support