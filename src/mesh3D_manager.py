from firedrake import *
import numpy as np
from scipy import sparse
import pyop2
from petsc4py import PETSc
from scipy import sparse
from scipy.sparse import csc_matrix




class mesh3D_manager:
    """
    A class to manage mesh hierarchy, adjacency matrices, function spaces, and submesh generation
    for Localized Orthogonal Decomposition (LOD) with RT element in Firedrake.
    """
    
    def __init__(self, base_mesh, L):
        """
        Initialize the mesh hierarchy.

        Parameters:
        - base_mesh: The base mesh (e.g., UnitSquareMesh).
        - L: Number of refinement levels (hierarchy has L levels in total).
        """
        self.hierarchy = MeshHierarchy(base_mesh, L)  # hierarchy[0]: coarse, hierarchy[-1]: fine
        self.mesh_c = self.hierarchy[0]
        self.mesh_f = self.hierarchy[-1]
        self.facets_verts_c_fn, self.facets_cells_c_fn = self._build_facets_related_connectivity(self.mesh_c)
        self.facets_verts_f_fn, self.facets_cells_f_fn = self._build_facets_related_connectivity(self.mesh_f)
        self.verts_coor_c_fn = self._build_vertex_coordinates(self.mesh_c)
        self.verts_coor_f_fn = self._build_vertex_coordinates(self.mesh_f)
        self.cells_verts_c_fn = self._build_cell_vertex_connectivity(self.mesh_c)
        self.cells_verts_f_fn = self._build_cell_vertex_connectivity(self.mesh_f)
        self.cells_facets_c_fn = self._build_cell_facet_connectivity(self.mesh_c)
        self.cells_facets_f_fn = self._build_cell_facet_connectivity(self.mesh_f)
        self.L = L
        self.dim = self.mesh_c.topological_dimension()
        self.Nf_c = self.mesh_c.num_facets()  # number of coarse facets
        self.Nf_f = self.mesh_f.num_facets()  # number of fine facets
        self.Nc_c = self.mesh_c.num_cells()   # number of coarse cells
        self.Nc_f = self.mesh_f.num_cells()   # number of fine cells
        self.H = self._build_hierarchy_matrix()  # hierarchy matrix from coarse cells to fine cells
        self.IQ = self.H.T.copy()  # interpolation matrix from fine DG0 to coarse DG0
        # self.PRT = self._build_PRT_matrix()  # prolongation matrix from coarse RT to fine RT
        self.IRT = self._build_interpolation_matrix()  # canonical RT interpolation matrix from fine RT to coarse RT
        # self.ORI_c = self._build_cell_facet_orientation(self.mesh_c)  # Cell-facet orientation on coarse mesh
        # self.ORI_f = self._build_cell_facet_orientation(self.mesh_f)  # Cell-facet orientation on fine mesh
        # Precompute adjacency on coarse
        self.cells_to_cells_c_fn = self._build_cell_cell_matrix(self.mesh_c)


        # DG0 for marking
        self.DG0_c = FunctionSpace(self.mesh_c, "DG", 0)
        self.DG0_f = FunctionSpace(self.mesh_f, "DG", 0)

    def _build_facets_related_connectivity(self, mesh_h):
        dm = mesh_h.topology_dm
        dim = mesh_h.topological_dimension()
        # Get index ranges for facets in coarse and fine meshes
        Nf = mesh_h.num_facets()
        facets_verts = np.ones((Nf, dim), dtype=np.int32)
        facets_cells = -1 * np.ones((Nf, 2), dtype=np.int32)

        cell_dm_to_fn_map = gen_cell_dm_to_fn_map(mesh_h)
        facet_fn_to_dm_map = gen_facet_fn_to_dm_map(mesh_h)
        vertex_dm_to_fn_map = gen_vertex_dm_to_fn_map(mesh_h)
        
        for facet_fn in range(Nf):
            facet_dm = facet_fn_to_dm_map[facet_fn]

            if mesh_h.cell_dimension() == 2:
                facet_verts_dm = dm.getCone(facet_dm)
            elif mesh_h.cell_dimension() == 3:
                facet_edges_dm = dm.getCone(facet_dm)
                facet_verts_dm = np.unique([dm.getCone(edge) for edge in facet_edges_dm])

            facet_verts_fn = np.sort([vertex_dm_to_fn_map[v_dm] for v_dm in facet_verts_dm])
            facets_verts[facet_fn] = facet_verts_fn

            support_cells_dm = dm.getSupport(facet_dm)
            # if len(support_cells_dm) <= 1:
            #     print(facet_fn)
            support_cells_fn = np.array([cell_dm_to_fn_map[cell_dm] for cell_dm in support_cells_dm])
            if len(support_cells_fn) != len(support_cells_dm):
                print(f"len(support_cells_fn) != len(support_cells_dm) for facet_fn: {facet_fn}")
            # Ensure exactly two entries (use -1 for boundary case)
            if len(support_cells_fn) == 1:
                support_cells_fn = np.array([support_cells_fn[0], -1], dtype=np.int32)
            elif len(support_cells_fn) == 0:
                support_cells_fn = np.array([-1, -1], dtype=np.int32)
            facets_cells[facet_fn] = np.sort(support_cells_fn)

        return facets_verts, facets_cells

    def _build_vertex_coordinates(self, mesh_h):
        verts_coor_fn = mesh_h.coordinates.dat.data.copy()
        return verts_coor_fn
    
    def _build_cell_vertex_connectivity(self, mesh_h):
        vertex_fs = FunctionSpace(mesh_h, "CG", 1)
        cells_verts_fn = vertex_fs.cell_node_map().values
        return cells_verts_fn

    def _build_cell_facet_connectivity(self, mesh_h):
        """Build the cell-to-facet connectivity for the given mesh `mesh_h`."""
        V = FunctionSpace(mesh_h, "RT", 1)
        return np.sort(V.cell_node_map().values, 1)


    def _build_hierarchy_matrix(self) -> csc_matrix:
        """Build the hierarchy matrix from coarse cells to fine cells."""
        H_rows, H_cols = [], []
        for cell_c in range(self.Nc_c):
            fine_cells_in_c = self.find_fine_cells_in_coarse([cell_c])
            H_rows.extend(fine_cells_in_c)
            H_cols.extend([cell_c for _ in fine_cells_in_c])
        vals = [1 for _ in H_rows]
        H = sparse.csc_matrix((vals, (H_rows, H_cols)), shape=(self.Nc_f, self.Nc_c), dtype=np.float64)
        return H
    
    def _build_PRT_matrix(self) -> csc_matrix:
        """Build the prolongation matrix from coarse RT space to fine RT space."""
        # Assemble PRT entries
        PRT_rows, PRT_cols, PRT_vals = [], [], []
        
        for facet_c_fn in range(self.Nf_c):
            facet_verts_fn = self.facets_verts_c_fn[facet_c_fn]
            verts_coor = self.verts_coor_c_fn[facet_verts_fn]
            
            P0_c = verts_coor[0]
            P1_c = verts_coor[1]
            tangent_c = P1_c - P0_c
            measure_facet_c = np.linalg.norm(tangent_c)
            normal_c = np.cross(np.append(tangent_c, 0.0), [0, 0, 1])[:2]
            normal_c /= measure_facet_c
            facet_mid_c = 0.5 * (P1_c + P0_c)
            support_cells_fn = self.facets_cells_c_fn[facet_c_fn]

            # Collect local contributions for this coarse edge
            local_rows = []  # fine edge indices
            local_vals = []
            for i, cell_fn in enumerate(support_cells_fn):
                if cell_fn < 0:
                    continue
                cell_verts_fn = self.cells_verts_c_fn[cell_fn]
                
                # Opposite vertex (not on the coarse edge)
                opp_vert_fn = [v for v in cell_verts_fn if v not in facet_verts_fn][0]
                opp_v_coor = self.verts_coor_c_fn[opp_vert_fn]
                
                P_mid = opp_v_coor - facet_mid_c
                sigma = (-1)*np.sign(np.dot(normal_c, P_mid))
                
                # Coarse cell area (|T|). Compute using cross product of edge vectors.
                # Triangle vertices coordinates:
                A = opp_v_coor
                B = P0_c
                C = P1_c
                measure_cell = 0.5 * abs(np.cross(B - A, C - A))
                
                # Collect all fine cells inside this coarse cell
                cells_fine_in_coarse_fn = self.find_fine_cells_in_coarse([cell_fn])
                facets_fine_in_coarse_fn = np.unique(np.hstack([self.cells_facets_f_fn[cell_f] for cell_f in cells_fine_in_coarse_fn]))
                for facet_f_fn in facets_fine_in_coarse_fn:
                    facet_verts_f_fn = self.facets_verts_f_fn[facet_f_fn]
                    verts_coor_f = self.verts_coor_f_fn[facet_verts_f_fn]
                    P0_f = verts_coor_f[0]
                    P1_f = verts_coor_f[1]
                    tangent_f = P1_f - P0_f
                    measure_facet_f = np.linalg.norm(tangent_f)
                    normal_f = np.cross(np.append(tangent_f, 0.0), [0, 0, 1])[:2]
                    normal_f /= measure_facet_f
                    # facet_centroid = 0.5*(P0_f + P1_f)
                    facet_centroid = verts_coor_f.mean(axis=0)
                    # Value of coarse RT basis at the midpoint projected on this normal_f
                    # This line needs to be checked for the orientation.
                    val = sigma * (measure_facet_f / (2 * measure_cell)) * np.dot(normal_f, (facet_centroid - opp_v_coor))
                    local_rows.append(facet_f_fn)
                    local_vals.append(val)
            
            if local_rows:
                unique_facets_f_fn, idx = np.unique(local_rows, return_index=True)
                unique_vals = [local_vals[j] for j in idx]
                # Append to global lists
                PRT_rows.extend(unique_facets_f_fn.tolist())
                PRT_cols.extend([facet_c_fn] * unique_facets_f_fn.size)
                PRT_vals.extend(unique_vals)
        PRT = sparse.csc_matrix((PRT_vals, (PRT_rows, PRT_cols)), shape=(self.Nf_f, self.Nf_c), dtype=np.float64)
        return PRT

    def _build_interpolation_matrix(self) -> csc_matrix:
        """Build the canonical interpolation matrix from fine RT space to coarse RT space."""
        eps = 1e-10

        IRT_rows, IRT_cols, IRT_vals = [], [], []

        # Loop over each coarse face (facet in 3D)
        for facet_c_fn in range(self.Nf_c):
            # Coarse face vertices (assuming triangular faces)
            verts_c = self.facets_verts_c_fn[facet_c_fn]
            if len(verts_c) != 3:
                raise ValueError("Assuming triangular faces")
            P0 = self.verts_coor_c_fn[verts_c[0]]
            P1 = self.verts_coor_c_fn[verts_c[1]]
            P2 = self.verts_coor_c_fn[verts_c[2]]

            v1 = P1 - P0
            v2 = P2 - P0
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < eps:
                continue  # degenerate face

            area_total = norm_n / 2

            # Get coarse cells sharing this face and all fine cells inside them
            coarse_cells = [c for c in self.facets_cells_c_fn[facet_c_fn] if c >= 0]
            if not coarse_cells:
                continue
            fine_cells = []
            for c in coarse_cells:
                fine_cells_in_c = self.find_fine_cells_in_coarse([c])
                fine_cells.extend(fine_cells_in_c)
            fine_cells = np.unique(fine_cells)

            # Collect all fine faces in these fine cells (candidates)
            if fine_cells.size == 0:
                continue
            
            fine_facets = np.unique(self.cells_facets_f_fn[fine_cells].flatten())

            # Compute midpoints of candidate fine faces
            facet_midpoints = np.mean(self.verts_coor_f_fn[self.facets_verts_f_fn[fine_facets]], axis=1)

            # Check on plane: dist = |dot(m - P0, n)| / norm_n <= eps
            dist = np.abs(np.dot(facet_midpoints - P0, n)) / norm_n
            on_plane = dist <= eps

            # Bounding box check
            P_coarse = np.vstack([P0, P1, P2])
            min_p = np.min(P_coarse, axis=0) - eps
            max_p = np.max(P_coarse, axis=0) + eps
            in_bbox = np.all((min_p <= facet_midpoints) & (facet_midpoints <= max_p), axis=1)

            # Candidates
            candidate_mask = on_plane & in_bbox
            candidate_facets = fine_facets[candidate_mask]
            candidate_midpoints = facet_midpoints[candidate_mask]

            # For each candidate, check inside triangle using barycentric and compute val
            for fine_face, m in zip(candidate_facets, candidate_midpoints):
                # Areas for barycentric
                a0 = np.linalg.norm(np.cross(P1 - m, P2 - m)) / 2
                a1 = np.linalg.norm(np.cross(P2 - m, P0 - m)) / 2
                a2 = np.linalg.norm(np.cross(P0 - m, P1 - m)) / 2
                lambdas = np.array([a0, a1, a2]) / area_total
                if np.all(lambdas >= -eps) and abs(np.sum(lambdas) - 1) <= eps:
                    # Inside, compute sign
                    fine_verts = self.facets_verts_f_fn[fine_face]
                    fv0 = self.verts_coor_f_fn[fine_verts[0]]
                    fv1 = self.verts_coor_f_fn[fine_verts[1]]
                    fv2 = self.verts_coor_f_fn[fine_verts[2]]
                    fine_v1 = fv1 - fv0
                    fine_v2 = fv2 - fv0
                    fine_n = np.cross(fine_v1, fine_v2)
                    val = np.sign(np.dot(fine_n, n))
                    IRT_rows.append(facet_c_fn)
                    IRT_cols.append(fine_face)
                    IRT_vals.append(val)
                else:
                    raise ValueError("mid points outside of the facet.")

        # Build sparse matrix (coarse faces x fine faces)
        IRT = sparse.csc_matrix((IRT_vals, (IRT_rows, IRT_cols)), shape=(self.Nf_c, self.Nf_f), dtype=np.float64)
        return IRT


    def _build_cell_vertex_matrix(self, mesh_h):
        """Create cell-to-vertex incidence matrix (Nt x Np) for function index, starting from 0."""
        vertex_fs = FunctionSpace(mesh_h, "CG", 1)
        t = vertex_fs.cell_node_map().values
        Nt = mesh_h.num_cells()
        Np = mesh_h.num_vertices()
        row = np.repeat(np.arange(Nt), self.dim+1)  # 3 for cells
        col = t.flatten()
        val = np.ones((self.dim+1) * Nt, dtype=int)
        tp_fn = sparse.csc_matrix((val, (row, col)), shape=(Nt, Np), dtype=int)
        return tp_fn

    def _build_cell_cell_matrix(self, mesh_h):
        """Create cell-to-cell adjacency via vertices (Nt x Nt) for function index, starting from 0."""
        tp_fn = self._build_cell_vertex_matrix(mesh_h)
        cells_to_cells_fn = tp_fn @ tp_fn.T
        return cells_to_cells_fn
    
    def _build_cell_facet_orientation(self, mesh_h):
        """
        Build the cell-facet orientation matrix for the given mesh `mesh_h`.
        The returned matrix ORI is of size Nc x (dim+1), where Nc is the number of cells.
        ORI[i, j] = 0 if the normal of facet cells_facets_fn[i, j] points *away from* cell i (outward normal),
        and ORI[i, j] = 1 if it points *into* cell i.
        """

        Nc = mesh_h.num_cells()
        dim = mesh_h.topological_dimension()  # 2 for triangle mesh, 3 for tetrahedral mesh
        
        # Get basic mesh connectivity and geometry
        cells_facets_fn = self._build_cell_facet_connectivity(mesh_h)    # shape: (Nc, dim+1)
        facets_verts_fn, facets_cells_fn = self._build_facets_related_connectivity(mesh_h)
        cells_verts_fn = self._build_cell_vertex_connectivity(mesh_h)
        verts_coor = self._build_vertex_coordinates(mesh_h)             # shape: (Nv, dim)
        
        # Initialize orientation matrix entries
        ORI = np.zeros((Nc, dim+1), dtype=int)

        # compute orientation for each cell-facet pair
        for cell in range(Nc):
            cell_facets = cells_facets_fn[cell]
            cell_verts_set = set(cells_verts_fn[cell])
            # Loop over each facet index in the cell's facet list (as given by cell_facets)
            for j, facet in enumerate(cell_facets):
                # Vertices of this facet
                facet_verts = facets_verts_fn[facet]
                # Compute facet centroid coordinates
                face_coords = verts_coor[list(facet_verts)]
                facet_centroid = face_coords.mean(axis=0)
                # Get the one vertex of cell i that is not on this facet (the "opposite vertex")
                face_verts_set = set(facet_verts)
                opp_vertex_id = (cell_verts_set - face_verts_set).pop()  # opposite vertex ID
                opp_vertex_coor = verts_coor[opp_vertex_id]

                # Compute the facet normal vector
                if dim == 2:
                    # Edge from first to second vertex
                    v0, v1 = face_coords[0], face_coords[1]
                    tangent = v1 - v0
                    # Cross product with out-of-plane unit vector (0,0,1) to get in-plane normal
                    measure_facet = np.linalg.norm(tangent)
                    normal = np.cross(np.append(tangent, 0.0), [0, 0, 1])[:2]
                    normal /= measure_facet
                else:  # dim == 3
                    # Two edges on the triangular face
                    v0, v1, v2 = face_coords[0], face_coords[1], face_coords[2]
                    e1 = v1 - v0
                    e2 = v2 - v0
                    # Normal (not normalized) perpendicular to the face
                    normal = np.cross(e1, e2)

                # Vector from facet centroid to the cell's opposite vertex
                to_cell_vec = opp_vertex_coor - facet_centroid
                # Determine orientation: 0 if normal points outward from cell, +1 if inward
                dot_val = np.dot(normal, to_cell_vec)
                ORI[cell, j] = int(0.5*(np.sign(dot_val) + 1))

        return ORI
    
    def _build_facets_cells_patch(self, mesh_h, cells_facets_h=None, ORI_h=None, cells_ind=None):
        """
        Build the facet-to-cell incidence matrix for a mesh patch.
        
        Parameters
        ----------
        mesh_h : firedrake.Mesh
            The Firedrake mesh hierarchy level to process.
        cells_facets_h : np.ndarray, optional
            Cell-to-facet connectivity array of shape (Nc, dim+1).
        ORI_h : np.ndarray, optional
            Orientation matrix from _build_cell_facet_orientation.
        cells_ind : np.ndarray, optional
            Indices of cells in the patch (default: all cells).
        
        Returns
        -------
        facets_cells_patch : np.ndarray
            (Nf_patch, 2) array, each row giving the cell indices on
            the negative-normal and positive-normal sides of a facet,
            or -1 for missing neighbors.
        facets_in_patch : np.ndarray
            The global indices of facets included in the patch.
        """
        # Determine triangles in the patch
        if cells_ind is None:
            cells_ind = np.arange(mesh_h.num_cells())  # all triangle indices
        else:
            cells_ind = np.asarray(cells_ind)

        # Precomputed connectivity arrays for facets (edges)
        if cells_facets_h is None:
            cells_facets = self._build_cell_facet_connectivity(mesh_h)
        else:
            cells_facets = cells_facets_h
        if ORI_h is None:
            ORI = self._build_cell_facet_orientation(mesh_h)
        else:
            ORI = ORI_h
            
        # Patch subset
        cells_facets_patch = cells_facets[cells_ind, :]
        ORI_patch = ORI[cells_ind, :]
        
        # Facets in the patch
        facets_in_patch1 = np.unique(cells_facets_patch.ravel())
        facets_in_patch1 = facets_in_patch1[facets_in_patch1 != -1]
        facets_in_patch1 = np.array(facets_in_patch1, dtype=int)
        
        Nf = mesh_h.num_facets()
        data = np.hstack([cells_ind, cells_ind, cells_ind])    
        facets_cells_patch = sparse.csc_matrix((data+1, \
                                            (np.hstack([cells_facets_patch[:,0], \
                                                        cells_facets_patch[:,1], \
                                                        cells_facets_patch[:,2]]), \
                                                np.hstack([ORI_patch[:,0], \
                                                        ORI_patch[:,1], \
                                                        ORI_patch[:,2]]))), shape=(Nf, 2), dtype=int)
        facets_in_patch = np.unique(facets_cells_patch.indices)
        
        if not np.array_equal(facets_in_patch1, facets_in_patch):
            raise ValueError("facets_in_patch mismatch detected.")

        facets_cells_patch = facets_cells_patch[facets_in_patch, :].toarray() - 1
        facets_cells_patch = np.array(facets_cells_patch)
        return facets_cells_patch, facets_in_patch

    def gen_map_sub_to_parent(self, submesh, V_sub, V_par):
        """
        This function now can be abandoned in favor of Firedrake's built-in
        functionality for mapping functions between submeshes and parent meshes: 
        f_parent.interpolate(f_sub, allow_missing_dofs=True).
        """ 

        """
        Generate a mapper function to copy a function from the submesh space V_sub
        (RT or broken RT) to the parent space V_par. Assumes identical elements
        on submesh and parent. Works exactly for broken RT; for conforming RT,
        assumes normal component on subdomain boundary is zero for clean extension.

        Parameters:
        - submesh: The Firedrake Submesh object.
        - V_sub: FunctionSpace on submesh (RT or BrokenElement(RT)).
        - V_par: FunctionSpace on parent mesh (same element as V_sub).

        Returns:
        - A function that takes f_sub (Function in V_sub) and returns f_parent
        (Function in V_par) with mapped values (zero elsewhere).
        """
        
        # Check compatibility
        if V_sub.ufl_element() != V_par.ufl_element():
            raise ValueError("V_sub and V_par must use identical elements.")

        # Get maps (precompute)
        child_to_parent_map = submesh.topology.submesh_child_cell_parent_cell_map  # pyop2.Map: sub_cells -> parent_cells
        sub_cell_node_map = V_sub.cell_node_map()
        parent_cell_node_map = V_par.cell_node_map()

        # Compose: parent_dofs = parent_cell_node_map o child_to_parent_map
        composed_map = pyop2.ComposedMap(parent_cell_node_map, child_to_parent_map)

        # DOFs per cell
        dofs_per_cell = V_sub.finat_element.space_dimension()

        # Kernel to copy DOFs (assumes identical function ordering)
        kernel_code = f"""
        void copy_dofs(double parent[{dofs_per_cell}], const double sub[{dofs_per_cell}]) {{
            for (int i = 0; i < {dofs_per_cell}; ++i) {{
                parent[i] = sub[i];
            }}
        }}
        """
        kernel = pyop2.Kernel(kernel_code, "copy_dofs")

        def mapper(f_sub):
            if f_sub.function_space() != V_sub:
                raise ValueError("f_sub must be in V_sub.")

            f_parent = Function(V_par)  # Initialized to zero

            # Parallel loop over submesh cells
            pyop2.par_loop(
                kernel,
                submesh.cell_set,  # Iterate over submesh cells
                f_parent.dat(pyop2.WRITE, composed_map),  # Write to parent DOFs via composition
                f_sub.dat(pyop2.READ, sub_cell_node_map)  # Read from sub DOFs
            )

            # Halo exchange (for parallel consistency; no-op in serial)
            if COMM_WORLD.size > 1:
                f_parent.dat.local_to_global_begin(pyop2.INC)
                f_parent.dat.local_to_global_end(pyop2.INC)
                f_parent.dat.global_to_fn_begin(pyop2.WRITE)
                f_parent.dat.global_to_fn_end(pyop2.WRITE)

            return f_parent
        return mapper
    
    def find_fine_cells_in_coarse(self, coarse_cells_fn):
        # Propagate coarse cells to fine level
        current_cells = coarse_cells_fn
        for i in range(self.L):
            c2f = self.hierarchy.coarse_to_fine_cells[i]  # num_c x num_children array
            current_cells = np.concatenate(c2f[current_cells])
        # assert np.unique(current_cells).size == current_cells.size
        fine_cells_fn = current_cells  # fine cells in patch
        return fine_cells_fn
    
    def gen_submesh_c_f(self, inds_patch_c, label_ind, label_prefix='cell_patch'):
        label = f'{label_prefix}_{label_ind}'
        # Mark coarse patch
        indicator_c = Function(self.DG0_c).assign(0)
        indicator_c.dat.data[inds_patch_c] = 1
        self.mesh_c.mark_entities(indicator_c, label_ind+1, label)
        patch_mesh_c = Submesh(self.mesh_c, self.dim, label_ind+1, label)

        cell_patch_f = self.find_fine_cells_in_coarse(inds_patch_c)
        # Mark fine patch
        indicator_f = Function(self.DG0_f).assign(0)
        indicator_f.dat.data[cell_patch_f] = 1
        self.mesh_f.mark_entities(indicator_f, label_ind+1, label)
        patch_mesh_f = Submesh(self.mesh_f, self.dim, label_ind+1, label)

        if label_prefix == 'cell_patch':
            cell_f = self.find_fine_cells_in_coarse([label_ind])
            indicator_cell_f = Function(self.DG0_f).assign(0)
            indicator_cell_f.dat.data[cell_f] = 1
            self.mesh_f.mark_entities(indicator_cell_f, label_ind+1)
        return patch_mesh_c, patch_mesh_f, cell_patch_f


    def gen_RT_dof_map_sub_to_par(self, submesh, V_sub, V_par):        
        # Assuming submesh, V_sub, V_par are defined as in the provided code
        child_to_parent_map = submesh.topology.submesh_child_cell_parent_cell_map
        sub_cell_node_map = V_sub.cell_node_map()
        parent_cell_node_map = V_par.cell_node_map()
        composed_map = pyop2.ComposedMap(parent_cell_node_map, child_to_parent_map)

        dofs_per_cell = V_sub.finat_element.space_dimension()

        # Extract values (use with_halo for parallel safety)
        sub_values = sub_cell_node_map.values_with_halo
        composed_values = composed_map.values_with_halo

        # Initialize map array for sub DoF to parent DoF
        sub_to_parent = np.full(V_sub.dim(), -1, dtype=np.int32)

        for cell_idx in range(sub_values.shape[0]):
            for local_dof in range(dofs_per_cell):
                sub_dof = sub_values[cell_idx, local_dof]
                parent_dof = composed_values[cell_idx, local_dof]
                sub_to_parent[sub_dof] = parent_dof
        
        # parent_interior_set = set(parent_interior_dofs)
        # interior_map = {
        #     sub_dof: sub_to_parent[sub_dof]
        #     for sub_dof in sub_interior_dofs
        #     if sub_to_parent[sub_dof] in parent_interior_set
        # }
        # return sub_to_parent[sub_interior_dofs]
        return sub_to_parent
    
    def gen_DRT_dof_map_sub_to_par(self, submesh, V_DRT_sub, V_DRT_par):        
        # Assuming submesh, V_DRT_sub, V_DRT_par are defined as in the provided code
        child_to_parent_map = submesh.topology.submesh_child_cell_parent_cell_map
        sub_cell_node_map = V_DRT_sub.cell_node_map()
        parent_cell_node_map = V_DRT_par.cell_node_map()
        composed_map = pyop2.ComposedMap(parent_cell_node_map, child_to_parent_map)

        dofs_per_cell = V_DRT_sub.finat_element.space_dimension()

        # Extract values (use with_halo for parallel safety)
        sub_values = sub_cell_node_map.values_with_halo
        composed_values = composed_map.values_with_halo

        # Initialize map array for sub DoF to parent DoF
        sub_to_parent = np.full(V_DRT_sub.dim(), -1, dtype=np.int32)

        for cell_idx in range(sub_values.shape[0]):
            for local_dof in range(dofs_per_cell):
                sub_dof = sub_values[cell_idx, local_dof]
                parent_dof = composed_values[cell_idx, local_dof]
                sub_to_parent[sub_dof] = parent_dof
        return sub_to_parent


    def gen_DG_dof_map_sub_to_par(self, submesh, Q_sub, Q_par):
        child_to_parent_map = submesh.topology.submesh_child_cell_parent_cell_map
        sub_cell_node_map = Q_sub.cell_node_map()
        parent_cell_node_map = Q_par.cell_node_map()
        composed_map = pyop2.ComposedMap(parent_cell_node_map, child_to_parent_map)
        
        dofs_per_cell = Q_sub.finat_element.space_dimension()
        
        # Extract with halo for parallel
        sub_values = sub_cell_node_map.values_with_halo
        composed_values = composed_map.values_with_halo
        
        # Sub DoF to parent DoF array
        sub_to_parent = np.full(Q_sub.dim(), -1, dtype=np.int32)
        
        for cell_idx in range(sub_values.shape[0]):
            for local_dof in range(dofs_per_cell):
                sub_dof = sub_values[cell_idx, local_dof]
                parent_dof = composed_values[cell_idx, local_dof]
                sub_to_parent[sub_dof] = parent_dof
        return sub_to_parent


import numpy as np


def gen_cell_fn_to_dm_map(mesh_h):
    """
    Return an array which represewnts a mapping from function index of the cell to the dm index of the cell.
    """
    dm = mesh_h.topology_dm
    cell_start, cell_end = dm.getDepthStratum(mesh_h.cell_dimension())
    cell_fn_to_dm_map = {}
    for ind_dm in range(cell_start, cell_end):
        ind_fn = mesh_h._cell_numbering.getOffset(ind_dm)
        cell_fn_to_dm_map[ind_fn] = int(ind_dm)
    return cell_fn_to_dm_map


def gen_cell_dm_to_fn_map(mesh_h):
    """
    Return an array which represewnts a mapping from dm index of the cell to the function index of the cell. 
    Map `ind_dm - cell_start` to `ind_fn`.
    """
    dm = mesh_h.topology_dm
    cell_start, cell_end = dm.getDepthStratum(mesh_h.cell_dimension())
    cell_dm_to_fn_map = {}
    for ind_dm in range(cell_start, cell_end):
        ind_fn = mesh_h._cell_numbering.getOffset(ind_dm)
        cell_dm_to_fn_map[ind_dm] = int(ind_fn)
    return cell_dm_to_fn_map


def gen_facet_fn_to_dm_map(mesh_h):
    """
    Return an array which represewnts a mapping from function index of the facet to the dm index of the facet.
    """
    facet_fn_to_dm_map = {}
    for ind_fn, ind_dm in enumerate(mesh_h._facet_ordering):
        facet_fn_to_dm_map[ind_fn] = int(ind_dm)
    return facet_fn_to_dm_map

def gen_facet_dm_to_fn_map(mesh_h):
    """
    Return an array which represewnts a mapping from dm index of the facet to the function index of the facet.
    """
    facet_dm_to_fn_map = {}
    for ind_fn, ind_dm in enumerate(mesh_h._facet_ordering):
        facet_dm_to_fn_map[ind_dm] = int(ind_fn)
    return facet_dm_to_fn_map


def gen_vertex_fn_to_dm_map(mesh_h):
    """
    Return an array which represewnts a mapping from function index of the vertex to the dm index of the vertex.
    """
    dm = mesh_h.topology_dm
    vertex_start, vertex_end = dm.getDepthStratum(0)
    vertex_fn_to_dm_map = {}
    for ind_dm in range(vertex_start, vertex_end):
        ind_fn = mesh_h._vertex_numbering.getOffset(ind_dm)
        vertex_fn_to_dm_map[ind_fn] = int(ind_dm)
    return vertex_fn_to_dm_map

def gen_vertex_dm_to_fn_map(mesh_h):
    """
    Return an array which represewnts a mapping from dm index of the vertex to the function index of the vertex.
    """
    dm = mesh_h.topology_dm
    vertex_start, vertex_end = dm.getDepthStratum(0)
    vertex_dm_to_fn_map = {}
    for ind_dm in range(vertex_start, vertex_end):
        ind_fn = mesh_h._vertex_numbering.getOffset(ind_dm)
        vertex_dm_to_fn_map[ind_dm] = int(ind_fn)
    return vertex_dm_to_fn_map



