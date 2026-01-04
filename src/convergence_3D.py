"""3D convergence experiments for the mixed LOD formulation."""

from __future__ import annotations

import argparse

import numpy as np
from firedrake import UnitCubeMesh, cos

from lod_new import mixed_LOD


def a_factory(a_data: np.ndarray):
    """Coefficient lookup compatible with Firedrake function signatures."""

    def a(x, y, z):
        col = np.floor(x * (np.size(a_data, 1))).astype(int)
        row = np.floor((1 - y) * (np.size(a_data, 0))).astype(int)
        dep = np.floor(z * (np.size(a_data, 2))).astype(int)
        # Clamp indices to avoid out-of-bounds
        col = np.clip(col, 0, np.size(a_data, 1) - 1)
        row = np.clip(row, 0, np.size(a_data, 0) - 1)
        dep = np.clip(dep, 0, np.size(a_data, 2) - 1)
        return a_data[row, col, dep]

    return a


def build_coefficient(coef: str, n_fine_level: int, seed: int) -> np.ndarray:
    """Create the coefficient array used by the PDE operator."""
    if coef == "1":
        return np.array([[[1.0]]])
    if coef == "noise":
        rng = np.random.default_rng(seed)
        alpha = 0.01
        beta = 1.0
        n = 2 ** (n_fine_level - 1)
        return alpha + (beta - alpha) * rng.integers(0, 2, size=(n, n, n))
    raise ValueError(f"Unknown coef '{coef}'. Use '1' or 'noise'.")


def f(x, y, z):
    return 3 * np.pi**2 * cos(np.pi * x) * cos(np.pi * y) * cos(np.pi * z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coef", default="noise", choices=["1", "noise"], help="Coefficient model.")
    parser.add_argument("--n-fine-level", type=int, default=4, help="Mesh refinement level.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for noise coefficients.")
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="*",
        default=[1, 2, 3, 3],
        help="Localization parameter k per coarse level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a_data = build_coefficient(args.coef, args.n_fine_level, args.seed)
    a = a_factory(a_data)

    if len(args.k_list) < args.n_fine_level - 1:
        raise ValueError("k-list must provide a value for each coarse level.")

    for i, n in enumerate(range(1, args.n_fine_level)):
        L = args.n_fine_level - n
        N = 2**n
        k = args.k_list[i]
        print("-----------------------------------------")
        print(f"Testing mixed LOD with 1/{2**n} coarse mesh, L={L}, k={k}")
        base_mesh = UnitCubeMesh(N, N, N)
        test_mixed_LOD = mixed_LOD(
            base_mesh,
            a,
            f,
            L,
            k,
            int_method="stable",
            printLevel=0,
            parallel_computation=True,
        )
        flux_lod, pres_lod = test_mixed_LOD.lod()
        flux_ref, pres_ref = test_mixed_LOD.ref(True)
        M_a = test_mixed_LOD.M_a_csr
        ref_ENorm = np.sqrt(flux_ref.dat.data.T @ M_a @ flux_ref.dat.data)

        Mq_csr = test_mixed_LOD.Mq_csr
        ref_Pres_Norm = np.sqrt(pres_ref.dat.data.T @ Mq_csr @ pres_ref.dat.data)

        print(f"Reference solution energy norm: {ref_ENorm}, pressure L2 norm: {ref_Pres_Norm}")
        err_arr = flux_lod.dat.data - flux_ref.dat.data
        rel_err_ENorm = np.sqrt(err_arr.T @ M_a @ err_arr) / ref_ENorm
        err_pres_arr = pres_lod.dat.data - pres_ref.dat.data
        rel_err_Pres_L2Norm = np.sqrt(err_pres_arr.T @ Mq_csr @ err_pres_arr) / ref_Pres_Norm

        print(
            f"1/{2**n}, error energy norm: {rel_err_ENorm}, ref energy norm: {ref_ENorm}, "
            f"pressure L2 norm error: {rel_err_Pres_L2Norm}, ref pressure L2 norm: {ref_Pres_Norm}"
        )


if __name__ == "__main__":
    main()
