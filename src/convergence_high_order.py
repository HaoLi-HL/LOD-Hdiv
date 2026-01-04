"""2D convergence experiments for the mixed LOD formulation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from firedrake import UnitSquareMesh, cos

from lod_new import mixed_LOD


def build_coefficient(coef: str, n_fine_level: int, seed: int) -> np.ndarray:
    """Create the coefficient array used by the PDE operator."""
    if coef == "1":
        return np.array([[1.0]])
    if coef == "noise":
        rng = np.random.default_rng(seed)
        alpha = 0.01
        beta = 1.0
        n = 2 ** (n_fine_level - 1)
        return alpha + (beta - alpha) * rng.integers(0, 2, size=(n, n))
    if coef == "channels":
        raise ValueError("coef='channels' requires a custom input array.")
    raise ValueError(f"Unknown coef '{coef}'. Use '1' or 'noise'.")


def a_factory(a_data: np.ndarray):
    """Coefficient lookup compatible with Firedrake function signatures."""

    def a(x, y):
        col = np.floor(x * (np.size(a_data, 1))).astype(int)
        row = np.floor((1 - y) * (np.size(a_data, 0))).astype(int)
        return a_data[row, col]

    return a


def f(x, y):
    return 2 * np.pi**2 * cos(np.pi * x) * cos(np.pi * y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coef", default="noise", choices=["1", "noise"], help="Coefficient model.")
    parser.add_argument("--n-fine-level", type=int, default=6, help="Mesh refinement level.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for RT spaces.")
    parser.add_argument("--k-min", type=int, default=1, help="Minimum localization parameter k.")
    parser.add_argument("--k-max", type=int, default=7, help="Maximum localization parameter k.")
    parser.add_argument("--n-min", type=int, default=1, help="Minimum coarse mesh exponent n.")
    parser.add_argument("--n-max", type=int, default=None, help="Maximum coarse mesh exponent n.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for noise coefficients.")
    parser.add_argument("--output-dir", default="num_results", help="Directory for NumPy outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_max = args.n_max if args.n_max is not None else args.n_fine_level - 1
    n_min = args.n_min
    k_min = args.k_min
    k_max = args.k_max

    if n_max < n_min:
        raise ValueError("n-max must be >= n-min.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_data = build_coefficient(args.coef, args.n_fine_level, args.seed)
    a = a_factory(a_data)

    u_eng_err_arr = np.zeros((n_max - n_min + 1, k_max - k_min + 1))
    pre_l2_err_arr = np.zeros((n_max - n_min + 1, k_max - k_min + 1))

    for n in range(n_min, n_max + 1):
        L = args.n_fine_level - n
        N = 2**n
        for k in range(k_min, k_max + 1):
            print("-----------------------------------------")
            start_time = time.time()
            base_mesh = UnitSquareMesh(N, N)
            test_mixed_LOD = mixed_LOD(
                base_mesh,
                a,
                f,
                L,
                k,
                degree=args.degree,
                int_method="stable",
                printLevel=0,
                parallel_computation=True,
            )
            flux_ref, pres_ref = test_mixed_LOD.ref(True)
            M_a = test_mixed_LOD.M_a_csr
            ref_ENorm = np.sqrt(flux_ref.dat.data.T @ M_a @ flux_ref.dat.data)
            Mq_csr = test_mixed_LOD.Mq_csr
            ref_Pres_Norm = np.sqrt(pres_ref.dat.data.T @ Mq_csr @ pres_ref.dat.data)
            print(
                f"Testing mixed LOD with 1/{N} coarse mesh, k={k}, degree={args.degree}. "
                f"Reference solution energy norm: {ref_ENorm:8f}, pressure L2 norm: {ref_Pres_Norm:8f}"
            )

            flux_lod, pres_lod = test_mixed_LOD.lod()
            solve_time = time.time() - start_time
            err_arr = flux_lod.dat.data - flux_ref.dat.data
            rel_err_ENorm = np.sqrt(err_arr.T @ M_a @ err_arr) / ref_ENorm
            err_pres_arr = pres_lod.dat.data - pres_ref.dat.data
            rel_err_Pres_L2Norm = np.sqrt(err_pres_arr.T @ Mq_csr @ err_pres_arr) / ref_Pres_Norm

            u_eng_err_arr[n - n_min, k - k_min] = rel_err_ENorm
            pre_l2_err_arr[n - n_min, k - k_min] = rel_err_Pres_L2Norm

            print(
                f"n={n}, k={k}, degree={args.degree}, error energy norm: {rel_err_ENorm:8f}, "
                f"pressure L2 norm error: {rel_err_Pres_L2Norm:8f}, solveT={solve_time:4f}"
            )

        np.save(output_dir / f"u_eng_err_arr_deg_{args.degree}_n_{n}.npy", u_eng_err_arr)
        np.save(output_dir / f"pre_l2_err_arr_deg_{args.degree}_n_{n}.npy", pre_l2_err_arr)


if __name__ == "__main__":
    main()
