# LOD-Hdiv

Code accompanying the paper "Localized Orthogonal Decomposition for H(div) Problems" (arXiv:2504.18322).
This repository provides the Firedrake-based implementation for the mixed LOD method,
including 2D/3D mesh utilities and convergence studies.

## What is in this repository

- `src/lod_new.py`: core mixed LOD implementation (`mixed_LOD`).
- `src/mesh2D_manager.py`, `src/mesh3D_manager.py`: mesh hierarchy and connectivity helpers.
- `src/convergence_high_order.py`: 2D convergence experiment runner.
- `src/convergence_3D.py`: 3D convergence experiment runner.

## Requirements

- Python 3.9+ (tested with 3.10)
- [Firedrake](https://www.firedrakeproject.org/) with PETSc and petsc4py
- NumPy, SciPy, Matplotlib

Because Firedrake is installed via its own installer/virtualenv, we recommend following
Firedrake's installation guide first, and then installing the remaining Python dependencies
in the same environment.

## Quick start

Clone the repository and run the 2D convergence script:

```bash
cd /path/to/LOD-Hdiv
python src/convergence_high_order.py --coef noise --n-fine-level 6 --degree 2
```

Run the 3D experiment (more expensive):

```bash
python src/convergence_3D.py --coef noise --n-fine-level 4
```

Results are written to `num_results/` (created automatically).

## Reproducibility notes

- The scripts accept `--seed` for random coefficients.
- 3D runs are significantly more expensive. Start with smaller `--n-fine-level`
  or reduce the list of `k` values if you run into memory/time constraints.

## Citing

If you use this code in academic work, please cite the paper:

```
@article{lod-hdiv-2025,
  title   = {Localized Orthogonal Decomposition for H(div) Problems},
  author  = {Authors},
  journal = {arXiv:2504.18322},
  year    = {2025}
}
```
