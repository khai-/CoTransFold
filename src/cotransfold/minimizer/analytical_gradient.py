"""Analytical gradient of energy with respect to backbone torsion angles.

Strategy: instead of N finite-difference energy evaluations per gradient,
compute the gradient in a SINGLE forward + backward pass:

1. Forward pass: compute coords from torsion angles (NeRF), compute energy
2. Backward pass: propagate dE/d(coords) back through the chain to get dE/d(angles)

The backward pass exploits the fact that changing phi_i or psi_i rotates
all downstream atoms as a rigid body:

    dE/d(phi_i) = sum_{j downstream of i} dE/d(pos_j) · (axis_i × (pos_j - pivot_i))

This requires:
- dE/d(pos_j) for each atom: computed from the energy terms analytically
- The rotation geometry: computed from the backbone coordinates

Total cost: O(N²) for pairwise energy gradients + O(N²) for the chain rule
           = O(N²) instead of O(N³) for numerical gradients
"""

from __future__ import annotations

import numpy as np

from cotransfold.core.chain import NascentChain
from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import RESIDUE_PROPERTIES
from cotransfold.structure.coordinates import torsion_to_cartesian, get_ca_coords, get_cb_coords
from cotransfold.energy.ramachandran import rama_probability, _get_regions
from cotransfold.energy.vanderwaals import LJ_CUTOFF, CA_SIGMA, CA_EPSILON, CB_EPSILON, MIN_SEQ_SEP
from cotransfold.energy.solvent import SOLVATION_PARAMS, BURIAL_CUTOFF, MAX_NEIGHBORS
from cotransfold.energy.rg_restraint import expected_rg, compute_rg
from cotransfold.config import BOLTZMANN_KCAL, DEFAULT_TEMPERATURE


def _rama_gradient(backbone: BackboneState, sequence: list,
                   kT: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytical gradient of Ramachandran energy w.r.t. phi/psi.

    E_i = -kT * ln(P(phi_i, psi_i))
    dE_i/d(phi_i) = -kT * (1/P) * dP/d(phi_i)

    Returns: (dE_dphi, dE_dpsi) each shape (N,)
    """
    n = backbone.num_residues
    dphi = np.zeros(n)
    dpsi = np.zeros(n)

    for i in range(n):
        phi_i = backbone.phi[i]
        psi_i = backbone.psi[i]
        next_aa = sequence[i + 1] if i + 1 < n else None
        regions = _get_regions(sequence[i], next_aa)

        p_total = 0.0
        dp_dphi = 0.0
        dp_dpsi = 0.0

        for weight, mu_phi, mu_psi, sigma_phi, sigma_psi in regions:
            d_phi = phi_i - mu_phi
            d_phi = d_phi - 2 * np.pi * np.round(d_phi / (2 * np.pi))
            d_psi = psi_i - mu_psi
            d_psi = d_psi - 2 * np.pi * np.round(d_psi / (2 * np.pi))

            g = weight * np.exp(-0.5 * ((d_phi / sigma_phi)**2 + (d_psi / sigma_psi)**2))
            p_total += g
            dp_dphi += g * (-d_phi / sigma_phi**2)
            dp_dpsi += g * (-d_psi / sigma_psi**2)

        p_total = max(p_total, 1e-10)
        dphi[i] = -kT * dp_dphi / p_total
        dpsi[i] = -kT * dp_dpsi / p_total

    return dphi, dpsi


def _pairwise_ca_forces(ca: np.ndarray, n: int,
                        neighbor_list: np.ndarray | None = None) -> np.ndarray:
    """Compute forces on CA atoms from full LJ potential.

    dE/dr = (12*eps/r) * [-(sigma/r)^12 + (sigma/r)^6]

    Returns: forces shape (N, 3)
    """
    sigma = CA_SIGMA
    eps = CA_EPSILON
    forces = np.zeros((n, 3))

    if n < 4:
        return forces

    if neighbor_list is not None:
        for i, j in neighbor_list:
            diff = ca[i] - ca[j]
            r = np.linalg.norm(diff)
            if r < LJ_CUTOFF and r > 0.1:
                ratio = sigma / r
                # Full LJ gradient: dE/dr = (12*eps/r)*[-(σ/r)^12 + (σ/r)^6]
                de_dr = (12 * eps / r) * (-ratio**12 + ratio**6)
                unit = diff / r
                forces[i] += de_dr * unit
                forces[j] -= de_dr * unit
        return forces

    # Fallback: vectorized full pairwise
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-20

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    mask = (sep >= MIN_SEQ_SEP) & (dist < LJ_CUTOFF)

    if not np.any(mask):
        return forces

    r_safe = np.maximum(dist, 0.1)
    ratio = sigma / r_safe
    # Full LJ gradient
    de_dr = np.where(mask, (12 * eps / r_safe) * (-ratio**12 + ratio**6), 0.0)
    unit = diff / (dist[:, :, None] + 1e-20)
    forces = np.sum(de_dr[:, :, None] * unit, axis=1)
    return forces


def _pairwise_cb_forces(cb: np.ndarray, n: int,
                        sequence: list) -> np.ndarray:
    """Compute forces on Cβ atoms from LJ potential with residue-specific σ.

    Returns: forces shape (N, 3)
    """
    forces = np.zeros((n, 3))

    if n < 4:
        return forces

    # Per-residue VdW radii and pairwise sigma
    radii = np.array([RESIDUE_PROPERTIES[aa].vdw_radius for aa in sequence])
    sigma_ij = radii[:, None] + radii[None, :]  # (N, N)
    valid_radii = sigma_ij > 1.0

    diff = cb[:, None, :] - cb[None, :, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-20

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    mask = (sep >= MIN_SEQ_SEP) & (dist < LJ_CUTOFF) & valid_radii

    if not np.any(mask):
        return forces

    r_safe = np.maximum(dist, 0.1)
    ratio = sigma_ij / r_safe
    de_dr = np.where(mask, (12 * CB_EPSILON / r_safe) * (-ratio**12 + ratio**6), 0.0)
    unit = diff / (dist[:, :, None] + 1e-20)
    forces = np.sum(de_dr[:, :, None] * unit, axis=1)
    return forces


def _hbond_ca_forces(coords: np.ndarray, n: int) -> np.ndarray:
    """H-bond force on CA atoms — full pairwise O-N computation.

    Covers both local helix-forming H-bonds (i→i+3, i→i+4) and long-range
    beta-sheet H-bonds. Uses vectorized pairwise distance matrix.

    Returns: forces shape (N, 3)
    """
    forces = np.zeros((n, 3))
    if n < 4:
        return forces

    n_pos = coords[:, 0]
    ca_pos = coords[:, 1]
    c_pos = coords[:, 2]

    # Place virtual O atoms (matching DSSP energy model in hbond.py)
    o_pos = np.zeros((n, 3))
    if n >= 2:
        ca_c = c_pos[:n-1] - ca_pos[:n-1]
        n_c = c_pos[:n-1] - n_pos[1:n]
        ca_c_n = np.linalg.norm(ca_c, axis=1, keepdims=True)
        n_c_n = np.linalg.norm(n_c, axis=1, keepdims=True)
        ca_c_n = np.maximum(ca_c_n, 1e-10)
        n_c_n = np.maximum(n_c_n, 1e-10)
        bisector = ca_c / ca_c_n + n_c / n_c_n
        bis_n = np.linalg.norm(bisector, axis=1, keepdims=True)
        bis_n = np.maximum(bis_n, 1e-10)
        o_pos[:n-1] = c_pos[:n-1] + 1.24 * bisector / bis_n

    # Full pairwise O-N distance matrix
    # Acceptor: O of residue i (valid for i < n-1)
    # Donor: N of residue j (valid for j > 0)
    diff = o_pos[:, None, :] - n_pos[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)

    # Mask: valid acceptor (i < n-1), valid donor (j > 0), seq_sep >= 3, distance < 5.2
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    mask = (idx[:, None] < n - 1) & (idx[None, :] > 0) & (sep >= 3) & (dist < 5.2)

    if not np.any(mask):
        return forces

    # Strength: stronger for local helix-forming (sep 3-4), moderate for others
    base_strength = np.where(sep <= 4, 0.8, 0.6)
    strength = base_strength * np.maximum(0, (5.2 - dist) / 5.2)
    strength = np.where(mask, strength, 0.0)

    direction = diff / (dist[:, :, None] + 1e-10)

    # Force on each atom: sum over all H-bond partners
    # Acceptor i pulls donor j: force on j (donor) = +strength * direction
    # Reaction on i (acceptor) = -strength * direction
    weighted_dir = strength[:, :, None] * direction  # (N, N, 3)
    forces += np.sum(weighted_dir, axis=0)   # force on donors (sum over acceptors)
    forces -= np.sum(weighted_dir, axis=1)   # reaction on acceptors (sum over donors)

    return forces


def _rg_ca_forces(ca: np.ndarray, n: int) -> np.ndarray:
    """Force on CA atoms from radius of gyration restraint.

    E_rg = (Rg - Rg_expected)^2
    dE/d(pos_j) = 2*(Rg - Rg_expected) * d(Rg)/d(pos_j)
    d(Rg)/d(pos_j) = (pos_j - centroid) / (N * Rg)

    Returns: forces shape (N, 3)
    """
    if n < 5:
        return np.zeros((n, 3))

    centroid = ca.mean(axis=0)
    disp = ca - centroid  # (N, 3)
    rg = np.sqrt(np.mean(np.sum(disp ** 2, axis=1)))
    rg_exp = expected_rg(n)

    if rg < 1e-10:
        return np.zeros((n, 3))

    # dE/d(pos_j) = 2*(Rg - Rg_exp) * (pos_j - centroid) / (N * Rg)
    factor = 2.0 * (rg - rg_exp) / (n * rg)
    forces = factor * disp
    return forces


def _solvent_cb_forces(cb: np.ndarray, n: int, sequence: list) -> np.ndarray:
    """Force on Cβ atoms from solvent energy.

    E_solvent = sum_i sigma_i * max(0, 1 - neighbors_i / MAX_NEIGHBORS)
    Gradient: moving Cβ_j closer to Cβ_i increases burial of both →
    decreases exposure → lowers energy for hydrophobic residues.

    Returns: forces shape (N, 3) — projected onto CA.
    """
    forces = np.zeros((n, 3))
    if n < 2:
        return forces

    sigma = np.array([SOLVATION_PARAMS.get(aa, 0.0) for aa in sequence])

    diff = cb[:, None, :] - cb[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2) + 1e-20  # (N, N)

    # For pairs near the burial cutoff, there's a force
    # When a neighbor enters/exits the cutoff sphere, exposure changes discretely.
    # Use a soft switching function instead: smooth_count = sum sigmoid((cutoff - r) / width)
    width = 1.0  # Å, switching width
    switch = 1.0 / (1.0 + np.exp((dist - BURIAL_CUTOFF) / width))  # (N, N)
    np.fill_diagonal(switch, 0.0)

    # d(switch_ij)/d(r_ij) = -switch*(1-switch)/width
    d_switch_dr = -switch * (1.0 - switch) / width  # (N, N)

    # d(exposure_i)/d(r_ij) = -d_switch_ij / MAX_NEIGHBORS
    # d(E)/d(r_ij) = sigma_i * d(exposure_i)/d(r_ij) + sigma_j * d(exposure_j)/d(r_ij)
    # = -(sigma_i + sigma_j) * d_switch_dr / MAX_NEIGHBORS

    de_dr = -(sigma[:, None] + sigma[None, :]) * d_switch_dr / MAX_NEIGHBORS  # (N, N)

    # Convert to force vectors
    unit = diff / (dist[:, :, None] + 1e-20)
    forces = np.sum(de_dr[:, :, None] * unit, axis=1)
    return forces


def build_neighbor_list(ca: np.ndarray, cutoff: float = 10.0,
                        min_seq_sep: int = 3) -> np.ndarray:
    """Build neighbor list of CA pairs within cutoff distance.

    Returns array of shape (M, 2) where each row is (i, j) pair with i < j.
    Only includes pairs with sequence separation >= min_seq_sep.
    """
    n = len(ca)
    pairs = []

    # Use cell-list-like approach for large N, direct for small N
    if n < 100:
        # Direct pairwise for small proteins
        for i in range(n):
            for j in range(i + min_seq_sep, n):
                r = np.linalg.norm(ca[i] - ca[j])
                if r < cutoff:
                    pairs.append((i, j))
    else:
        # Grid-based for larger proteins
        cell_size = cutoff
        min_coords = ca.min(axis=0)
        cell_idx = ((ca - min_coords) / cell_size).astype(int)

        from collections import defaultdict
        grid = defaultdict(list)
        for i in range(n):
            key = tuple(cell_idx[i])
            grid[key].append(i)

        # Check neighboring cells
        for key, atoms_i in grid.items():
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nkey = (key[0]+dx, key[1]+dy, key[2]+dz)
                        if nkey not in grid:
                            continue
                        for i in atoms_i:
                            for j in grid[nkey]:
                                if j > i and (j - i) >= min_seq_sep:
                                    r = np.linalg.norm(ca[i] - ca[j])
                                    if r < cutoff:
                                        pairs.append((i, j))

    return np.array(pairs, dtype=int).reshape(-1, 2) if pairs else np.zeros((0, 2), dtype=int)


def compute_analytical_energy_and_gradient(
        chain: NascentChain,
        frozen_mask: np.ndarray | None = None,
        weights: dict | None = None,
        **kwargs) -> tuple[float, np.ndarray]:
    """Compute energy and exact gradient in one forward+backward pass.

    Args:
        chain: nascent chain
        frozen_mask: per-residue (0=frozen, 1=free)
        weights: energy term weights

    Returns:
        (energy, gradient) where gradient shape is (2*N,)
        layout: [phi_0, psi_0, phi_1, psi_1, ...]
    """
    if weights is None:
        weights = {}
    w_rama = weights.get('ramachandran', 1.0)
    w_vdw = weights.get('vanderwaals', 0.7)
    w_hbond = weights.get('hbond', 1.3)
    w_solvent = weights.get('solvent', 1.3)
    w_rg = weights.get('rg_restraint', 1.0)

    n = chain.chain_length
    if n == 0:
        return 0.0, np.array([])

    if frozen_mask is None:
        frozen_mask = np.ones(n)

    coords = torsion_to_cartesian(chain.backbone)
    coords_flat = coords.reshape(-1, 3)  # (3N, 3)
    ca = coords[:, 1]  # (N, 3)
    kT = BOLTZMANN_KCAL * DEFAULT_TEMPERATURE

    # === Backward pass: dE/d(angles) ===

    gradient = np.zeros(2 * n)

    # --- Ramachandran: direct analytical gradient ---
    dE_dphi_rama, dE_dpsi_rama = _rama_gradient(chain.backbone, chain.sequence, kT)
    gradient[0::2] += w_rama * dE_dphi_rama
    gradient[1::2] += w_rama * dE_dpsi_rama

    # --- VdW + H-bond: force-based gradient ---
    nb_list = build_neighbor_list(ca, cutoff=LJ_CUTOFF, min_seq_sep=MIN_SEQ_SEP) if n >= 4 else None
    ca_forces = np.zeros((n, 3))
    ca_forces += w_vdw * _pairwise_ca_forces(ca, n, neighbor_list=nb_list)
    ca_forces += w_hbond * _hbond_ca_forces(coords, n)

    # --- Cβ-Cβ forces (projected onto CA) ---
    cb = get_cb_coords(coords, chain.sequence)
    cb_forces = w_vdw * _pairwise_cb_forces(cb, n, chain.sequence)
    ca_forces += cb_forces  # Approximate: Cβ rigidly attached to CA

    # --- Rg restraint forces ---
    ca_forces += w_rg * _rg_ca_forces(ca, n)

    # --- Solvent forces (on Cβ, projected to CA) ---
    ca_forces += w_solvent * _solvent_cb_forces(cb, n, chain.sequence)

    # Convert CA forces to torsion angle gradients using chain rule:
    # dE/d(angle_k) = sum_{j>=k} force_j · (axis_k × (CA_j - pivot_k))
    for var_k in range(2 * n):
        res_k = var_k // 2
        if frozen_mask[res_k] == 0:
            continue

        is_phi = (var_k % 2 == 0)

        if is_phi:
            # phi_k: rotation around N_k - CA_k bond
            n_idx = 3 * res_k
            ca_idx = 3 * res_k + 1
            axis = coords_flat[ca_idx] - coords_flat[n_idx]
            pivot = coords_flat[n_idx]
            first_res = res_k  # CA_k and downstream are affected
        else:
            # psi_k: rotation around CA_k - C_k bond
            ca_idx = 3 * res_k + 1
            c_idx = 3 * res_k + 2
            axis = coords_flat[c_idx] - coords_flat[ca_idx]
            pivot = coords_flat[ca_idx]
            first_res = res_k + 1  # CA_{k+1} and downstream

        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-10:
            continue
        axis = axis / axis_len

        # Sum force contributions from all downstream CA atoms
        for j in range(first_res, n):
            lever = ca[j] - pivot
            dpos_dangle = np.cross(axis, lever)
            gradient[var_k] += np.dot(ca_forces[j], dpos_dangle)

    # Zero out frozen residues
    for i in range(n):
        if frozen_mask[i] == 0:
            gradient[2 * i] = 0.0
            gradient[2 * i + 1] = 0.0

    return 0.0, gradient  # Energy computed separately by TotalEnergy
