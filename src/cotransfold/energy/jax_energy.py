"""JAX-based energy function with automatic differentiation.

All energy terms reimplemented as pure functions using jax.numpy.
This enables:
1. Exact analytical gradients via jax.grad (no finite differences)
2. JIT compilation for 2-5x additional speedup
3. GPU acceleration (future)

The energy function is a single pure function:
    energy = total_energy(phi, psi, omega, sequence_props, tunnel_positions, tunnel_params)

All parameters are passed as arrays (no Python objects) for JIT compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cotransfold.config import (
    BOND_LENGTH_N_CA, BOND_LENGTH_CA_C, BOND_LENGTH_C_N,
    BOND_ANGLE_N_CA_C, BOND_ANGLE_CA_C_N, BOND_ANGLE_C_N_CA,
    BOLTZMANN_KCAL, DEFAULT_TEMPERATURE,
)

# Enable 64-bit precision for accuracy matching numpy
jax.config.update("jax_enable_x64", True)


# ============================================================================
# NeRF: Torsion → Cartesian (JAX version)
# ============================================================================

def _place_atom_jax(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray,
                    bond_length: float, bond_angle: float,
                    torsion: jnp.ndarray) -> jnp.ndarray:
    """Place atom D given A, B, C using NeRF (JAX-compatible)."""
    bc = c - b
    bc_norm = bc / jnp.linalg.norm(bc)
    ab = b - a
    n = jnp.cross(ab, bc_norm)
    n_norm = n / jnp.linalg.norm(n)
    m = jnp.cross(n_norm, bc_norm)

    d_x = -bond_length * jnp.cos(bond_angle)
    d_y = bond_length * jnp.sin(bond_angle) * jnp.cos(torsion)
    d_z = bond_length * jnp.sin(bond_angle) * jnp.sin(torsion)

    return c + d_x * bc_norm + d_y * m + d_z * n_norm


def torsion_to_cartesian_jax(phi: jnp.ndarray, psi: jnp.ndarray,
                              omega: jnp.ndarray) -> jnp.ndarray:
    """Convert torsion angles to Cartesian coords (JAX, uses scan for efficiency).

    Returns shape (N, 3, 3) for [N, CA, C] per residue.
    """
    n_res = len(phi)

    # Place first 3 atoms canonically
    atom0 = jnp.array([0.0, 0.0, 0.0])
    atom1 = jnp.array([BOND_LENGTH_N_CA, 0.0, 0.0])
    angle_n_ca_c = BOND_ANGLE_N_CA_C
    atom2 = jnp.array([
        BOND_LENGTH_N_CA + BOND_LENGTH_CA_C * jnp.cos(jnp.pi - angle_n_ca_c),
        BOND_LENGTH_CA_C * jnp.sin(jnp.pi - angle_n_ca_c),
        0.0,
    ])

    all_atoms = [atom0, atom1, atom2]

    # Build remaining atoms sequentially
    for atom_idx in range(3, n_res * 3):
        res_idx = atom_idx // 3
        atom_type = atom_idx % 3

        a = all_atoms[atom_idx - 3]
        b = all_atoms[atom_idx - 2]
        c = all_atoms[atom_idx - 1]

        if atom_type == 0:  # N
            bl, ba, tor = BOND_LENGTH_C_N, BOND_ANGLE_CA_C_N, psi[res_idx - 1]
        elif atom_type == 1:  # CA
            bl, ba, tor = BOND_LENGTH_N_CA, BOND_ANGLE_C_N_CA, omega[res_idx]
        else:  # C
            bl, ba, tor = BOND_LENGTH_CA_C, BOND_ANGLE_N_CA_C, phi[res_idx]

        new_atom = _place_atom_jax(a, b, c, bl, ba, tor)
        all_atoms.append(new_atom)

    coords = jnp.stack(all_atoms).reshape(n_res, 3, 3)
    return coords


# ============================================================================
# Energy Terms (pure JAX functions)
# ============================================================================

def _ramachandran_energy(phi: jnp.ndarray, psi: jnp.ndarray,
                         rama_classes: jnp.ndarray,
                         kT: float) -> jnp.ndarray:
    """Ramachandran statistical potential.

    rama_classes: integer array (0=general, 1=glycine, 2=proline, 3=preproline)
    """
    # Define region parameters as arrays for each class
    # General: alpha-helix and beta-sheet regions
    # Simplified to 2 main Gaussians per class for JIT efficiency

    # Alpha-helix center
    alpha_phi, alpha_psi = jnp.radians(-63.0), jnp.radians(-43.0)
    alpha_sigma = jnp.radians(15.0)
    # Beta center
    beta_phi, beta_psi = jnp.radians(-120.0), jnp.radians(130.0)
    beta_sigma = jnp.radians(20.0)
    # PPII center
    ppii_phi, ppii_psi = jnp.radians(-65.0), jnp.radians(145.0)
    ppii_sigma = jnp.radians(15.0)

    def _periodic_dist(a, b):
        d = a - b
        return d - 2 * jnp.pi * jnp.round(d / (2 * jnp.pi))

    dphi_a = _periodic_dist(phi, alpha_phi)
    dpsi_a = _periodic_dist(psi, alpha_psi)
    p_alpha = 0.40 * jnp.exp(-0.5 * (dphi_a**2 + dpsi_a**2) / alpha_sigma**2)

    dphi_b = _periodic_dist(phi, beta_phi)
    dpsi_b = _periodic_dist(psi, beta_psi)
    p_beta = 0.25 * jnp.exp(-0.5 * (dphi_b**2 + dpsi_b**2) / beta_sigma**2)

    dphi_p = _periodic_dist(phi, ppii_phi)
    dpsi_p = _periodic_dist(psi, ppii_psi)
    p_ppii = 0.15 * jnp.exp(-0.5 * (dphi_p**2 + dpsi_p**2) / ppii_sigma**2)

    # Background
    p_bg = 0.01

    p_total = p_alpha + p_beta + p_ppii + p_bg

    # Glycine: add left-handed helix
    dphi_la = _periodic_dist(phi, jnp.radians(63.0))
    dpsi_la = _periodic_dist(psi, jnp.radians(43.0))
    p_left = 0.20 * jnp.exp(-0.5 * (dphi_la**2 + dpsi_la**2) / jnp.radians(20.0)**2)
    is_gly = (rama_classes == 1).astype(jnp.float64)
    p_total = p_total + is_gly * p_left

    p_total = jnp.maximum(p_total, 1e-10)
    return -kT * jnp.sum(jnp.log(p_total))


def _hbond_energy(coords: jnp.ndarray) -> jnp.ndarray:
    """Hydrogen bond energy using DSSP model (vectorized JAX)."""
    n = coords.shape[0]
    if n < 4:
        return 0.0

    n_atoms = coords[:, 0]   # (N, 3)
    ca_atoms = coords[:, 1]
    c_atoms = coords[:, 2]

    # Place H atoms: opposite CA direction from N
    ca_n = n_atoms - ca_atoms
    ca_n_norms = jnp.linalg.norm(ca_n, axis=1, keepdims=True)
    ca_n_norms = jnp.maximum(ca_n_norms, 1e-10)
    h_atoms = n_atoms + 1.0 * ca_n / ca_n_norms  # (N, 3)

    # Place O atoms: bisector of CA-C and N(next)-C
    ca_c = c_atoms[:-1] - ca_atoms[:-1]
    n_c = c_atoms[:-1] - n_atoms[1:]
    ca_c_n = jnp.linalg.norm(ca_c, axis=1, keepdims=True)
    n_c_n = jnp.linalg.norm(n_c, axis=1, keepdims=True)
    ca_c_n = jnp.maximum(ca_c_n, 1e-10)
    n_c_n = jnp.maximum(n_c_n, 1e-10)
    bisector = ca_c / ca_c_n + n_c / n_c_n
    bis_n = jnp.linalg.norm(bisector, axis=1, keepdims=True)
    bis_n = jnp.maximum(bis_n, 1e-10)
    o_pos = c_atoms[:-1] + 1.24 * bisector / bis_n  # (N-1, 3)

    # Pad o_atoms to length N
    o_atoms = jnp.zeros_like(n_atoms)
    o_atoms = o_atoms.at[:-1].set(o_pos)

    # Pairwise O-N distances for all acceptor(i)-donor(j) pairs
    r_on = jnp.linalg.norm(o_atoms[:, None, :] - n_atoms[None, :, :], axis=2)  # (N, N)

    # Sequence separation mask
    idx = jnp.arange(n)
    sep = idx[None, :] - idx[:, None]  # j - i
    mask = (sep >= 3) & (idx[:, None] < n - 1) & (idx[None, :] > 0)
    mask = mask & (r_on < 5.2)

    # Compute DSSP distances for all valid pairs
    r_ch = jnp.linalg.norm(c_atoms[:, None, :] - h_atoms[None, :, :], axis=2)
    r_oh = jnp.linalg.norm(o_atoms[:, None, :] - h_atoms[None, :, :], axis=2)
    r_cn = jnp.linalg.norm(c_atoms[:, None, :] - n_atoms[None, :, :], axis=2)

    # Avoid div by zero
    r_on_safe = jnp.maximum(r_on, 0.01)
    r_ch_safe = jnp.maximum(r_ch, 0.01)
    r_oh_safe = jnp.maximum(r_oh, 0.01)
    r_cn_safe = jnp.maximum(r_cn, 0.01)

    e_hbond = 0.42 * 0.20 * 332.0 * (
        1.0 / r_on_safe + 1.0 / r_ch_safe - 1.0 / r_oh_safe - 1.0 / r_cn_safe
    )

    # Only attractive, only valid pairs
    e_hbond = jnp.where(mask, jnp.minimum(e_hbond, 0.0), 0.0)
    return jnp.sum(e_hbond)


def _vdw_energy(coords: jnp.ndarray) -> jnp.ndarray:
    """Van der Waals repulsive energy (CA-CA, vectorized JAX)."""
    n = coords.shape[0]
    if n < 4:
        return 0.0

    ca = coords[:, 1]  # (N, 3)
    diff = ca[:, None, :] - ca[None, :, :]  # (N, N, 3)
    dist = jnp.linalg.norm(diff, axis=2)    # (N, N)

    idx = jnp.arange(n)
    sep = jnp.abs(idx[:, None] - idx[None, :])
    mask = (sep >= 3) & (idx[:, None] < idx[None, :])  # Upper triangle

    sigma = 3.8
    r_safe = jnp.maximum(dist, 0.1)
    ratios = sigma / r_safe
    e_vdw = 0.05 * ratios ** 12

    # Only where r < sigma and valid pair
    e_vdw = jnp.where(mask & (dist < sigma), e_vdw, 0.0)
    return jnp.sum(e_vdw)


def _bonded_energy(omega: jnp.ndarray) -> jnp.ndarray:
    """Omega angle planarity restraint."""
    return jnp.sum(10.0 * (1.0 - jnp.cos(omega - jnp.pi)))


def _solvent_energy(coords: jnp.ndarray, solvation_params: jnp.ndarray,
                    tunnel_positions: jnp.ndarray,
                    tunnel_length: float) -> jnp.ndarray:
    """Implicit solvation energy (JAX)."""
    n = coords.shape[0]
    if n < 2:
        return 0.0

    ca = coords[:, 1]
    exposed = tunnel_positions > tunnel_length

    diff = ca[:, None, :] - ca[None, :, :]
    dist = jnp.linalg.norm(diff, axis=2)
    dist = dist + jnp.eye(n) * 100.0  # Exclude self

    neighbor_counts = jnp.sum(dist < 8.0, axis=1).astype(jnp.float64)
    exposure = jnp.maximum(0.0, 1.0 - neighbor_counts / 10.0)

    return jnp.sum(jnp.where(exposed, solvation_params * exposure, 0.0))


def _tunnel_energy(coords: jnp.ndarray,
                   charges: jnp.ndarray,
                   tunnel_positions: jnp.ndarray,
                   tunnel_radii: jnp.ndarray,
                   tunnel_potentials: jnp.ndarray,
                   tunnel_length: float,
                   wall_spring: float = 5.0,
                   wall_buffer: float = 1.5) -> jnp.ndarray:
    """Tunnel constraint energy (JAX)."""
    n = coords.shape[0]
    if n < 3:
        return 0.0

    ca = coords[:, 1]
    in_tunnel = (tunnel_positions >= 0) & (tunnel_positions <= tunnel_length)

    # Radial offsets from backbone curvature
    offsets = jnp.zeros(n)
    if n >= 3:
        v1 = ca[1:-1] - ca[:-2]
        v2 = ca[2:] - ca[1:-1]
        n1 = jnp.linalg.norm(v1, axis=1, keepdims=True)
        n2 = jnp.linalg.norm(v2, axis=1, keepdims=True)
        n1 = jnp.maximum(n1, 1e-10)
        n2 = jnp.maximum(n2, 1e-10)
        cos_a = jnp.sum((v1 / n1) * (v2 / n2), axis=1)
        cos_a = jnp.clip(cos_a, -1.0, 1.0)
        inner_offsets = jnp.maximum(2.3 * (1.0 - cos_a) / 2.0 + 1.0, 0.0)
        offsets = offsets.at[1:-1].set(inner_offsets)

    # Wall repulsion
    wall_dist = tunnel_radii - offsets - wall_buffer
    wall_energy = jnp.where(in_tunnel & (wall_dist < 0), wall_spring * wall_dist ** 2, 0.0)

    # Electrostatic
    elec_energy = jnp.where(in_tunnel & (charges != 0.0), charges * tunnel_potentials, 0.0)

    return jnp.sum(wall_energy) + jnp.sum(elec_energy)


# ============================================================================
# Combined energy function
# ============================================================================

def total_energy_jax(phi: jnp.ndarray, psi: jnp.ndarray, omega: jnp.ndarray,
                     rama_classes: jnp.ndarray,
                     charges: jnp.ndarray,
                     solvation_params: jnp.ndarray,
                     tunnel_positions: jnp.ndarray,
                     tunnel_radii: jnp.ndarray,
                     tunnel_potentials: jnp.ndarray,
                     tunnel_length: float,
                     weights: dict | None = None) -> jnp.ndarray:
    """Compute total energy from torsion angles (JAX, differentiable).

    Args:
        phi, psi, omega: torsion angles, shape (N,)
        rama_classes: integer array for Ramachandran class per residue
        charges: residue charges, shape (N,)
        solvation_params: solvation cost per residue, shape (N,)
        tunnel_positions: distance from PTC per residue, shape (N,)
        tunnel_radii: tunnel radius at each residue position, shape (N,)
        tunnel_potentials: electrostatic potential at each position, shape (N,)
        tunnel_length: total tunnel length (scalar)
        weights: energy term weights dict

    Returns:
        Total energy (scalar)
    """
    if weights is None:
        weights = {}
    w_rama = weights.get('ramachandran', 1.0)
    w_hbond = weights.get('hbond', 1.0)
    w_vdw = weights.get('vanderwaals', 0.5)
    w_bonded = weights.get('bonded', 0.5)
    w_solvent = weights.get('solvent', 0.5)
    w_tunnel = weights.get('tunnel', 1.0)

    kT = BOLTZMANN_KCAL * DEFAULT_TEMPERATURE

    # Build coordinates from torsion angles
    coords = torsion_to_cartesian_jax(phi, psi, omega)

    e = 0.0
    e = e + w_rama * _ramachandran_energy(phi, psi, rama_classes, kT)
    e = e + w_hbond * _hbond_energy(coords)
    e = e + w_vdw * _vdw_energy(coords)
    e = e + w_bonded * _bonded_energy(omega)
    e = e + w_solvent * _solvent_energy(coords, solvation_params,
                                         tunnel_positions, tunnel_length)
    e = e + w_tunnel * _tunnel_energy(coords, charges, tunnel_positions,
                                       tunnel_radii, tunnel_potentials,
                                       tunnel_length)
    return e


# Gradient function: exact derivatives w.r.t. phi and psi
_grad_phi_psi = jax.grad(total_energy_jax, argnums=(0, 1))


def compute_energy_and_grad(phi, psi, omega, rama_classes, charges,
                            solvation_params, tunnel_positions,
                            tunnel_radii, tunnel_potentials,
                            tunnel_length, weights=None):
    """Compute energy and exact gradients w.r.t. phi and psi.

    Returns:
        (energy, grad_phi, grad_psi)
    """
    energy = total_energy_jax(phi, psi, omega, rama_classes, charges,
                              solvation_params, tunnel_positions,
                              tunnel_radii, tunnel_potentials,
                              tunnel_length, weights)
    grad_phi, grad_psi = _grad_phi_psi(phi, psi, omega, rama_classes, charges,
                                        solvation_params, tunnel_positions,
                                        tunnel_radii, tunnel_potentials,
                                        tunnel_length, weights)
    return float(energy), np.array(grad_phi), np.array(grad_psi)
