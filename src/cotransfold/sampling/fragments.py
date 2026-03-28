"""Secondary-structure-based fragment library for Monte Carlo moves.

Generates local structure fragments (3-mer and 9-mer phi/psi windows) based
on predicted secondary structure. Each fragment is a set of torsion angles
drawn from the known distributions for helix, strand, and coil regions.

This replaces random Gaussian perturbations with physically realistic
local geometry, dramatically improving the quality of MC sampling.
"""

from __future__ import annotations

import numpy as np
from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES


# Torsion angle distributions per SS type (mean, std in radians)
# Derived from high-resolution PDB structures
SS_TORSIONS = {
    'H': {  # Alpha helix
        'phi_mean': np.radians(-63.0),
        'phi_std': np.radians(8.0),
        'psi_mean': np.radians(-42.0),
        'psi_std': np.radians(8.0),
    },
    'E': {  # Beta strand
        'phi_mean': np.radians(-120.0),
        'phi_std': np.radians(15.0),
        'psi_mean': np.radians(130.0),
        'psi_std': np.radians(15.0),
    },
    'C': {  # Coil/loop — wider distribution
        'phi_mean': np.radians(-80.0),
        'phi_std': np.radians(40.0),
        'psi_mean': np.radians(50.0),
        'psi_std': np.radians(60.0),
    },
}

# Additional coil conformations (polyproline II, left-handed helix, etc.)
COIL_MODES = [
    # Polyproline II
    {'phi': np.radians(-75.0), 'psi': np.radians(145.0), 'weight': 0.3},
    # Type I turn (i+1 position)
    {'phi': np.radians(-60.0), 'psi': np.radians(-30.0), 'weight': 0.2},
    # Type I turn (i+2 position)
    {'phi': np.radians(-90.0), 'psi': np.radians(0.0), 'weight': 0.2},
    # Extended
    {'phi': np.radians(-130.0), 'psi': np.radians(150.0), 'weight': 0.3},
]


def predict_ss(sequence: list[AminoAcid]) -> str:
    """Simple secondary structure prediction from sequence.

    Uses Chou-Fasman propensities to assign H/E/C per residue.
    This is a basic predictor — real applications would use PSIPRED.

    Returns: string of H/E/C per residue.
    """
    n = len(sequence)
    ss = ['C'] * n

    # Sliding window: if 4+ consecutive residues have high helix propensity, assign H
    for i in range(n - 3):
        window = sequence[i:i+4]
        avg_helix = np.mean([RESIDUE_PROPERTIES[aa].helix_propensity for aa in window])
        avg_sheet = np.mean([RESIDUE_PROPERTIES[aa].sheet_propensity for aa in window])

        if avg_helix > 1.05:
            for j in range(i, min(i + 4, n)):
                if ss[j] != 'H':  # Don't override existing H
                    ss[j] = 'H'
        elif avg_sheet > 1.05 and avg_helix < 0.95:
            for j in range(i, min(i + 4, n)):
                if ss[j] == 'C':  # Only assign E to coil
                    ss[j] = 'E'

    # Extend helices: if flanked by H, convert C to H
    for _ in range(2):  # Two passes
        for i in range(1, n - 1):
            if ss[i] == 'C' and ss[i-1] == 'H' and ss[i+1] == 'H':
                ss[i] = 'H'

    return ''.join(ss)


def generate_fragment(ss_string: str, start: int, length: int,
                      rng: np.random.RandomState | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single fragment of phi/psi angles based on SS prediction.

    Args:
        ss_string: predicted SS string (H/E/C)
        start: starting residue index
        length: fragment length (3 or 9)
        rng: random state for reproducibility

    Returns:
        (phi_fragment, psi_fragment) each shape (length,)
    """
    if rng is None:
        rng = np.random.RandomState()

    phi = np.zeros(length)
    psi = np.zeros(length)

    for i in range(length):
        pos = start + i
        if pos >= len(ss_string):
            ss = 'C'
        else:
            ss = ss_string[pos]

        if ss in ('H', 'E'):
            t = SS_TORSIONS[ss]
            phi[i] = rng.normal(t['phi_mean'], t['phi_std'])
            psi[i] = rng.normal(t['psi_mean'], t['psi_std'])
        else:
            # Coil: sample from mixture of modes
            mode = rng.choice(len(COIL_MODES),
                              p=[m['weight'] for m in COIL_MODES])
            m = COIL_MODES[mode]
            phi[i] = rng.normal(m['phi'], np.radians(15.0))
            psi[i] = rng.normal(m['psi'], np.radians(15.0))

    # Wrap to [-pi, pi]
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    psi = np.arctan2(np.sin(psi), np.cos(psi))

    return phi, psi


class FragmentLibrary:
    """Library of SS-based phi/psi fragments for MC moves.

    Precomputes N_FRAGS candidate fragments per position for both
    3-mer and 9-mer lengths.
    """

    def __init__(self, sequence: list[AminoAcid], n_frags: int = 200,
                 seed: int = 42):
        self.sequence = sequence
        self.n = len(sequence)
        self.ss = predict_ss(sequence)
        self.n_frags = n_frags
        self._rng = np.random.RandomState(seed)

        # Precompute fragment libraries
        self.frags_3 = self._build_library(3)
        self.frags_9 = self._build_library(9)

    def _build_library(self, length: int) -> dict[int, list]:
        """Build fragment library for given length."""
        library = {}
        for start in range(self.n - length + 1):
            frags = []
            for _ in range(self.n_frags):
                phi, psi = generate_fragment(self.ss, start, length, self._rng)
                frags.append((phi, psi))
            library[start] = frags
        return library

    def sample_fragment(self, start: int, length: int = 9) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random fragment at the given position."""
        lib = self.frags_9 if length >= 9 else self.frags_3
        max_start = self.n - length
        start = min(start, max_start)
        if start < 0:
            start = 0
        if start not in lib:
            return generate_fragment(self.ss, start, length, self._rng)
        idx = self._rng.randint(len(lib[start]))
        return lib[start][idx]
