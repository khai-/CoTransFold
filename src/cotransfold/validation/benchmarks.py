"""Benchmark proteins for validation.

Small, well-studied proteins with known experimental structures
and fast folding kinetics. These are the standard test cases
for protein folding methods.

Reference CA coordinates are from ideal secondary structures
(since we don't ship PDB files). For real validation, load
experimental coordinates from the PDB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.structure.coordinates import torsion_to_cartesian, get_ca_coords


@dataclass
class BenchmarkProtein:
    """A benchmark protein for validation."""
    name: str
    pdb_id: str
    sequence: str
    expected_ss: str        # Expected secondary structure (H=helix, E=strand, C=coil)
    description: str
    reference_ca: np.ndarray | None = None  # Experimental CA coords if available


def _make_ideal_ca(ss: str) -> np.ndarray:
    """Generate ideal CA coordinates from secondary structure string.

    Uses ideal phi/psi angles for each SS type:
    H: alpha-helix (phi=-57, psi=-47)
    E: beta-strand (phi=-120, psi=130)
    C: coil (phi=-80, psi=0)
    """
    n = len(ss)
    phi = np.zeros(n)
    psi = np.zeros(n)

    for i, s in enumerate(ss):
        if s == 'H':
            phi[i] = np.radians(-57)
            psi[i] = np.radians(-47)
        elif s == 'E':
            phi[i] = np.radians(-120)
            psi[i] = np.radians(130)
        else:  # C
            phi[i] = np.radians(-80)
            psi[i] = np.radians(0)

    backbone = BackboneState(phi=phi, psi=psi, omega=np.full(n, np.pi))
    coords = torsion_to_cartesian(backbone)
    return get_ca_coords(coords)


# Chignolin — 10-residue miniprotein with beta-hairpin
# PDB: 1UAO
CHIGNOLIN = BenchmarkProtein(
    name="Chignolin",
    pdb_id="1UAO",
    sequence="GYDPETGTWG",
    expected_ss="CCEECCEECC",
    description="10-residue designed beta-hairpin miniprotein. "
                "Folds in ~0.6 μs. Contains a type I' beta-turn.",
)

# Trp-cage — 20-residue miniprotein with alpha-helix and polyproline
# PDB: 1L2Y
TRP_CAGE = BenchmarkProtein(
    name="Trp-cage",
    pdb_id="1L2Y",
    sequence="NLYIQWLKDGGPSSGRPPPS",
    expected_ss="CHHHHHHHCCCCCCCCCCCC",
    description="20-residue miniprotein. Folds in ~4 μs. "
                "N-terminal alpha-helix wraps around Trp6.",
)

# Villin headpiece — 35-residue three-helix bundle
# PDB: 1VII
VILLIN_HEADPIECE = BenchmarkProtein(
    name="Villin headpiece",
    pdb_id="1VII",
    sequence="LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    expected_ss="CHHHHHHHHHCCHHHHHHHHHCCHHHHHHHHHHHHC",
    description="35-residue three-helix bundle subdomain. "
                "Folds in ~0.7 μs. Classic fast-folding benchmark.",
)

# Protein B (BdpA) — 46-residue three-helix bundle
# PDB: 1BDD (domain)
PROTEIN_B = BenchmarkProtein(
    name="Protein B",
    pdb_id="1BDD",
    sequence="TADNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSVSKEIL",
    expected_ss="CCCHHHHHHHHHHHCCCCCHHHHHHHHHHHCCCHHHHHHHHHHHHHCC",
    description="46-residue three-helix bundle from Protein A. "
                "Folds in ~6 μs.",
)

# WW domain — 34-residue all-beta protein
# PDB: 1PIN
WW_DOMAIN = BenchmarkProtein(
    name="WW domain",
    pdb_id="1PIN",
    sequence="KLPPGWEKRMSRDGRVYYFNHITGTTQFERPSG",
    expected_ss="CCCCCEEEECCCCCEEEECCCCCCEEEEECCCC",
    description="34-residue three-stranded beta-sheet. "
                "Folds in ~13 μs. Tests beta-sheet prediction.",
)


# All benchmark proteins
ALL_BENCHMARKS = [CHIGNOLIN, TRP_CAGE, VILLIN_HEADPIECE, PROTEIN_B, WW_DOMAIN]


def get_benchmark(name: str) -> BenchmarkProtein:
    """Look up a benchmark protein by name or PDB ID."""
    for bp in ALL_BENCHMARKS:
        if bp.name.lower() == name.lower() or bp.pdb_id.lower() == name.lower():
            return bp
    raise ValueError(f"Unknown benchmark: {name}. "
                     f"Available: {[b.name for b in ALL_BENCHMARKS]}")


def get_reference_ca(benchmark: BenchmarkProtein) -> np.ndarray:
    """Get reference CA coordinates for a benchmark protein.

    If experimental coordinates are not loaded, generates ideal
    coordinates from the expected secondary structure.
    """
    if benchmark.reference_ca is not None:
        return benchmark.reference_ca
    return _make_ideal_ca(benchmark.expected_ss)
