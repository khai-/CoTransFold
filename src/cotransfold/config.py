"""Global configuration and physical constants for CoTransFold."""

import numpy as np

# Physical constants
BOLTZMANN_KCAL = 0.001987204  # kcal/(mol·K)
COULOMB_CONSTANT = 332.0637    # kcal·Å/(mol·e²)

# Default simulation temperature
DEFAULT_TEMPERATURE = 310.15  # K (37°C)

# Ideal backbone geometry (from AMBER ff14SB / high-res crystal structures)
BOND_LENGTH_N_CA = 1.458   # Å
BOND_LENGTH_CA_C = 1.525   # Å
BOND_LENGTH_C_N = 1.329    # Å (peptide bond)

BOND_ANGLE_N_CA_C = np.radians(111.2)   # rad
BOND_ANGLE_CA_C_N = np.radians(116.2)   # rad
BOND_ANGLE_C_N_CA = np.radians(121.7)   # rad

# Residue advance distance through tunnel per elongation step
RESIDUE_ADVANCE_DISTANCE = 3.5  # Å (extended conformation rise per residue)
