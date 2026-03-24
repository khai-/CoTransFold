"""Simulation trajectory — records snapshots at each translation step."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cotransfold.core.conformation import BackboneState
from cotransfold.core.residue import AminoAcid


@dataclass
class StepSnapshot:
    """State recorded after adding one residue."""
    step: int
    amino_acid: AminoAcid
    chain_length: int
    num_exposed: int
    energy_total: float
    energy_decomposed: dict[str, float]
    folding_time: float           # seconds available for folding at this step
    is_rare_codon: bool
    minimization_iterations: int
    minimization_converged: bool
    backbone: BackboneState       # Copy of backbone state after minimization


@dataclass
class SimulationTrajectory:
    """Complete trajectory of a co-translational folding simulation."""
    snapshots: list[StepSnapshot] = field(default_factory=list)
    organism: str = ''
    sequence_str: str = ''

    def add_snapshot(self, snapshot: StepSnapshot) -> None:
        self.snapshots.append(snapshot)

    @property
    def num_steps(self) -> int:
        return len(self.snapshots)

    @property
    def final_backbone(self) -> BackboneState | None:
        if self.snapshots:
            return self.snapshots[-1].backbone
        return None

    @property
    def energy_trace(self) -> np.ndarray:
        """Array of total energies at each step."""
        return np.array([s.energy_total for s in self.snapshots])

    @property
    def exposed_trace(self) -> np.ndarray:
        """Array of number of exposed residues at each step."""
        return np.array([s.num_exposed for s in self.snapshots])

    def get_energy_component_trace(self, component: str) -> np.ndarray:
        """Get energy trace for a specific component (e.g., 'hbond')."""
        return np.array([
            s.energy_decomposed.get(component, 0.0) for s in self.snapshots
        ])

    def summary(self) -> str:
        """Human-readable summary of the simulation."""
        if not self.snapshots:
            return "Empty trajectory"
        first = self.snapshots[0]
        last = self.snapshots[-1]
        rare_count = sum(1 for s in self.snapshots if s.is_rare_codon)
        return (
            f"CoTransFold Simulation Summary\n"
            f"{'=' * 40}\n"
            f"Organism: {self.organism}\n"
            f"Sequence: {self.sequence_str[:50]}{'...' if len(self.sequence_str) > 50 else ''}\n"
            f"Length: {last.chain_length} residues\n"
            f"Exposed at end: {last.num_exposed} residues\n"
            f"Final energy: {last.energy_total:.2f} kcal/mol\n"
            f"Rare codons: {rare_count}\n"
            f"Energy components:\n"
            + '\n'.join(f"  {k}: {v:.2f}" for k, v in last.energy_decomposed.items())
        )
