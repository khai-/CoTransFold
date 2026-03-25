"""Co-translational folding simulation engine.

The main loop that ties everything together:
1. Sequentially add residues at the PTC
2. Advance chain through the ribosome exit tunnel
3. Apply tunnel constraints and energy minimization
4. Record trajectory snapshots

This is the core of the CoTransFold simulator — the first protein
structure predictor that models the actual co-translational folding
process rather than thermodynamic equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cotransfold.core.residue import AminoAcid, sequence_from_string, sequence_to_string
from cotransfold.core.chain import NascentChain
from cotransfold.core.trajectory import SimulationTrajectory, StepSnapshot
from cotransfold.energy.total import TotalEnergy
from cotransfold.energy.ramachandran import RamachandranEnergy
from cotransfold.energy.hbond import HydrogenBondEnergy
from cotransfold.energy.vanderwaals import VanDerWaalsEnergy
from cotransfold.energy.bonded import BondedEnergy
from cotransfold.energy.tunnel_energy import TunnelEnergy
from cotransfold.energy.solvent import SolventEnergy
from cotransfold.tunnel.geometry import TunnelGeometry
from cotransfold.tunnel.electrostatics import TunnelElectrostatics
from cotransfold.tunnel.organisms import get_tunnel
from cotransfold.kinetics.translation import TranslationSchedule
from cotransfold.chaperones.program import ChaperoneProgram
from cotransfold.minimizer.gradient import GradientMinimizer
from cotransfold.structure.coordinates import torsion_to_cartesian


@dataclass
class SimulationConfig:
    """Configuration for a co-translational folding simulation."""
    organism: str = 'ecoli'
    temperature: float = 310.15          # K (37°C)
    min_steps_per_residue: int = 20      # Minimum minimization steps
    max_steps_per_residue: int = 100     # Maximum minimization steps
    time_to_steps_factor: float = 1000.0 # seconds -> minimization steps
    use_tunnel: bool = True
    use_kinetics: bool = True
    use_chaperones: bool = True
    use_solvent: bool = True

    # Energy term weights
    w_ramachandran: float = 1.0
    w_hbond: float = 1.0
    w_vanderwaals: float = 0.5
    w_bonded: float = 0.5
    w_tunnel: float = 1.0
    w_solvent: float = 0.5

    # Tunnel energy parameters
    tunnel_wall_spring: float = 5.0
    tunnel_wall_buffer: float = 1.5
    tunnel_elec_weight: float = 1.0

    # Minimizer parameters
    minimizer: str = 'numpy'            # 'jax' (autodiff) or 'numpy' (finite differences)
    minimizer_ftol: float = 1e-5
    minimizer_gtol: float = 1e-3
    minimizer_gradient_step: float = 1e-4


class SimulationEngine:
    """Co-translational protein folding simulator.

    Models the sequential emergence of a nascent polypeptide from the
    ribosome exit tunnel, with energy minimization at each step constrained
    by tunnel geometry, electrostatics, and translation kinetics.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._setup()

    def _setup(self) -> None:
        """Initialize tunnel, energy function, and minimizer."""
        cfg = self.config

        # Tunnel
        if cfg.use_tunnel:
            self._tunnel_geom, self._tunnel_elec = get_tunnel(cfg.organism)
        else:
            self._tunnel_geom = None
            self._tunnel_elec = None

        # Energy function
        self._energy = TotalEnergy()
        self._energy.add_term(RamachandranEnergy(cfg.temperature), cfg.w_ramachandran)
        self._energy.add_term(HydrogenBondEnergy(), cfg.w_hbond)
        self._energy.add_term(VanDerWaalsEnergy(), cfg.w_vanderwaals)
        self._energy.add_term(BondedEnergy(), cfg.w_bonded)

        if cfg.use_tunnel and self._tunnel_geom and self._tunnel_elec:
            self._energy.add_term(
                TunnelEnergy(
                    self._tunnel_geom, self._tunnel_elec,
                    wall_spring=cfg.tunnel_wall_spring,
                    wall_buffer=cfg.tunnel_wall_buffer,
                    elec_weight=cfg.tunnel_elec_weight,
                ),
                cfg.w_tunnel,
            )

        if cfg.use_solvent:
            self._energy.add_term(SolventEnergy(), cfg.w_solvent)

        # Chaperone program
        self._chaperone_program = None
        if cfg.use_chaperones:
            self._chaperone_program = ChaperoneProgram(cfg.organism)

        # Minimizer
        self._use_jax = cfg.minimizer == 'jax'
        if self._use_jax:
            from cotransfold.minimizer.jax_minimizer import JaxMinimizer
            self._jax_minimizer = JaxMinimizer(
                max_iterations=cfg.max_steps_per_residue,
                ftol=cfg.minimizer_ftol,
                gtol=cfg.minimizer_gtol,
            )
            self._jax_weights = {
                'ramachandran': cfg.w_ramachandran,
                'hbond': cfg.w_hbond,
                'vanderwaals': cfg.w_vanderwaals,
                'bonded': cfg.w_bonded,
                'solvent': cfg.w_solvent if cfg.use_solvent else 0.0,
                'tunnel': cfg.w_tunnel if cfg.use_tunnel else 0.0,
            }
        self._minimizer = GradientMinimizer(
            max_iterations=cfg.max_steps_per_residue,
            gradient_step=cfg.minimizer_gradient_step,
            ftol=cfg.minimizer_ftol,
            gtol=cfg.minimizer_gtol,
        )

    def simulate(self,
                 sequence: list[AminoAcid] | str,
                 codons: list[str] | None = None,
                 callback: Callable[[StepSnapshot], None] | None = None,
                 ) -> SimulationTrajectory:
        """Run a full co-translational folding simulation.

        Args:
            sequence: amino acid sequence (list of AminoAcid or one-letter string)
            codons: optional mRNA codon sequence for translation kinetics
            callback: optional function called after each step

        Returns:
            SimulationTrajectory with snapshots at each residue addition
        """
        # Parse sequence
        if isinstance(sequence, str):
            aa_seq = sequence_from_string(sequence)
            seq_str = sequence
        else:
            aa_seq = list(sequence)
            seq_str = sequence_to_string(aa_seq)

        n = len(aa_seq)
        cfg = self.config

        # Build translation schedule
        if codons and cfg.use_kinetics:
            from cotransfold.kinetics.codon_table import CodonUsageTable
            table = CodonUsageTable.ecoli()
            schedule = TranslationSchedule.from_codons(codons, table)
        elif cfg.use_kinetics:
            schedule = TranslationSchedule.from_amino_acids(aa_seq, cfg.organism)
        else:
            schedule = TranslationSchedule.uniform(n)

        # Initialize
        chain = NascentChain.empty()
        trajectory = SimulationTrajectory(
            organism=cfg.organism,
            sequence_str=seq_str,
        )

        tunnel_length = self._tunnel_geom.length if self._tunnel_geom else 90.0

        # === Main simulation loop ===
        for i in range(n):
            aa = aa_seq[i]

            # 1. Add residue at PTC with initial extended conformation
            chain.add_residue(aa)

            # 2. Update tunnel exposure
            chain.update_exposure(tunnel_length)

            # 3. Compute frozen mask from tunnel position
            if cfg.use_tunnel and self._tunnel_geom:
                p = self._tunnel_geom.params
                frozen_mask = chain.get_frozen_mask(
                    upper_end=p.upper_end,
                    constriction_end=p.constriction_end,
                )
            else:
                frozen_mask = np.ones(chain.chain_length)

            # 4. Determine minimization steps from translation kinetics
            fold_time = schedule.total_times[i]
            n_steps = int(fold_time * cfg.time_to_steps_factor)
            n_steps = max(cfg.min_steps_per_residue,
                          min(n_steps, cfg.max_steps_per_residue))

            # 5. Query chaperone program
            chaperone_actions = []
            chaperone_energy = 0.0
            if self._chaperone_program:
                elapsed = float(schedule.cumulative_times[i])
                chaperone_actions = self._chaperone_program.active_chaperones(
                    chain, elapsed)
                chaperone_energy = self._chaperone_program.get_total_energy_modifier(
                    chaperone_actions)

            # 6. Energy minimization
            energy_kwargs = {}
            if cfg.use_tunnel:
                energy_kwargs['tunnel_positions'] = chain.tunnel_position
                energy_kwargs['tunnel_length'] = tunnel_length

            if self._use_jax:
                jax_kwargs = dict(energy_kwargs)
                jax_kwargs.pop('tunnel_length', None)
                result = self._jax_minimizer.minimize(
                    chain, None,
                    frozen_mask=frozen_mask,
                    max_iterations=n_steps,
                    weights=self._jax_weights,
                    tunnel_length=tunnel_length,
                    **jax_kwargs,
                )
            else:
                result = self._minimizer.minimize(
                    chain, self._energy,
                    frozen_mask=frozen_mask,
                    max_iterations=n_steps,
                    **energy_kwargs,
                )

            # 7. Compute final energy decomposition
            energy_decomposed = self._energy.compute_decomposed(chain, **energy_kwargs)
            if chaperone_energy != 0.0:
                energy_decomposed['chaperone'] = chaperone_energy
            energy_total = sum(energy_decomposed.values())

            # 8. Record snapshot
            snapshot = StepSnapshot(
                step=i,
                amino_acid=aa,
                chain_length=chain.chain_length,
                num_exposed=chain.num_exposed,
                energy_total=energy_total,
                energy_decomposed=energy_decomposed,
                folding_time=fold_time,
                is_rare_codon=bool(schedule.is_rare[i]),
                minimization_iterations=result.n_iterations,
                minimization_converged=result.converged,
                backbone=chain.backbone.copy(),
            )
            trajectory.add_snapshot(snapshot)

            if callback:
                callback(snapshot)

        return trajectory

    def simulate_and_export(self,
                            sequence: list[AminoAcid] | str,
                            output_pdb: str,
                            codons: list[str] | None = None,
                            ) -> SimulationTrajectory:
        """Run simulation and export final structure to PDB.

        Args:
            sequence: amino acid sequence
            output_pdb: path to write PDB file
            codons: optional mRNA codons

        Returns:
            SimulationTrajectory
        """
        from cotransfold.structure.pdb_io import write_pdb

        if isinstance(sequence, str):
            aa_seq = sequence_from_string(sequence)
        else:
            aa_seq = list(sequence)

        trajectory = self.simulate(sequence, codons)

        if trajectory.final_backbone:
            coords = torsion_to_cartesian(trajectory.final_backbone)
            write_pdb(output_pdb, coords, aa_seq)

        return trajectory
