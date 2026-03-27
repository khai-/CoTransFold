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

from cotransfold.core.residue import AminoAcid, RESIDUE_PROPERTIES, sequence_from_string, sequence_to_string
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

    # Post-translation equilibration
    equilibration_steps: int = 500    # Minimization steps after translation completes
    use_equilibration: bool = True

    # Helix seeding in vestibule
    use_helix_seeding: bool = True
    helix_propensity_threshold: float = 1.1  # Chou-Fasman threshold for helix seeding

    # Energy term weights
    w_ramachandran: float = 1.0
    w_hbond: float = 1.3
    w_vanderwaals: float = 0.7
    w_bonded: float = 0.5
    w_tunnel: float = 1.0
    w_solvent: float = 1.3

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
        self._use_fast = cfg.minimizer == 'fast'
        self._energy_weights = {
            'ramachandran': cfg.w_ramachandran,
            'hbond': cfg.w_hbond,
            'vanderwaals': cfg.w_vanderwaals,
            'bonded': cfg.w_bonded,
            'solvent': cfg.w_solvent if cfg.use_solvent else 0.0,
            'tunnel': cfg.w_tunnel if cfg.use_tunnel else 0.0,
        }
        if self._use_jax:
            from cotransfold.minimizer.jax_minimizer import JaxMinimizer
            self._jax_minimizer = JaxMinimizer(
                max_iterations=cfg.max_steps_per_residue,
                ftol=cfg.minimizer_ftol,
                gtol=cfg.minimizer_gtol,
            )
        if self._use_fast:
            from cotransfold.minimizer.fast_minimizer import FastMinimizer
            self._fast_minimizer = FastMinimizer(
                max_iterations=cfg.max_steps_per_residue,
                ftol=cfg.minimizer_ftol,
                gtol=cfg.minimizer_gtol,
            )
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

            # 1. Add residue at PTC
            # Helix seeding: if this residue is a helix-former and the
            # vestibule is wide enough, initialize with helical angles
            # instead of extended. This models the observation that
            # alpha-helices form in the ribosome vestibule (cryo-EM data).
            init_phi = np.radians(-120.0)  # default: extended
            init_psi = np.radians(120.0)
            if cfg.use_helix_seeding and chain.chain_length >= 3:
                # Check if the previous few residues are in the vestibule
                # (where helices can form, ~50-90Å from PTC)
                if (chain.chain_length > 0 and
                    len(chain.tunnel_position) > 0 and
                    chain.tunnel_position[0] > 50.0):  # Earliest residue past vestibule start
                    # Check if this is a helix-forming residue
                    props = RESIDUE_PROPERTIES[aa]
                    if props.helix_propensity >= cfg.helix_propensity_threshold:
                        init_phi = np.radians(-57.0)  # alpha-helix
                        init_psi = np.radians(-47.0)
            chain.add_residue(aa, phi=init_phi, psi=init_psi)

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
                    weights=self._energy_weights,
                    tunnel_length=tunnel_length,
                    **jax_kwargs,
                )
            elif self._use_fast:
                result = self._fast_minimizer.minimize(
                    chain, self._energy,
                    frozen_mask=frozen_mask,
                    max_iterations=n_steps,
                    weights=self._energy_weights,
                    **energy_kwargs,
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

        # === Post-translation equilibration ===
        # After all residues are translated, run extended minimization
        # on the full chain with all residues free. This models the
        # post-ribosomal folding that occurs after the protein is released.
        if cfg.use_equilibration and cfg.equilibration_steps > 0 and chain.chain_length > 0:
            # All residues are now free (no tunnel constraints)
            free_mask = np.ones(chain.chain_length)

            # Energy kwargs without tunnel (protein has been released)
            eq_kwargs = {}
            if cfg.use_tunnel:
                # Keep tunnel positions for solvent calculation but
                # mark all as outside tunnel for energy
                eq_kwargs['tunnel_positions'] = np.full(
                    chain.chain_length, tunnel_length + 100.0)
                eq_kwargs['tunnel_length'] = tunnel_length

            if self._use_fast:
                result = self._fast_minimizer.minimize(
                    chain, self._energy,
                    frozen_mask=free_mask,
                    max_iterations=cfg.equilibration_steps,
                    weights=self._energy_weights,
                    **eq_kwargs,
                )
            else:
                result = self._minimizer.minimize(
                    chain, self._energy,
                    frozen_mask=free_mask,
                    max_iterations=cfg.equilibration_steps,
                    **eq_kwargs,
                )

            # Record final equilibrated state as last snapshot
            energy_decomposed = self._energy.compute_decomposed(chain, **eq_kwargs)
            energy_total = sum(energy_decomposed.values())
            eq_snapshot = StepSnapshot(
                step=n,
                amino_acid=aa_seq[-1],
                chain_length=chain.chain_length,
                num_exposed=chain.chain_length,  # All exposed post-release
                energy_total=energy_total,
                energy_decomposed=energy_decomposed,
                folding_time=0.0,
                is_rare_codon=False,
                minimization_iterations=result.n_iterations,
                minimization_converged=result.converged,
                backbone=chain.backbone.copy(),
            )
            trajectory.add_snapshot(eq_snapshot)

            if callback:
                callback(eq_snapshot)

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
            from cotransfold.structure.confidence import compute_confidence
            coords = torsion_to_cartesian(trajectory.final_backbone)
            conf = compute_confidence(trajectory.final_backbone, aa_seq)
            write_pdb(output_pdb, coords, aa_seq, confidence=conf.per_residue)

        return trajectory
