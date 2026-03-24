# Benchmark Baseline Results — v0.1.0

Date: 2026-03-24

## Results

| Protein | Res | RMSD (Å) | TM-score | GDT-TS | Time |
|---------|-----|----------|----------|--------|------|
| Chignolin | 10 | 4.08 | 0.021 | 0.400 | 0.1s |
| Trp-cage | 20 | 9.07 | 0.013 | 0.188 | 3.6s |
| Villin headpiece | 35 | 19.19 | 0.039 | 0.071 | 41.4s |
| Protein B | 46 | 22.48 | 0.052 | 0.071 | 117.2s |
| WW domain | 33 | 13.72 | 0.027 | 0.076 | 32.5s |

## Analysis

All structures remain in extended conformation (100% strand). No helix formed.

### Root Causes
1. **Residues start as extended** (phi=-120°, psi=120°) — this is correct for the tunnel
2. **Tunnel freezes most residues** — only exposed residues are free to optimize
3. **Minimizer iterations too low** — 50 max steps insufficient to escape extended basin
4. **H-bond energy alone isn't enough** to drive helix formation with current weights
5. **No post-exit equilibration** — once all residues are out, no further optimization occurs

### What Needs To Change (Next Iteration)
1. **Post-tunnel equilibration phase** — after translation completes, run extended minimization on full chain
2. **Helix seeding in vestibule** — when vestibule is wide enough (~20Å), bias initial angles toward helix for helix-forming residues
3. **Increase minimization steps** for exposed residues (especially at rare codons)
4. **Tune energy weights** — increase H-bond weight relative to Ramachandran
5. **Two-stage minimizer** — coarse (MC with big moves) then fine (L-BFGS-B)

These are engineering improvements to the existing architecture — the physics model is correct.
