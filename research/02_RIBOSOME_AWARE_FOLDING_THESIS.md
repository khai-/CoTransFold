# The Ribosome-Aware Folding Thesis

## Core Insight
Protein folding is NOT a thermodynamic equilibrium problem solved in free solution.
It is a **sequential, kinetically controlled process** shaped by the ribosome's physical geometry,
electrostatics, translation speed, and chaperone program.

**No existing structure predictor models this.** This is the open opportunity.

---

## 1. THE RIBOSOME IS A FOLDING MACHINE

### Exit Tunnel Geometry
- **Length**: ~80-100Å (accommodates ~30-40 residues in extended conformation)
- **Width**: 10-20Å at widest, narrowing to **~8Å at the constriction** (uL4/uL22 proteins)
- **Three functional zones**:
  1. **Upper tunnel** (near PTC): ~10Å wide, prevents all folding
  2. **Constriction** (uL4/uL22): dynamic gate that transiently narrows below water-molecule diameter and opens wide enough for alpha-helices (McGrath & Kolar, bioRxiv 2026)
  3. **Vestibule** (near exit): ~20Å wide, alpha-helices can form here
- **Bacterial tunnels are ~39% larger** in volume than eukaryotic ones
- Six eukaryotic protists have prokaryote-like tunnels, challenging binary classification

### Electrostatic Environment
- **Positive charges at the constriction** — interacts with negatively charged nascent chains
- **Negative charges at the exit** — repels negatively charged proteins
- ~50% of proteome (negatively charged) experiences **+8.4 kcal/mol destabilization** on ribosome surface
- ~40% of proteome (positively charged) is unaffected or stabilized
- **The ribosome creates fundamentally different energy landscapes based on protein charge**

### Active Entropic Contribution
- **The ribosome expands unfolded nascent chains**, reducing entropic penalty of folding by **up to 30 kcal/mol** (Streit & Christodoulou, Nature 2024)
- This creates on-ribosome folding intermediates that **do not exist in free solution**
- Cryo-EM has captured two stable intermediates (I1, I2) of an immunoglobulin domain on the ribosome that cannot form off-ribosome (Waudby et al., bioRxiv 2025)

### The Tunnel Is Not Passive
- Nascent chain conformation in the tunnel **allosterically modulates PTC geometry** over 10-30Å distances
- The ribosome directs unfolded vs. folded nascent chains through **two different pathways** at the exit (Cassaignau et al., bioRxiv 2025)
- Folding generates **10-50 pN pulling forces** transmitted 100Å back to the PTC
- Charged residues create electrostatic forces that **mechanochemically alter translation speed**

---

## 2. TRANSLATION KINETICS DETERMINE FOLD OUTCOME

### Codon Usage Creates Folding Pauses
- Rare codons cause translational pauses (~10-100x slower)
- These pauses give specific segments time to fold before downstream residues emerge
- Conserved rare codons appear **~30 residues downstream** of optimal cotranslational folding lengths (Bitran & Shakhnovich, PNAS 2020)

### Synonymous Mutations Cause Disease (Proof of Concept)
- **CFTR (cystic fibrosis)**: synonymous mutations (same amino acid, different codon) that change translation speed cause protein misfolding and disease (Nature Comms, 2020)
- Same sequence → different structure, purely due to translation kinetics
- **This directly violates the assumption that sequence alone determines structure**

### Vectorial Folding Changes the Game
- N-terminus folds first, before C-terminus exists
- A designed fluorescent protein shows **2-fold bias toward N-terminal structure** when translated vs refolded in vitro
- Multi-domain proteins: the human ribosome **delays domain docking** — domains fold independently before assembling (Pellowe et al., Nat. Struct. Mol. Biol., 2025)
- N-terminus location matters: PNAS 2024 showed it significantly impacts folding outcome

---

## 3. CHAPERONE PROGRAM IS PART OF THE SYSTEM

### Chaperones Sense the Nascent Chain INSIDE the Tunnel
- **NAC (Nascent polypeptide-Associated Complex)** senses nascent chains when they are **<30 amino acids long** — still inside the tunnel (Lee et al., Nature 2025)
- This is far earlier than previously thought

### Chaperone Actions
| Chaperone | When | Action |
|---|---|---|
| **NAC** | <30 aa (in tunnel) | Senses and prepares for exit |
| **Trigger Factor** | At tunnel exit | Accelerates folding by enhancing polypeptide collapse (PNAS 2025) |
| **SRP** | At tunnel exit | Targets membrane proteins to ER/membrane |
| **Hsp70/Ssb** | Post-exit | Prevents aggregation, assists folding |
| **GroEL/ES** | Post-exit | Unfolds-then-encapsulates on the ribosome (Nature Comms 2025) |

### Combinatorial Program
- A **proteome-wide combinatorial chaperone program** has been mapped (Nature Comms, 2025)
- Different proteins get different chaperone combinations
- The program is sequence-dependent and predictable

---

## 4. WHY THIS EXPLAINS ALPHAFOLD'S FAILURES

| AlphaFold Failure | Co-translational Explanation |
|---|---|
| **Fold-switching proteins** (~100 known, potentially 4% of PDB) | Marginal stability (>-3 kcal/mol) means co-translational kinetics determine which fold is adopted. AlphaFold fails on 94% of these |
| **Multi-domain proteins** | Domains fold sequentially as they exit; AlphaFold treats them simultaneously |
| **Intrinsically disordered regions** | May be ordered co-translationally but disordered in solution |
| **Membrane proteins** | SRP recognition at the tunnel exit determines topology; AlphaFold ignores this |
| **Allosteric conformations** | Translation kinetics may bias toward specific conformational states |

---

## 5. EXISTING COMPUTATIONAL MODELS (Limited)

### GPCTF Framework (2025)
- General Protein Cotranslational Folding simulation platform
- Generated >8 ms of MD trajectory data
- Shows cotranslational folding produces **more helix-rich structures** with **fewer non-native contacts** than refolding
- Most advanced existing tool, but not used for structure prediction

### Coarse-Grained Tunnel Models
- Automated CG pipeline accurately captures tunnel geometry (bioRxiv 2025)
- Enables fast simulation of nascent chain dynamics in the tunnel
- Could be the basis for a practical prediction tool

### The Gap
**No one has built a structure predictor that models co-translational folding.**
Every tool (AlphaFold, ESMFold, Rosetta, MD) assumes folding from a complete chain in solution.

---

## 6. THE DETERMINISTIC PREDICTOR ARCHITECTURE

### Inputs
1. **Amino acid sequence** (primary)
2. **Codon sequence** (translation speed profile)
3. **Ribosome tunnel model** (geometry + electrostatics, organism-specific)
4. **Chaperone program** (which chaperones, when)

### Process: Sequential Simulation
```
For each residue position i = 1 to N:
  1. Add residue i to the nascent chain
  2. Compute translation pause (from codon usage)
  3. Apply tunnel constraints (geometry, electrostatics)
  4. Run energy minimization for the exposed portion
  5. Apply chaperone effects at appropriate chain lengths
  6. Record intermediate structure
Final structure = result of sequential folding process
```

### Why This Is Deterministic
- Each step is governed by known physics (tunnel geometry, electrostatics, energy function)
- Translation speed is determined by codon sequence
- Chaperone engagement is predictable from sequence features
- No random sampling, no statistical inference — just sequential physics

### Key Advantages Over AlphaFold
1. **Models the actual biological process**, not a thermodynamic abstraction
2. **Uses codon information** — AlphaFold ignores this entirely
3. **Naturally handles multi-domain proteins** — sequential folding
4. **Explains fold-switching** — translation kinetics + tunnel environment
5. **Organism-specific** — different ribosomes, different tunnels, different predictions
6. **Physically grounded** — every prediction has a causal explanation

---

## 7. KEY PAPERS (Ribosome + Co-translational Folding)

### Foundational
1. Streit, Christodoulou et al. (2024) — "The ribosome lowers the entropic penalty of protein folding" — *Nature*
2. Lee et al. (2025) — "NAC controls nascent chain fate through tunnel sensing" — *Nature*
3. Bitran & Shakhnovich (2020) — "Cotranslational folding allows misfolding-prone proteins to circumvent deep kinetic traps" — *PNAS*
4. Pellowe et al. (2025) — "The human ribosome modulates multidomain protein biogenesis" — *Nat. Struct. Mol. Biol.*

### Structural
5. Waudby et al. (2025) — "Structures of protein folding intermediates on the ribosome" — *bioRxiv*
6. Wang et al. (2025) — "Cotranslational protein folding through non-native intermediates" — *Science Advances*
7. Gersteuer et al. (2024) — "SecM arrest peptide traps pre-peptide-bond state at 2.0Å" — *Nat. Commun.*
8. McGrath & Kolar (2026) — "Early nascent polypeptide dynamics coupled to tunnel constriction flexibility" — *bioRxiv*

### Computational
9. GPCTF framework (2025) — >8 ms trajectory data for cotranslational folding
10. Advanced CG tunnel model (2025) — automated pipeline for tractable tunnel simulations

### Translation Kinetics
11. CFTR synonymous mutations (2020) — proof that codon choice affects fold — *Nature Comms*
12. Chakravarty & Porter (2025) — fold-switching proteins review — *Ann. Rev. Biophysics*
13. Native Fold Delay metric (2025) — quantifies temporal delays in co-translational folding — *Nature Comms*

### Chaperone Program
14. Till et al. (2025) — "Trigger factor accelerates nascent chain compaction" — *PNAS*
15. Proteome-wide chaperone program mapping (2025) — *Nature Comms*
16. GroEL/ES unfolds-then-encapsulates on ribosome (2025) — *Nature Comms*
