# The Ribosome Exit Tunnel and Co-Translational Protein Folding: Deep Research Synthesis

## Why This Matters for Deterministic Folding

Proteins do not fold in a vacuum after being fully synthesized. They fold **during translation**, emerging N-terminus first through a ~100 A tunnel inside the ribosome. The tunnel's geometry, electrostatics, and molecular interactions **actively shape the folding energy landscape**. Any deterministic folding model that ignores the ribosome is missing the initial conditions of folding.

---

## 1. RIBOSOME EXIT TUNNEL: EXACT DIMENSIONS AND ARCHITECTURE

### Overall Geometry
- **Length**: ~80-100 A from the peptidyl transferase center (PTC) to the exit port
- **Width**: 10-20 A diameter (varies along length)
- **Volume**: Bacteria: (3.85 +/- 0.37) x 10^4 A^3; Eukaryotes: (2.78 +/- 0.13) x 10^4 A^3; Archaea: intermediate
- **Capacity**: Can accommodate ~30-40 amino acids in extended conformation, ~60-70 in compacted/alpha-helical form
- **Composition**: Primarily 23S rRNA (bacteria) / 28S rRNA (eukaryotes), lined by loops of ribosomal proteins

### Three Functional Zones (Kolar et al., WIREs RNA, 2024)
The tunnel is divided into three stages based on the position of the nascent chain N-terminus:

| Zone | Location | Width | Key Features |
|------|----------|-------|--------------|
| **Upper tunnel (PTC-proximal)** | 0-30 A from PTC | ~10 A | Narrowest; peptide bond formation; nascent chain constrained to extended or helical |
| **Constriction zone** | 30-50 A from PTC | ~10 A at narrowest | uL4/uL22 beta-hairpin loops protrude inward; major regulatory checkpoint |
| **Lower tunnel/Vestibule** | 50-100 A from PTC | 15-20 A, widening | First tertiary structure possible; uL23, uL24, eL39 line walls; exit port |

> **Key paper**: Kolar MH, McGrath H, Nepomuceno G, Cernekova M. "Three Stages of Nascent Protein Translocation Through the Ribosome Exit Tunnel." *WIREs RNA*, 2024. PMID: 39496527.

### The Constriction Site: A Dynamic Gate

The constriction formed by uL4 and uL22 beta-hairpin loops is **not a static bottleneck** but a dynamic, flexible gate:

- In empty ribosomes, the constriction can **transiently narrow below the diameter of a water molecule**, fully occluding the tunnel
- At other times it **opens wide enough to accommodate a narrow alpha-helix**
- Even a short nascent polypeptide (~5-10 residues) shifts the constriction toward **wider conformations by ~0.2 nm**
- The constriction functions as an **adaptive gate** that responds to its contents

> **Key paper**: McGrath H, Cernekova M, Kolar MH. "Early nascent polypeptide dynamics are coupled to the flexibility of the ribosomal tunnel constriction." *bioRxiv*, March 2026. (Analysis of 222 E. coli ribosome structures + all-atom MD simulations)

### Electrostatic Potential Map

The tunnel interior has a **predominantly negative electrostatic potential** due to rRNA phosphate backbone:

- The phosphate moieties of 23S rRNA create a negatively charged environment throughout
- A **local increase in positive electrostatic potential** occurs at the constriction, where conserved Arg/Lys residues of uL4 (5 conserved between positions 71-92 in eukaryotes) and uL22 (up to 7 Arg/Lys conserved between positions 154-176) protrude into the tunnel
- Strong electrostatic screening is provided by **water molecules** (not mobile ions) attracted to ribosomal phosphate moieties buried near the tunnel wall
- Positively charged residues in the nascent chain experience **electrostatic drag**, slowing translocation and altering translation kinetics
- The motion of nascent polypeptides can be described as **one-dimensional diffusion driven by electrostatics and entropy**

> **Key paper**: Joiret M, Kerff F, Rapino F et al. "Ribosome exit tunnel electrostatics." *Phys Rev E* 105:014409, 2022. PMID: 35193250.

---

## 2. RIBOSOMAL PROTEINS AND rRNA LINING THE TUNNEL

### uL4 (formerly L4)
- **Location**: Constriction site, ~30-35 A from PTC
- **Key feature**: Beta-hairpin loop protrudes into tunnel lumen
- **Function**: Forms one wall of the major constriction; mutations here affect antibiotic sensitivity, translation arrest, and protein folding
- **Conservation**: Universal across all domains of life; the loop tip is highly conserved
- **Interactions**: Nascent chain contacts modulate PTC activity allosterically

### uL22 (formerly L22)
- **Location**: Constriction site, opposite uL4
- **Key feature**: Extended beta-hairpin loop reaches deep into tunnel
- **Function**: Second wall of constriction; critical for macrolide antibiotic binding (erythromycin binds near the uL22 loop)
- **Conservation**: Universal; the beta-hairpin tip is invariant
- **Disease relevance**: Mutations confer macrolide resistance in pathogenic bacteria
- **Together with uL4**: Creates a 10 A bottleneck that can discriminate between folded and unfolded nascent chain segments

> **Key paper**: Ahn M, Wlodarski T, Mitropoulou A et al. "Modulating co-translational protein folding by rational design and ribosome engineering." *Nat Commun* 13:4191, 2022. PMID: 35869078. (Engineered constriction site alterations changed folding outcomes)

### uL23 (formerly L23)
- **Location**: Vestibule/exit port region
- **Key feature**: Extended loop in the vestibule; **primary docking site for chaperones and SRP**
- **Function**: Direct contact with emerging nascent chains; binding site for trigger factor (bacteria), NAC, RAC/Ssb (eukaryotes), and SRP
- **Critical**: Serves as the molecular "landing pad" connecting the tunnel to the entire cotranslational chaperone network
- Cryo-EM confirms that Ssb (the yeast ribosome-bound Hsp70) docks at uL23/Rpl25

### uL24 (formerly L24)
- **Location**: Vestibule, near exit port
- **Function**: Lines the lower tunnel wall; contacts with nascent chains attenuate folding
- **Key finding**: Together with uL23, creates the vestibule environment where initial tertiary folding events occur

### uL29 (formerly L29)
- **Location**: Near tunnel exit on ribosome surface
- **Function**: Longer nascent chains interact weakly with L29; may help orient emerging chains
- **Less studied than uL23/uL24** but implicated in nascent chain surface interactions

### eL39 (eukaryote/archaea-specific)
- **Location**: Vestibule region; **replaces the uL23 tunnel loop** found in bacteria
- **Function**: Creates a positively charged "latch" at the tunnel exit; interacts with the 23S rRNA "gate" structure (helix 24 tetraloop)
- **Critical for proper folding**: Deletion impairs maturation of the exit tunnel and causes **protein misfolding and aggregation during translation**

> **Key paper**: Micic J, Rodriguez-Galan O, Babiano R et al. "Ribosomal protein eL39 is important for maturation of the nascent polypeptide exit tunnel and proper protein folding during translation." *Nucleic Acids Res* 50:6453-6473, 2022. PMID: 35639884.

### 23S rRNA (and 28S rRNA equivalent)
- Forms **~80% of the tunnel wall surface**
- Key nucleotides at the PTC: A2451, U2585, A2602 (E. coli numbering) — catalyze peptide bond formation
- rRNA nucleotides form specific contacts with arrest peptides (see Section 6)
- Helix 24 tetraloop (residues 494-497): acts as a "gate" at the tunnel exit
- Helix 50 and helix 59 regions: interact with nascent chains in the vestibule
- 23S rRNA nucleotides at the PTC are conformationally distorted by arrest peptides like VemP and SecM

---

## 3. THE PEPTIDYL TRANSFERASE CENTER (PTC)

### Mechanism of Peptide Bond Formation
- PTC is an **RNA enzyme (ribozyme)** — no protein residues directly participate in catalysis
- Located at the interface of 50S subunit, formed by domain V of 23S rRNA
- Catalyzes nucleophilic attack of the alpha-amino group of A-site aminoacyl-tRNA on the carbonyl carbon of P-site peptidyl-tRNA
- The 2'-OH of A76 on P-site tRNA acts as a proton shuttle

### How PTC Affects Nascent Chain Conformation
- **Conformational freedom of the nascent peptide impairs PTC decompaction** during peptide bond formation — a straightened peptide anchored in the exit tunnel allows efficient translocation of peptidyl-tRNA from A-site to P-site
- The first ~4-5 residues of the nascent chain are in a **constrained, extended conformation** near the PTC
- **Allosteric communication**: Nascent chain interactions with uL4/uL22 at the constriction can propagate back ~30 A to modulate PTC catalytic geometry
- Arrest peptides exploit this: they distort PTC geometry to prevent peptide bond formation (see Section 6)

### Macrolide Antibiotics and Remote PTC Inhibition
- Macrolides bind in the tunnel ~10 A from the PTC but **remotely inhibit PTC function** depending on the nascent peptide sequence
- The Arg/Lys-X-Arg/Lys motif in the nascent chain is particularly susceptible to macrolide-induced stalling
- This demonstrates that **tunnel contents allosterically regulate PTC activity over distances of 10-30 A**

---

## 4. RIBOSOME-ASSOCIATED CHAPERONES

### 4A. Trigger Factor (TF) — Bacteria

**Structure**: Three domains — N-terminal ribosome-binding domain, PPIase domain, and C-terminal substrate-binding cradle. Total MW ~48 kDa. Forms a "dragon-shaped" molecule.

**Binding site**: Docks on uL23 at the tunnel exit via its N-terminal domain.

**Mechanism (2024-2025 findings)**:
- TF **scans** nascent chains by transient binding events (~50 ms), then **locks** into a stable binding mode (~1 s) as the chain collapses and folds
- This interplay is **reciprocal**: TF binding collapses nascent chains and stabilizes partial folds, while nascent chain compaction prolongs TF binding
- TF binding is **enthalpy-driven** with micromolar affinity to empty ribosomes; **nanomolar affinity** achieved through favorable entropic contribution when nascent chains are present
- TF does NOT destabilize cotranslational folding intermediates — it **protects** incipient structure
- TF alters the cotranslational folding pathway by **keeping the nascent peptide dynamic until the full domain emerges**, preventing premature misfolding

**Coordination with other chaperones**:
- Chaperone binding is disfavored close to the ribosome, **allowing folding to precede chaperone recruitment**
- TF recognizes compact folding intermediates and dictates DnaJ access to nascent chains
- DnaJ recruits DnaK to sequence-diverse solvent-accessible sites
- Neither TF, DnaJ, nor DnaK destabilize cotranslational intermediates — they **collaborate to protect incipient structure**

> **Key papers**:
> - Roeselova A, Maslen SL, Shivakumaraswamy S et al. "Mechanism of chaperone coordination during cotranslational protein folding in bacteria." *Mol Cell* 84:2597-2611, 2024. PMID: 38908370.
> - Wales TE, Pajak A, Roeselova A et al. "Resolving chaperone-assisted protein folding on the ribosome at the peptide level." *Nat Struct Mol Biol* 31:1898-1907, 2024. PMID: 38987455.
> - Till MS et al. "Trigger factor accelerates nascent chain compaction and folding." *PNAS* 122(30), 2025. PMID: 40711920. (Optical tweezers + single-molecule FRET)

### 4B. NAC (Nascent polypeptide-Associated Complex) — Eukaryotes

**Structure**: Heterodimer of alpha-NAC and beta-NAC subunits. Conserved across all eukaryotes.

**Breakthrough discovery (2025)**:
NAC is a **multi-faceted regulator** that coordinates translation elongation, co-translational folding, and organelle targeting through distinct interactions with nascent polypeptides **both inside and outside the ribosome exit tunnel**:

1. **Intra-tunnel sensing**: NAC engages ribosomes with **extremely short nascent polypeptides (<30 amino acids)** still inside the tunnel, in a sequence-specific manner — this was entirely unexpected
2. **Early elongation slowdown**: Initial NAC interactions induce a translation rate reduction that **tunes ribosome flux and prevents ribosome collisions**
3. **Broad proteomic engagement**: Thousands of sequence-specific NAC binding events across the nascent proteome, engaging hydrophobic and helical motifs
4. **Pathway sorting**: NAC distinguishes cytosolic, nuclear, ER, and mitochondrial proteins and directs them to appropriate fates

**NAC as a molecular hub (2026 review)**:
- NAC organizes cotranslational interactions into efficient protein-biogenesis pathways
- Cytonuclear proteins require immediate N-terminal maturation and folding
- ER and mitochondrial proteins must remain unfolded for targeting
- NAC coordinates this by selectively controlling factor access to the ribosome

> **Key papers**:
> - Lee JH, Rabl L, Gamerdinger M et al. "NAC controls nascent chain fate through tunnel sensing and chaperone action." *Nature* 588:474-480, 2025. PMID: 41430436.
> - Gamerdinger M, Burg N, Deuerling E. "Ribosome-NAC collaboration: A regulatory platform for cotranslational chaperones, enzymes, and targeting factors." *Mol Cell* 86:337-351, 2026. PMID: 41605218.

### 4C. RAC/Ssb System — Eukaryotes (Yeast Model)

**Components**:
- **RAC** (Ribosome-Associated Complex): obligate heterodimer of Zuo1 (Hsp40/J-domain protein) + Ssz1 (Hsp70-like, but catalytically inactive)
- **Ssb1/2**: Ribosome-bound Hsp70 that directly binds emerging nascent chains

**Mechanism**:
- RAC on the nascent-chain-free ribosome is in an **autoinhibited conformation**
- When a nascent chain reaches the tunnel exit, RAC undergoes **large-scale structural remodeling**: Zuo1 J-domain becomes accessible to stimulate Ssb's ATPase
- Ssb-ATP is positioned with its substrate-binding domain **close to the tunnel exit** to receive nascent chains
- Cryo-EM identifies **uL23/Rpl25 as the ribosomal binding site** for Ssb

**Coexistence with NAC**:
- NAC and Zuotin/Hsp70 systems **coexist at the ribosome tunnel exit in vivo** — they are not mutually exclusive
- Cross-linking data shows that even in NAC's presence, Hsp70 can position its peptide-binding site at the tunnel exit

> **Key papers**:
> - Zhang Y, Grundmann L, Vollmar L et al. "The cotranslational cycle of the ribosome-bound Hsp70 homolog Ssb." *Nat Commun* 17:886, 2026. PMID: 41545346.
> - Ziegelhoffer T, Verma AK, Delewski W et al. "NAC and Zuotin/Hsp70 chaperone systems coexist at the ribosome tunnel exit in vivo." *Nucleic Acids Res* 52:3346-3358, 2024. PMID: 38224454.
> - Chen Y, Tsai B, Li N et al. "Structural remodeling of ribosome associated Hsp40-Hsp70 chaperones during co-translational folding." *Nat Commun* 13:3410, 2022. PMID: 35701497.

### 4D. SRP (Signal Recognition Particle)

- SRP binds emerging **signal sequences** (hydrophobic N-terminal segments) at the tunnel exit
- Competes with TF/NAC for access to nascent chains at uL23
- FRET studies (2025) show increased dynamic excursions of signal sequences from SRP on ribosomes with longer nascent chains, leading to suboptimal SRP conformation
- There is a **limited temporal window** for cotranslational ER targeting during protein synthesis
- NAC amplifies the effects of longer nascent chains to further exclude SRP from non-ER proteins

---

## 5. FORCE GENERATION DURING TRANSLATION

### Does the Ribosome Exert Mechanical Force on the Nascent Chain?

**Yes — and forces flow in both directions.**

### Folding-Generated Pulling Force
- Cotranslational folding of nascent chain domains outside the tunnel generates a **pulling force** on residues still inside the tunnel
- This force is transmitted **~10 nm (100 A) back to the PTC**, altering the energy barrier to peptide bond formation
- Force generation depends on: domain **stability**, **topology**, and **translation speed**
- Measured forces: **10-50 pN** range (optical tweezers)

### Electrostatic Force
- Positively charged residues (Arg, Lys) generate forces that **move the P-site amino acid away from the A-site amino acid**
- These conformational changes increase the transition state barrier to peptide bond formation
- This is how charged residues **mechanochemically alter translation speed** — the "electrostatic ratchet"

### Force in Ribosome Quality Control (2025-2026 breakthrough)
- When translation stalls, the ribosome-associated quality control (RQC) pathway appends **CAT tails** (CArboxyl-Terminal amino acid tails)
- CAT tailing operates in two modes:
  - **Extrusion mode**: Thr/Ala mixture; increases lysine accessibility for ubiquitylation
  - **Release mode**: Ala-only; triggered by **mechanical pulling forces** (from Cdc48p motor protein pulling on ubiquitinated nascent chain)
- The switch between modes is **force-regulated** — pulling forces on the nascent chain change which amino acids are added
- Failed mode-switching leads to proteotoxic aggregation

> **Key papers**:
> - "Mechanical forces regulate the composition and fate of stalled nascent chains." *Mol Cell*, 2025. (Cdc48p pulling forces switch CAT tail composition)
> - Bustamante CJ et al. "Domain topology, stability, and translation speed determine mechanical force generation on the ribosome." *PNAS* 116:5523-5532, 2019.

---

## 6. ARREST PEPTIDES AND RIBOSOME STALLING

### General Mechanism
Arrest peptides are specific amino acid sequences in the nascent chain that interact with the tunnel walls to **stall the ribosome**. They serve as regulatory sensors (e.g., for membrane protein insertion, amino acid availability, antibiotic presence).

### SecM (E. coli) — Force-Sensing Arrest Peptide

**2024 high-resolution structure (2.0 A)**:
- SecM arrests translation by stabilizing Pro-tRNA in the A-site in a geometry that **prevents peptide bond formation** with the peptidyl-tRNA in the P-site
- SecM forms an alpha-helix deep inside the tunnel (**12 A deeper** than VemP's helix)
- Unlike VemP, the SecM alpha-helix does **not directly perturb PTC nucleotide conformations** — instead it traps a pre-peptide-bond-formation state
- Pulling force on the SecM nascent chain (from Sec translocation machinery) **relieves the stall** — this is how SecM senses whether SecA/SecYEG is functioning

**2025 rescue structure**:
- 2.8 A cryo-EM structure of SecM-stalled ribosome in complex with YheS (an ABCF ATPase that rescues SecM-stalled ribosomes)

> **Key paper**: Gersteuer F et al. "The SecM arrest peptide traps a pre-peptide bond formation state of the ribosome." *Nat Commun* 15:2431, 2024. PMID: 38503753.

### VemP (Vibrio alginolyticus) — Compaction-Sensing Arrest Peptide
- Employs **extreme compaction and secondary structure formation** (two distinct alpha-helices) inside the tunnel
- The tunnel **drives** VemP's helical folding — it doesn't just passively accommodate it
- VemP's helices directly contact PTC nucleotides, distorting catalytic geometry

### Engineered Arrest Peptides (2025)
- An engineered ribosomal arrest peptide (eRAP) was created combining features from TnaC (tryptophan-sensing) and ErmCL (erythromycin-sensing)
- eRAP shows **enhanced stalling efficiency** compared to natural arrest peptides
- Cryo-EM revealed trigger factor can **promote helical formation** of eRAP inside the tunnel
- eRAP makes extensive contacts with rRNA nucleotides resembling both TRP- and ERY-mediated stalling

> **Key paper**: Sriramoju MK, Ko TP, Draczkowski P et al. "Structural basis of enhanced stalling efficiency of an engineered ribosome arrest peptide." *Nucleic Acids Res* 53(19):gkaf978, 2025. PMID: 41099701.

### Key Tunnel Residues in Arrest
- A2058, A2059 (23S rRNA): macrolide binding site; mutations confer resistance
- A2062: "sensor" nucleotide that changes conformation in response to tunnel contents
- U2585, A2602: PTC residues distorted by arrest peptides
- uL4 K63, uL22 R90 region: constriction-site contacts with arrest peptides

---

## 7. THE VESTIBULE: WHERE FOLDING BEGINS

### Definition and Dimensions
- The vestibule is the **widened, funnel-shaped region** of the last ~20-30 A of the tunnel before the exit port
- Width expands from ~15 A to ~20 A
- Lined by uL23, uL24, and (in eukaryotes) eL39
- **First location where tertiary folding is physically possible** due to sufficient space

### What Folds in the Vestibule
- **Alpha-helices**: Can form throughout the tunnel, but are most stable in the vestibule
- **Small tertiary motifs** (<100 aa): Zinc fingers, small helical bundles can fold here
- **Transmembrane hairpins**: Form in the vestibule before membrane insertion
- Larger domains must emerge further before folding

### Folding Thermodynamics in the Vestibule
- The driving force for co-translational folding is **weaker** in the vestibule than in bulk solution due to **greater water ordering**
- The ribosome defines a **unique energy landscape** — cotranslational folding intermediates form that do **not exist** during refolding in solution
- The vectorial nature of folding (N-to-C) favors **local interactions**, creating intermediates impossible in equilibrium refolding

### Two Folding-Dependent Pathways Through the Vestibule (2025)
- Nascent chains take **two distinct pathways** through the vestibule depending on their folding state
- **Unfolded chains**: Follow a pathway shaped by specific rRNA helices
- **Folded/compact chains**: Redirected through an alternative route
- The ribosome **dynamically modulates** which path the nascent chain takes based on its conformational state
- This affects accessibility to chaperones and other cotranslational factors

> **Key paper**: Cassaignau AME et al. "The ribosome directs nascent chains through two folding-dependent pathways." *bioRxiv*, April 2025. (Integrated cryo-EM + NMR + MD simulations on FLN5-6 immunoglobulin domains)

### The Vestibule as a Decision Point
The vestibule is where four fates are decided:
1. **Fold** — if the domain is small enough and has emerged sufficiently
2. **Chaperone capture** — TF/NAC/RAC-Ssb bind and protect
3. **SRP targeting** — signal sequences recognized for ER/membrane targeting
4. **Stay unfolded** — for translocation-competent proteins

---

## 8. DIFFERENCES ACROSS ORGANISMS

### Bacteria vs Eukaryotes vs Archaea

| Feature | Bacteria | Eukaryotes | Archaea |
|---------|----------|------------|---------|
| **Tunnel volume** | ~3.85 x 10^4 A^3 | ~2.78 x 10^4 A^3 | Intermediate |
| **Tunnel length** | ~80-100 A | ~80-100 A | ~80-100 A |
| **Constriction site width** | ~10 A (second site ~4 A) | ~10 A (second site ~5 A) | Similar to eukaryotes |
| **eL39 protein** | Absent | Present (vestibule) | Present |
| **uL23 tunnel loop** | Present, extended | Shorter (replaced by eL39) | Variable |
| **Primary chaperone** | Trigger factor | NAC + RAC/Ssb | NAC homologs |
| **Folding/knotting effect** | Narrower bacterial constriction enhances folding & knotting | Wider second constriction; trapping/arrest more prone | Similar to eukaryotes |

### Surprising Protist Diversity (2025)
- Analysis of **762 ribosome structures** revealed that **six eukaryotic protist species** have tunnel geometries **remarkably similar to prokaryotes/archaea**
- Four specific sequence modifications in ribosomal proteins and rRNAs account for these variations
- These modifications were detected in additional protist species lacking 3D structural data
- This challenges the simple bacteria-vs-eukaryote tunnel dichotomy

> **Key paper**: Dao Duc K et al. "Detection of archaeal- and prokaryotic-like ribosome exit tunnels within eukaryotic kingdoms." *bioRxiv*, April 2025.

### Functional Consequences
- The narrower bacterial tunnel **enhances** co-translational folding and knotting of small proteins
- Eukaryotic tunnels are more prone to **trapping and arrest** events
- Bacterial tunnels permit **faster escape** of small nascent proteins
- These geometric differences may explain why bacteria and eukaryotes use different cotranslational chaperone systems

> **Key paper**: Yu S, Srebnik S, Dao Duc K. "Geometric differences in the ribosome exit tunnel impact the escape of small nascent proteins." *Biophys J* 122:72-83, 2023. PMID: 36463403.

---

## 9. RECENT CRYO-EM STRUCTURES OF FOLDING INTERMEDIATES

### Folding Intermediates Inside/Near the Tunnel

**1. Structures of co-translational folding intermediates (2025)**
- Structurally characterized two folding intermediates (I1 and I2) of an immunoglobulin-like domain on the ribosome
- Used 19F NMR + paramagnetic relaxation enhancement + protein engineering + MD simulations
- **Key finding**: Highly stable folding intermediates exist during translation that are **absent or only transiently populated off the ribosome**
- These intermediates persist well beyond complete domain emergence from the tunnel

> Waudby CA et al. "Structures of protein folding intermediates on the ribosome." *bioRxiv*, April 2025.

**2. Ribosome-nascent globin complex (2024)**
- Single-particle cryo-EM of ribosome-bound apomyoglobin at multiple chain lengths
- Within the tunnel core: interactions similar to previous reports
- In the **vestibule**: nascent chain becomes more dynamic; interacts with rRNA and uL23
- Combined chemical cross-linking, cryo-EM, and fluorescence anisotropy

> Masse MM, Hutchinson RB, Morgan CE et al. "Mapping Protein-Protein Interactions at Birth: Single-Particle Cryo-EM Analysis of a Ribosome-Nascent Globin Complex." *ACS Cent Sci* 10:508-519, 2024. PMID: 38435509.

**3. Nascent chain surface interactions (2024)**
- Foldable protein sequences interact with **specific ribosomal surface sites** near the exit tunnel
- These interactions differ from those of intrinsically disordered nascent chains
- The ribosome surface near the exit actively participates in early folding events

> Masse MM, Guzman-Luna V, Varela AE et al. "Nascent chains derived from a foldable protein sequence interact with specific ribosomal surface sites near the exit tunnel." *Sci Rep* 14:12328, 2024. PMID: 38811604.

**4. Multi-domain protein folding (Nature Chemistry, 2022)**
- The ribosome **stabilizes partially folded intermediates** of a nascent multi-domain protein
- Partially folded states that would be unstable in solution are **kinetically trapped** by the ribosome
- This allows sequential, domain-by-domain folding

**5. Cotranslational folding through non-native intermediates (2025)**
- Early intermediates stabilized through **non-native hydrophobic interactions** before rearranging
- Disrupting non-native interactions destabilizes intermediates and impairs folding
- Trigger factor alters the pathway by keeping nascent peptide dynamic until full domain emergence
- **Surface-exposed residues** play an unexpected role in on-ribosome folding

> Wang Z, Bitran ER et al. "Cotranslational protein folding through non-native structural intermediates." *Science Advances*, 2025.

---

## 10. COMPUTATIONAL MODELS OF TUNNEL EFFECTS

### All-Atom Molecular Dynamics

**1. Evolution of the tunnel across the tree of life (2025)**
- All-atom MD simulations on **55 distinct cytoplasmic ribosome structures**
- Mapped steric accessibility through the "eyes" of the nascent chain at five different stages of translation
- Revealed **topological and stage-dependent complexity** invisible to static geometric approaches
- Domain-specific variations in tunnel properties affect nascent chain behavior

> "Evolution of the ribosomal exit tunnel through the eyes of the nascent chain." *bioRxiv*, December 2025.

**2. Constriction dynamics (2026)**
- All-atom MD of complete bacterial ribosome with nascent polypeptides of varying length and composition
- The constriction is a **dynamic, flexible gate** — not a static bottleneck
- Nascent polypeptide presence shifts constriction width by ~2 A
- Sequence composition of the nascent chain affects constriction behavior

> McGrath H et al. *bioRxiv*, March 2026.

**3. Single-residue effects in the tunnel (2024)**
- MD simulations showing that **individual amino acid substitutions** change nascent chain behavior inside the tunnel
- Different residues at the same position alter interactions with tunnel walls, translocation rates, and conformational sampling

> Pardo-Avila F, Kudva R, Levitt M. "Single-residue effects on the behavior of a nascent polypeptide chain inside the ribosome exit tunnel." *bioRxiv*, August 2024. PMID: 39229094.

### Coarse-Grained Models

**4. Advanced CG model for tunnel simulations (2025)**
- Automated pipeline to extract tunnel geometry at high resolution
- Converted into CG bead model that accurately captures tunnel geometry
- Enables simulation of co- and post-translational processes computationally prohibitive with all-atom approaches

> "Advanced coarse-grained model for fast simulation of nascent polypeptide chain dynamics within the ribosome." *Biophys J* / *bioRxiv*, 2025.

**5. General Protein Cotranslational Folding (GPCTF) framework (2025)**
- Modeled ribosomal exit tunnel + translation process
- Extensive MD on three proteins of varying topologies (>8 milliseconds total trajectory)
- **Key findings**:
  - Cotranslational folding produces **more helix-rich structures with fewer non-native interactions** upon tunnel expulsion
  - Subsequent folding follows the same pathway as free folding but with **different pathway ratios**
  - Translation speed modulates the ratio of folding pathways

> "Pathway regulation mechanism by cotranslational protein folding." *Communications Chemistry*, 2025.

**6. Tunnel as chaperonin analog (Wruck et al., 2021)**
- Optical tweezers + single-molecule FRET + CG MD simulations
- Tunnel **accelerates folding and stabilizes the folded state** of small zinc-finger domain ADR1a
- Electrostatic interactions with the tunnel lower the folding free energy barrier
- The tunnel functions analogously to a **chaperonin cage**

> Wruck F, Tian P, Kudva R et al. "The ribosome modulates folding inside the ribosomal exit tunnel." *Commun Biol* 4:523, 2021. PMID: 33953328.

---

## SYNTHESIS: IMPLICATIONS FOR DETERMINISTIC FOLDING

### What the Ribosome Tells Us About Why AlphaFold Is Incomplete

1. **Folding is NOT a single equilibrium process** — it begins inside the tunnel, proceeds through non-native intermediates that only exist during translation, and is modulated by translation speed, chaperone timing, and mechanical forces.

2. **The initial conditions matter** — the N-terminus folds first, in a confined, electrostatically charged environment. This creates folding intermediates with no solution-phase analog. A deterministic model must account for these initial conditions.

3. **The tunnel is an active participant** — it is not a passive conduit. It accelerates folding of some structures (alpha-helices), destabilizes others (tertiary folds near the vestibule), and acts as a dynamic gate at the constriction. The tunnel wall chemistry drives specific backbone hydrogen bonding patterns.

4. **Force is information** — mechanical forces from folding propagate back to the PTC, altering translation kinetics. The system is a feedback loop: folding rate affects translation speed, which affects folding rate.

5. **Sequence encodes more than structure** — the amino acid sequence encodes not just the final fold but also the cotranslational folding pathway, translation kinetics (codon usage), chaperone recruitment timing, and arrest peptide regulatory programs.

6. **The path to the fold is deterministic but vectorial** — unlike equilibrium refolding, cotranslational folding is constrained to proceed N-to-C, one residue at a time, through a specific geometric and electrostatic environment. This dramatically reduces the search space.

### Key Parameters a Deterministic Model Would Need

| Parameter | Value/Range | Source |
|-----------|-------------|--------|
| Tunnel length | 80-100 A | Crystallography/cryo-EM |
| Tunnel width (upper) | ~10 A | Crystallography |
| Tunnel width (constriction) | 10 A nominal, dynamic 3-15 A | MD simulations (2026) |
| Tunnel width (vestibule) | 15-20 A | Crystallography |
| Electrostatic potential (tunnel) | Net negative, local positive at constriction | Joiret et al. 2022 |
| Translation speed | ~4-22 aa/s (bacteria), ~3-6 aa/s (eukaryotes) | Ribosome profiling |
| Folding force transmitted | 10-50 pN | Optical tweezers |
| Alpha-helix formation | Begins at ~10 residues from PTC | Cryo-EM + MD |
| Tertiary folding onset | ~30-40 residues from PTC (vestibule) | Cryo-EM |
| Chaperone engagement | ~40-60 residues (when chain reaches exit) | Selective ribosome profiling |

---

## COMPLETE BIBLIOGRAPHY (2020-2026)

### Reviews
1. Samatova E, Komar AA, Rodnina MV. "How the ribosome shapes cotranslational protein folding." *Curr Opin Struct Biol* 84:102740, 2024. PMID: 38071940.
2. Kolar MH, McGrath H, Nepomuceno G, Cernekova M. "Three stages of nascent protein translocation through the ribosome exit tunnel." *WIREs RNA* 15:e1873, 2024. PMID: 39496527.
3. Lentzsch AM, Lee JH, Shan SO. "Mechanistic insights into protein biogenesis and maturation on the ribosome." *J Mol Biol* 436:168815, 2025. PMID: 40024436.
4. Gamerdinger M, Burg N, Deuerling E. "Ribosome-NAC collaboration: A regulatory platform for cotranslational chaperones, enzymes, and targeting factors." *Mol Cell* 86:337-351, 2026. PMID: 41605218.
5. Waudby CA, Burridge C, Cabrita LD. "Thermodynamics of co-translational folding and ribosome-nascent chain interactions." *Curr Opin Struct Biol* 74:102357, 2022. PMID: 35390638.
6. Koubek J, Schmitt J, Galmozzi CV et al. "Mechanisms of cotranslational protein maturation in bacteria." *Front Mol Biosci* 8:689755, 2021. PMID: 34113653.

### Tunnel Structure, Electrostatics, and Geometry
7. Joiret M, Kerff F, Rapino F et al. "Ribosome exit tunnel electrostatics." *Phys Rev E* 105:014409, 2022. PMID: 35193250.
8. Yu S, Srebnik S, Dao Duc K. "Geometric differences in the ribosome exit tunnel impact the escape of small nascent proteins." *Biophys J* 122:72-83, 2023. PMID: 36463403.
9. Micic J, Rodriguez-Galan O, Babiano R et al. "Ribosomal protein eL39 is important for maturation of the nascent polypeptide exit tunnel and proper protein folding during translation." *Nucleic Acids Res* 50:6453-6473, 2022. PMID: 35639884.
10. McGrath H, Cernekova M, Kolar MH. "Early nascent polypeptide dynamics are coupled to the flexibility of the ribosomal tunnel constriction." *bioRxiv*, March 2026.

### Cross-Domain Tunnel Comparisons
11. Dao Duc K et al. "Detection of archaeal- and prokaryotic-like ribosome exit tunnels within eukaryotic kingdoms." *bioRxiv*, April 2025.
12. Dao Duc K et al. "Evolution of the ribosomal exit tunnel through the eyes of the nascent chain." *bioRxiv*, December 2025.

### Cotranslational Folding — Structural Studies
13. Masse MM, Hutchinson RB, Morgan CE et al. "Mapping Protein-Protein Interactions at Birth: Single-Particle Cryo-EM Analysis of a Ribosome-Nascent Globin Complex." *ACS Cent Sci* 10:508-519, 2024. PMID: 38435509.
14. Masse MM, Guzman-Luna V, Varela AE et al. "Nascent chains derived from a foldable protein sequence interact with specific ribosomal surface sites near the exit tunnel." *Sci Rep* 14:12328, 2024. PMID: 38811604.
15. Cassaignau AME et al. "The ribosome directs nascent chains through two folding-dependent pathways." *bioRxiv*, April 2025.
16. Waudby CA et al. "Structures of protein folding intermediates on the ribosome." *bioRxiv*, April 2025.
17. Wang Z, Bitran ER et al. "Cotranslational protein folding through non-native structural intermediates." *Science Advances*, 2025.

### Cotranslational Folding — Biophysics
18. Wruck F, Tian P, Kudva R et al. "The ribosome modulates folding inside the ribosomal exit tunnel." *Commun Biol* 4:523, 2021. PMID: 33953328.
19. Ahn M, Wlodarski T, Mitropoulou A et al. "Modulating co-translational protein folding by rational design and ribosome engineering." *Nat Commun* 13:4191, 2022. PMID: 35869078.
20. Vu QV, Jiang Y, Li MS. "The driving force for co-translational protein folding is weaker in the ribosome vestibule due to greater water ordering." *Chem Sci* 12:12927, 2021. PMID: 34659725.

### Chaperones
21. Lee JH, Rabl L, Gamerdinger M et al. "NAC controls nascent chain fate through tunnel sensing and chaperone action." *Nature*, 2025. PMID: 41430436.
22. Roeselova A, Maslen SL, Shivakumaraswamy S et al. "Mechanism of chaperone coordination during cotranslational protein folding in bacteria." *Mol Cell* 84:2597-2611, 2024. PMID: 38908370.
23. Wales TE, Pajak A, Roeselova A et al. "Resolving chaperone-assisted protein folding on the ribosome at the peptide level." *Nat Struct Mol Biol* 31:1898-1907, 2024. PMID: 38987455.
24. Till MS et al. "Trigger factor accelerates nascent chain compaction and folding." *PNAS* 122(30), 2025. PMID: 40711920.
25. Zhang Y, Grundmann L, Vollmar L et al. "The cotranslational cycle of the ribosome-bound Hsp70 homolog Ssb." *Nat Commun* 17:886, 2026. PMID: 41545346.
26. Ziegelhoffer T, Verma AK, Delewski W et al. "NAC and Zuotin/Hsp70 chaperone systems coexist at the ribosome tunnel exit in vivo." *Nucleic Acids Res* 52:3346-3358, 2024. PMID: 38224454.
27. Chen Y, Tsai B, Li N et al. "Structural remodeling of ribosome associated Hsp40-Hsp70 chaperones during co-translational folding." *Nat Commun* 13:3410, 2022. PMID: 35701497.
28. Deckert A, Cassaignau AME, Wang X et al. "Common sequence motifs of nascent chains engage the ribosome surface and trigger factor." *PNAS* 118:e2103015118, 2021. PMID: 34930833.
29. Nunez E, Saha P, Ibarluzea MG et al. "Multivalent interactions between chaperone and ribosome-nascent chain complex revealed by high-speed AFM and MD simulations." *ACS Nano*, 2025. PMID: 41380087.
30. Lee K, Ziegelhoffer T, Delewski W et al. "Pathway of Hsp70 interactions at the ribosome." *Nat Commun* 12:5550, 2021. PMID: 34580293.

### Arrest Peptides and Stalling
31. Gersteuer F et al. "The SecM arrest peptide traps a pre-peptide bond formation state of the ribosome." *Nat Commun* 15:2431, 2024. PMID: 38503753.
32. Sriramoju MK, Ko TP, Draczkowski P et al. "Structural basis of enhanced stalling efficiency of an engineered ribosome arrest peptide." *Nucleic Acids Res* 53(19):gkaf978, 2025. PMID: 41099701.
33. Su T, Kudva R, Becker T et al. "Structural basis of L-tryptophan-dependent inhibition of release factor 2 by the TnaC arrest peptide." *Nucleic Acids Res* 49:9539-9547, 2021. PMID: 34403461.
34. Ando Y, Kobo A, Niwa T et al. "A mini-hairpin shaped nascent peptide blocks translation termination by a distinct mechanism." *Nat Commun* 16:2249, 2025. PMID: 40057501.
35. Judd HNG, Martinez AK, Klepacki D et al. "Functional domains of a ribosome arresting peptide are affected by surrounding nonconserved residues." *J Biol Chem* 300:105655, 2024. PMID: 38395310.

### Computational Models
36. "Advanced coarse-grained model for fast simulation of nascent polypeptide chain dynamics within the ribosome." *Biophys J* / *bioRxiv*, 2025.
37. "Pathway regulation mechanism by cotranslational protein folding." *Communications Chemistry*, 2025.
38. Pardo-Avila F, Kudva R, Levitt M. "Single-residue effects on the behavior of a nascent polypeptide chain inside the ribosome exit tunnel." *bioRxiv*, August 2024.
39. "Ribosome tunnel environment drives the formation of alpha-helix during cotranslational folding." *J Chem Inf Model*, 2024.
40. Requiao RD, Barros GC, Domitrovic T. "Influence of nascent polypeptide positive charges on translation dynamics." *Biochem J* 477:2921-2934, 2020. PMID: 32797214.

### Force and Quality Control
41. "Mechanical forces regulate the composition and fate of stalled nascent chains." *Mol Cell*, 2025.
42. Venezian J, Bar-Yosef H, Ben-Arie Zilberman H et al. "Diverging co-translational protein complex assembly pathways are governed by interface energy distribution." *Nat Commun* 15:2619, 2024. PMID: 38528060.
