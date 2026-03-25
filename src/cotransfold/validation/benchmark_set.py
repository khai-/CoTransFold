"""Curated benchmark set for rigorous validation.

Each benchmark protein has:
- Experimental structure (high-resolution PDB)
- AlphaFold prediction (from AF DB via UniProt ID)
- Category and metadata for stratified analysis

Categories:
1. Ultra-small (10-30 res): miniproteins
2. Small single-domain (30-80 res): classic folders
3. Medium single-domain (80-150 res): enzymes, globins
4. All-alpha: helical bundles
5. All-beta: sheets, barrels
6. Mixed alpha/beta: TIM barrels, Rossmann folds
7. Co-translational: proteins with known on-ribosome folding data
8. Fold-switching: proteins with multiple folds
9. Mutation-sensitive: known destabilizing mutations
10. Multi-domain: domain docking order matters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkEntry:
    """A single protein in the benchmark set."""
    name: str
    pdb_id: str
    chain_id: str
    uniprot_id: str
    sequence: str
    category: str
    n_residues: int
    resolution: float          # Å
    organism: str
    fold_class: str            # SCOP/CATH classification
    description: str
    expected_ss: str = ''      # Expected secondary structure if known
    mutations: list[str] = field(default_factory=list)  # Known folding mutations
    cotranslational_data: bool = False  # Has on-ribosome folding data?
    alphafold_plddt: float = 0.0       # Average AlphaFold pLDDT if known


# =============================================================================
# BENCHMARK SET — Curated high-confidence proteins
# =============================================================================

BENCHMARK_SET: list[BenchmarkEntry] = [

    # --- Category 1: Ultra-small (10-30 residues) ---

    BenchmarkEntry(
        name="Chignolin",
        pdb_id="1UAO", chain_id="A", uniprot_id="",
        sequence="GYDPETGTWG",
        category="ultra_small", n_residues=10, resolution=1.10,
        organism="Designed", fold_class="beta-hairpin",
        description="10-residue designed miniprotein, beta-hairpin, folds in 0.6μs",
    ),
    BenchmarkEntry(
        name="Trp-cage",
        pdb_id="1L2Y", chain_id="A", uniprot_id="",
        sequence="NLYIQWLKDGGPSSGRPPPS",
        category="ultra_small", n_residues=20, resolution=0.0,  # NMR
        organism="Designed", fold_class="alpha+coil",
        description="20-residue miniprotein, N-terminal helix, folds in 4μs",
    ),
    BenchmarkEntry(
        name="Villin headpiece HP35",
        pdb_id="1YRF", chain_id="A", uniprot_id="",
        sequence="LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        category="ultra_small", n_residues=35, resolution=1.07,
        organism="Gallus gallus", fold_class="three-helix bundle",
        description="35-residue three-helix bundle, folds in 0.7μs, classic benchmark",
    ),

    # --- Category 2: Small single-domain (30-80 residues) ---

    BenchmarkEntry(
        name="Ubiquitin",
        pdb_id="1UBQ", chain_id="A", uniprot_id="P0CG48",
        sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        category="small_domain", n_residues=76, resolution=1.80,
        organism="Homo sapiens", fold_class="alpha/beta (ubiquitin-like)",
        description="76-residue ubiquitin fold, extremely well-studied, alpha+beta",
        mutations=["V26A (destabilizing)", "I30V (mild)", "L67A (destabilizing)"],
    ),
    BenchmarkEntry(
        name="Protein G B1",
        pdb_id="1PGA", chain_id="A", uniprot_id="P06654",
        sequence="MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        category="small_domain", n_residues=56, resolution=1.92,
        organism="Streptococcus sp.", fold_class="alpha+beta",
        description="56-residue protein G B1 domain, alpha-helix + 4-strand beta-sheet",
        mutations=["L5V (faster folding)", "T16A", "T18A"],
    ),
    BenchmarkEntry(
        name="SH3 domain (Fyn)",
        pdb_id="1SHF", chain_id="A", uniprot_id="P06241",
        sequence="EEAIHKEAIGRQFGQNKKELPQCEFCDVEIIALVKDYAIKDDQEMSEDEWVQAINKGYKVRSGKL",
        category="small_domain", n_residues=65, resolution=1.80,
        organism="Homo sapiens", fold_class="all-beta (SH3)",
        description="SH3 domain, all-beta fold, well-characterized folding kinetics",
    ),
    BenchmarkEntry(
        name="WW domain (Pin1)",
        pdb_id="1PIN", chain_id="A", uniprot_id="Q13526",
        sequence="KLPPGWEKRMSRDGRVYYFNHITGTTQFERPSG",
        category="small_domain", n_residues=33, resolution=1.35,
        organism="Homo sapiens", fold_class="all-beta (WW)",
        description="34-residue WW domain, three-stranded beta-sheet, folds in 13μs",
    ),

    # --- Category 3: Medium single-domain (80-150 residues) ---

    BenchmarkEntry(
        name="Lysozyme (hen egg)",
        pdb_id="1AKI", chain_id="A", uniprot_id="P00698",
        sequence="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        category="medium_domain", n_residues=129, resolution=1.50,
        organism="Gallus gallus", fold_class="alpha+beta",
        description="129-residue lysozyme, alpha+beta, extremely well-characterized",
        mutations=["W62G (destabilizing)", "D52N (catalytic)", "I55V"],
    ),
    BenchmarkEntry(
        name="Barnase",
        pdb_id="1A2P", chain_id="A", uniprot_id="P00648",
        sequence="AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSEGFQINHSQFIELDGFPRTIPQADAMKEAGINVD",
        category="medium_domain", n_residues=114, resolution=1.50,
        organism="Bacillus amyloliquefaciens", fold_class="alpha+beta",
        description="110-residue RNase, gold standard for folding studies",
    ),
    BenchmarkEntry(
        name="Myoglobin",
        pdb_id="1MBN", chain_id="A", uniprot_id="P02185",
        sequence="VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        category="medium_domain", n_residues=153, resolution=1.40,
        organism="Physeter macrocephalus", fold_class="all-alpha (globin)",
        description="153-residue myoglobin, all-alpha globin fold, 8 helices",
    ),

    # --- Category 4: All-alpha ---

    BenchmarkEntry(
        name="Four-helix bundle (ROP)",
        pdb_id="1ROP", chain_id="A", uniprot_id="P03051",
        sequence="MTKQEKTALNMARFIRSQTLTLLEKLNELDADEQADICESLHDHADELYRSCLARFGDDGENL",
        category="all_alpha", n_residues=63, resolution=1.70,
        organism="E. coli plasmid", fold_class="all-alpha (four-helix bundle)",
        description="63-residue repressor of primer, four-helix bundle homodimer",
    ),
    BenchmarkEntry(
        name="Calmodulin (N-domain)",
        pdb_id="1CLL", chain_id="A", uniprot_id="P0DP23",
        sequence="ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK",
        category="all_alpha", n_residues=148, resolution=1.70,
        organism="Homo sapiens", fold_class="all-alpha (EF-hand)",
        description="148-residue calmodulin, all-alpha EF-hand domain pair",
    ),

    # --- Category 5: All-beta ---

    BenchmarkEntry(
        name="Immunoglobulin FLN5",
        pdb_id="1QFH", chain_id="A", uniprot_id="P21333",
        sequence="KAGPGEGSVTEKAVQPTFDALAGDTISPKLTVSAGDTSISGHTFEVRVTREDSVPGQE",
        category="all_beta", n_residues=58, resolution=1.90,
        organism="Homo sapiens", fold_class="all-beta (immunoglobulin)",
        description="FLN5 Ig domain from filamin A, has cryo-EM on-ribosome intermediates",
        cotranslational_data=True,
    ),
    BenchmarkEntry(
        name="Tenascin FnIII",
        pdb_id="1TEN", chain_id="A", uniprot_id="P24821",
        sequence="RLDAPSQIEVKDVTDTTALITWFKPLAEIDGIELTYGIKDVPGDRTTIDLTEDENQYSIGNLKPDTEYEVSLISRRGDMSSNPAKETFTT",
        category="all_beta", n_residues=90, resolution=1.80,
        organism="Homo sapiens", fold_class="all-beta (FnIII)",
        description="89-residue fibronectin type III domain, beta-sandwich",
    ),

    # --- Category 6: Mixed alpha/beta ---

    BenchmarkEntry(
        name="DHFR (E. coli)",
        pdb_id="1RX2", chain_id="A", uniprot_id="P0ABQ4",
        sequence="MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR",
        category="mixed_alpha_beta", n_residues=159, resolution=1.09,
        organism="E. coli", fold_class="alpha/beta (Rossmann-like)",
        description="159-residue DHFR, Rossmann fold, has co-translational folding data",
        cotranslational_data=True,
    ),
    BenchmarkEntry(
        name="Thioredoxin",
        pdb_id="2TRX", chain_id="A", uniprot_id="P0AA25",
        sequence="SDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
        category="mixed_alpha_beta", n_residues=108, resolution=1.68,
        organism="E. coli", fold_class="alpha/beta (thioredoxin fold)",
        description="108-residue thioredoxin, classic alpha/beta fold",
    ),

    # --- Category 7: Co-translational folding data ---

    BenchmarkEntry(
        name="GFP (superfolder)",
        pdb_id="2B3P", chain_id="A", uniprot_id="P42212",
        sequence="MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        category="cotranslational", n_residues=238, resolution=1.45,
        organism="Aequorea victoria", fold_class="all-beta (beta-barrel)",
        description="GFP beta-barrel, requires co-translational folding for chromophore formation",
        cotranslational_data=True,
    ),

    # --- Category 8: Fold-switching ---

    BenchmarkEntry(
        name="RfaH (NTD)",
        pdb_id="2OUG", chain_id="A", uniprot_id="P0AFS3",
        sequence="MKNIRNFSIIAHIESDKSFEGFIDKVTFEPEKPSYELSNKALTAQNFVSAEQLRAQIAAEKFGTDMKLAGVLTGSTSRKITELH",
        category="fold_switching", n_residues=84, resolution=2.10,
        organism="E. coli", fold_class="alpha-helix ↔ beta-sheet switch",
        description="RfaH CTD switches from all-alpha to all-beta. AlphaFold fails on this.",
    ),
    BenchmarkEntry(
        name="KaiB (fold-switch)",
        pdb_id="2QKE", chain_id="A", uniprot_id="Q79PF5",
        sequence="MDLNEFERLISTLQSSADRFMAEIIGPQGLVKTDHILLHFIRHGEMKLILERDAKTLTIPDSAIRSWLQMPGFANVELASINPVDAILFSQHVESVKSNETFASNLIDELSKNPSGADG",
        category="fold_switching", n_residues=119, resolution=1.80,
        organism="Synechococcus elongatus", fold_class="fold-switch (thioredoxin ↔ 4HB)",
        description="KaiB switches from thioredoxin fold to four-helix bundle during circadian cycle",
    ),

    # --- Category 9: AlphaFold weaknesses (high-res PDB, low AF confidence) ---

    BenchmarkEntry(
        name="CLIC1 (metamorphic)",
        pdb_id="1K0M", chain_id="A", uniprot_id="O00299",
        sequence="MAEEQPQVELFVKAGSDGAKIGNCPFSQRLFMVLWLKGVTFNVTTVDTKRRTETVQKLCPGGELFLHQNKEKNAEELAAFNKGETSLKKLNMSNKNADKNQAISIAKNWDAGNRWVKSKKVGSKYLFQALGQHFVDYLRDHDINLRGNEYDKNQHEVVAKYNSDYQLKIWDKLNGELFQKKEELDKICPYDHEIHFLDISPLTYNFEANMKAMFNQMEELGS",
        category="af_weakness", n_residues=222, resolution=1.40,
        organism="Homo sapiens", fold_class="metamorphic (GST-like ↔ TM channel)",
        description="CLIC1 exists as soluble GST-fold AND transmembrane channel. AF predicts only soluble form. 1.4Å resolution.",
        alphafold_plddt=82.0,
    ),
    BenchmarkEntry(
        name="SARS-CoV-2 Orf8",
        pdb_id="7JTL", chain_id="A", uniprot_id="P0DTC8",
        sequence="MKFLVFLGIITTVAAFHQECSLQSCTQHQPYVVDDPCPIHFYSKWYIRVGARKSAPLIELCVDEAGSKSPIQYIDIGNYTVSCLPFTINCQEPKLGSLVVRCSFNVNLRATQLFLNPNVTWPKVQDVILNQQSRQRLQPRAADTKKTKTSLNLHSFLRGRDHLVDGSMDKFVVTSAYFTITDEKIGNIISKVQHAANTEIVALTSKLKDAIFNAQVGLIELKQLEK",
        category="af_weakness", n_residues=226, resolution=2.04,
        organism="SARS-CoV-2", fold_class="novel Ig-like fold",
        description="Novel Ig-like fold with unique disulfide patterns. AF pLDDT ~55-65. Post-training novel structure.",
        alphafold_plddt=60.0,
    ),
    BenchmarkEntry(
        name="Mad2 (open form)",
        pdb_id="1DUJ", chain_id="A", uniprot_id="Q13257",
        sequence="MKTYIPEDFLLEGVSSQMLDNGNNQSVGSQRPGTKPLNASSGTNVTPTQFTSYSTMALPDSTPTYGLHFATTDNQRYFTDHQSFVMSWTPVDSMNRGKISQRSFLIGALEHEDHLKALDKSSLEVTLKQLKLEETDFPRKCLFSIFRTDEEQEKYKDTSFLELPKGVRKYLSFHCPVIAQEEIRDQYFTSYNIIDGVPCPHDGLAEELTRALTVALETLMTIYQDIDEEDLQRAFEELAALC",
        category="af_weakness", n_residues=242, resolution=2.05,
        organism="Homo sapiens", fold_class="topological switch (open ↔ closed)",
        description="Mad2 switches between topologically distinct folds. AF predicts closed form; open form has different beta-strand topology.",
        alphafold_plddt=75.0,
    ),
    BenchmarkEntry(
        name="Serpin (alpha-1-antitrypsin)",
        pdb_id="1QLP", chain_id="A", uniprot_id="P01009",
        sequence="EDPQGDAAQKTDTSHHDQDHPTFNKITPNLAEFAFSLYRQLAHQSNSTNIFFSPVSIATAFAMLSLGTKADTHDEILEGLNFNLTEIPEAQIHEGFQELLRTLNQPDSQLQLTTGNGLFLSEGLKLVDKFLEDVKKLYHSEAFTVNFGDTEEAKKQINDYVEKGTQGKIVDLVKELDRDTVFALVNYIFFKGKWERPFEVKDTEEEDFHVDQVTTVKVPMMKRLGMFNIQHCKKLSSWVLLMKYLGNATAIFFLPDEGKLQHLENELTHDIITKFLENEDRRSASLHLPKLSITGTYDLKSVLGQLGITKVFSNGADLSGVTEEAPLKLSKAVHKAVLTIDEKGTEAAGAMFLEAIPMSIPPEVKFNKPFVFLMIEQNTKSPLFMGKVVNPTQK",
        category="af_weakness", n_residues=394, resolution=2.00,
        organism="Homo sapiens", fold_class="metastable (serpin fold)",
        description="Metastable serpin. AF predicts native form but misses the cleaved/relaxed form. Kinetically trapped during co-translational folding.",
        alphafold_plddt=88.0,
    ),
    BenchmarkEntry(
        name="Prion protein PrP",
        pdb_id="6LNI", chain_id="A", uniprot_id="P04156",
        sequence="MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSDYEDRYYRENMHRYPNQVYYRPMDEYSNQNNFVHDCVNITIKQHTVTTTTKGENFTETDVKMMERVVEQMCITQYERESQAYYQRGS",
        category="af_weakness", n_residues=230, resolution=1.70,
        organism="Homo sapiens", fold_class="IDP + globular (misfolding-prone)",
        description="N-terminal 100 residues: AF pLDDT <40. Structured C-domain: ~75. Cannot predict PrPSc amyloid form. Disease-relevant misfolding.",
        alphafold_plddt=45.0,
    ),
    BenchmarkEntry(
        name="T4 lysozyme",
        pdb_id="2LZM", chain_id="A", uniprot_id="P00720",
        sequence="MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
        category="af_weakness", n_residues=164, resolution=1.70,
        organism="Bacteriophage T4", fold_class="alpha+beta (lysozyme)",
        description="700+ crystal structures showing conformational dynamics. AF pLDDT ~90 but predicts single state. Open/closed hinge motion missed.",
        alphafold_plddt=90.0,
    ),
    BenchmarkEntry(
        name="Top7 (designed)",
        pdb_id="1QYS", chain_id="A", uniprot_id="",
        sequence="DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLE",
        category="af_weakness", n_residues=93, resolution=1.25,
        organism="Designed", fold_class="novel (Rosetta designed)",
        description="De novo designed protein with novel topology. No natural homologs. AF struggles with novel folds not in training set.",
        alphafold_plddt=58.0,
    ),

    # --- Category 10: Multi-domain ---

    BenchmarkEntry(
        name="Adenylate kinase",
        pdb_id="4AKE", chain_id="A", uniprot_id="P69441",
        sequence="MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG",
        category="multi_domain", n_residues=214, resolution=2.00,
        organism="E. coli", fold_class="alpha/beta (P-loop NTPase)",
        description="214-residue adenylate kinase, 3 domains with large-scale conformational change",
        cotranslational_data=True,
    ),
]


def get_by_category(category: str) -> list[BenchmarkEntry]:
    """Get all benchmark entries in a category."""
    return [e for e in BENCHMARK_SET if e.category == category]


def get_by_name(name: str) -> BenchmarkEntry:
    """Look up by name or PDB ID."""
    for e in BENCHMARK_SET:
        if e.name.lower() == name.lower() or e.pdb_id.lower() == name.lower():
            return e
    raise ValueError(f"Unknown benchmark: {name}")


def get_all_categories() -> list[str]:
    """List all categories."""
    return sorted(set(e.category for e in BENCHMARK_SET))


def summary_table() -> str:
    """Print a summary table of the benchmark set."""
    lines = [
        f"{'Name':<25} {'PDB':>4} {'Res':>4} {'Å':>5} {'Category':<20} {'CoTrans':>7}",
        "-" * 75,
    ]
    for e in BENCHMARK_SET:
        ct = "yes" if e.cotranslational_data else ""
        lines.append(
            f"{e.name:<25} {e.pdb_id:>4} {e.n_residues:>4} {e.resolution:>5.2f} "
            f"{e.category:<20} {ct:>7}"
        )
    return '\n'.join(lines)
