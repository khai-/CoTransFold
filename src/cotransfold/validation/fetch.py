"""Fetch structures from RCSB PDB and AlphaFold Database.

Downloads:
1. Experimental structures from RCSB PDB (mmCIF/PDB format)
2. AlphaFold predictions from the AlphaFold Protein Structure Database

All files are cached locally in data/structures/.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# Cache directory
CACHE_DIR = Path(__file__).parents[3] / 'data' / 'structures'


def _ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / 'pdb').mkdir(exist_ok=True)
    (CACHE_DIR / 'alphafold').mkdir(exist_ok=True)


def fetch_pdb(pdb_id: str, force: bool = False) -> str:
    """Download a PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB identifier (e.g., '1L2Y')
        force: re-download even if cached

    Returns:
        Path to the downloaded PDB file
    """
    _ensure_cache_dir()
    pdb_id = pdb_id.upper()
    filepath = CACHE_DIR / 'pdb' / f'{pdb_id}.pdb'

    if filepath.exists() and not force:
        return str(filepath)

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f"Downloading PDB {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {e}")

    return str(filepath)


def fetch_alphafold(uniprot_id: str, force: bool = False) -> str:
    """Download an AlphaFold prediction from the AlphaFold DB.

    Uses the API to discover the correct model version, then downloads.

    Args:
        uniprot_id: UniProt accession (e.g., 'P00698')
        force: re-download even if cached

    Returns:
        Path to the downloaded PDB file
    """
    _ensure_cache_dir()
    uniprot_id = uniprot_id.upper()

    # Check for any cached version
    af_dir = CACHE_DIR / 'alphafold'
    existing = list(af_dir.glob(f'AF-{uniprot_id}-F1-model_v*.pdb'))
    if existing and not force:
        return str(existing[0])

    # Query API to get the correct download URL
    import json
    api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}'
    print(f"Querying AlphaFold DB for {uniprot_id}...")
    try:
        with urllib.request.urlopen(api_url) as resp:
            data = json.loads(resp.read())
        if isinstance(data, list) and data:
            pdb_url = data[0].get('pdbUrl', '')
            version = data[0].get('latestVersion', 4)
            plddt = data[0].get('globalMetricValue', 0)
        else:
            raise RuntimeError("Unexpected API response format")
    except Exception as e:
        raise RuntimeError(f"Failed to query AlphaFold API for {uniprot_id}: {e}")

    if not pdb_url:
        raise RuntimeError(f"No PDB URL found for {uniprot_id}")

    filepath = af_dir / f'AF-{uniprot_id}-F1-model_v{version}.pdb'
    print(f"  Downloading from {pdb_url} (pLDDT={plddt:.1f})...")
    try:
        urllib.request.urlretrieve(pdb_url, filepath)
        print(f"  Saved to {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to download AlphaFold {uniprot_id}: {e}")

    return str(filepath)


def fetch_benchmark_pair(pdb_id: str, chain_id: str,
                         uniprot_id: str) -> tuple[str, str]:
    """Download both experimental and AlphaFold structures.

    Returns:
        (pdb_path, alphafold_path)
    """
    pdb_path = fetch_pdb(pdb_id)
    af_path = fetch_alphafold(uniprot_id)
    return pdb_path, af_path
