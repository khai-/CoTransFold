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

    Args:
        uniprot_id: UniProt accession (e.g., 'P0AES4')
        force: re-download even if cached

    Returns:
        Path to the downloaded PDB file
    """
    _ensure_cache_dir()
    uniprot_id = uniprot_id.upper()
    filepath = CACHE_DIR / 'alphafold' / f'AF-{uniprot_id}-F1-model_v4.pdb'

    if filepath.exists() and not force:
        return str(filepath)

    url = (f'https://alphafold.ebi.ac.uk/files/'
           f'AF-{uniprot_id}-F1-model_v4.pdb')
    print(f"Downloading AlphaFold prediction for {uniprot_id}...")
    try:
        urllib.request.urlretrieve(url, filepath)
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
