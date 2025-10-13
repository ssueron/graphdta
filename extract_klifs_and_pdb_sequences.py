#!/usr/bin/env python3
"""
Extract KLIFS data for specific proteins and download PDB sequences.
- Filters KLIFS data by protein names from proteins_klifs.txt
- Downloads SEQRES (full sequence) and ATOM (observed residues) from PDB
"""

import json
import csv
import time
import os
from pathlib import Path
import requests
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient

# =========================
# CONFIG
# =========================
TARGET_PROTEINS_FILE = "data/chembl_pkis2/proteins_klifs.txt"
OUTPUT_CSV = "klifs_filtered_pocket_sequences.csv"
OUTPUT_SEQRES = "pdb_seqres_sequences.csv"
OUTPUT_ATOM = "pdb_atom_sequences.csv"
ERRORS_CSV = "extraction_errors.csv"
SLEEP_BETWEEN_CALLS = 0.1

# =========================
# KLIFS NAME MAPPING
# =========================
# Common variations between target names and KLIFS names
NAME_MAPPINGS = {
    # Aurora kinases
    'AURKA': 'AurA', 'AURKB': 'AurB', 'AURKC': 'AurC',
    # p38 kinases
    'p38-alpha': 'p38a', 'p38-beta': 'p38b', 'p38-gamma': 'p38g', 'p38-delta': 'p38d',
    # AMPK
    'AMPK-alpha1': 'AMPKa1', 'AMPK-alpha2': 'AMPKa2',
    # MAP kinases
    'ERK1': 'Erk1', 'ERK2': 'Erk2', 'ERK8': 'Erk5',
    # Others
    'ACVR1': 'ALK2', 'ACVR1B': 'ALK4', 'ACVR2A': 'ActR2A', 'ACVRL1': 'ALK1',
    'BMPR1A': 'ALK3', 'BMPR1B': 'BMPR1B', 'BMPR2': 'BMPR2',
    'TGFBR1': 'TGFbR1', 'TGFBR2': 'TGFbR2',
    'CHEK1': 'CHK1', 'CHEK2': 'CHK2',
    'CSNK1A1': 'CK1a', 'CSNK1D': 'CK1d', 'CSNK1E': 'CK1e', 'CSNK1G1': 'CK1g1',
    'CSNK2A1': 'CK2a1', 'CSNK2A2': 'CK2a2',
    'EPHA1': 'EphA1', 'EPHA2': 'EphA2', 'EPHA3': 'EphA3', 'EPHA4': 'EphA4',
    'EPHB1': 'EphB1', 'EPHB2': 'EphB2', 'EPHB3': 'EphB3', 'EPHB4': 'EphB4',
    'FGFR1': 'FGFR1', 'FGFR2': 'FGFR2', 'FGFR3': 'FGFR3', 'FGFR4': 'FGFR4',
    'IKK-alpha': 'IKKa', 'IKK-beta': 'IKKb', 'IKK-epsilon': 'IKKe',
    'INSR': 'IR', 'IGF1R': 'IGF1R', 'INSRR': 'IRR',
    'MEK1': 'MAP2K1', 'MEK2': 'MAP2K2', 'MKK7': 'MAP2K7', 'MEK4': 'MAP2K4',
    'PDGFRA': 'PDGFRa', 'PDGFRB': 'PDGFRb',
    'PDPK1': 'PDK1',
    'PKAC-alpha': 'PKA',
    'ROCK1': 'ROCK1', 'ROCK2': 'ROCK2',
    'S6K1': 'p70S6K',
    'SGK': 'SGK1',
    'VEGFR2': 'VEGFR2', 'FLT1': 'VEGFR1', 'FLT3': 'FLT3', 'FLT4': 'VEGFR3',
}

def load_target_proteins():
    """Load target protein names from JSON file."""
    with open(TARGET_PROTEINS_FILE, 'r') as f:
        proteins_dict = json.load(f)
    return set(proteins_dict.keys())

def create_name_mapping(target_names, klifs_names):
    """Create mapping from target names to KLIFS names."""
    mapping = {}

    # First, apply predefined mappings
    for target, klifs in NAME_MAPPINGS.items():
        if target in target_names and klifs in klifs_names:
            mapping[target] = klifs

    # Then, exact matches
    for name in target_names:
        if name in klifs_names and name not in mapping:
            mapping[name] = name

    # Case-insensitive matches
    klifs_lower = {k.lower(): k for k in klifs_names}
    for target in target_names:
        if target not in mapping:
            if target.lower() in klifs_lower:
                mapping[target] = klifs_lower[target.lower()]

    return mapping

def setup_klifs_client():
    """Setup KLIFS API client."""
    http = RequestsClient()
    client = SwaggerClient.from_url(
        "https://klifs.net/swagger_v2/swagger.json",
        http_client=http,
        config={
            "also_return_response": False,
            "validate_swagger_spec": False,
            "validate_responses": False,
            "validate_requests": False,
            "use_models": False,
        },
    )
    return client

def safe_call(op, **kwargs):
    """Call KLIFS API with error handling."""
    specs = {p["name"]: p for p in op.operation.op_spec.get("parameters", [])}
    fixed = {}
    for k, v in kwargs.items():
        p = specs.get(k)
        ptype = p.get("type") if p else None
        if (not ptype) and p and isinstance(p.get("schema"), dict):
            ptype = p["schema"].get("type")
        if ptype == "array" and not isinstance(v, (list, tuple)):
            v = [v]
        fixed[k] = v
    try:
        out = op(**fixed).result()
        if isinstance(out, list) and out and isinstance(out[0], int) and out[0] >= 400:
            return ("ERROR", out[0], out[1] if len(out) > 1 else "KLIFS error")
        return out
    except Exception as e:
        msg = str(e)
        code = 400 if "unknown kinase id" in msg.lower() else 500
        return ("ERROR", code, msg)

def is_wild_type(client, structure_id):
    """Check if structure has no modified residues (is wild-type)."""
    structures_get_mod_res = client.Structures.get_structure_modified_residues
    out = safe_call(structures_get_mod_res, structure_ID=int(structure_id))
    if isinstance(out, tuple) and out[0] == "ERROR":
        return True  # Keep structure if check fails
    return (out in (None, "", [], ()))

def fetch_pdb_seqres(pdb_id, chain):
    """
    Fetch SEQRES sequence from PDB (full sequence from header).
    Uses RCSB PDB REST API.
    """
    try:
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{chain}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Get the sequence
            sequence = data.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', '')
            return sequence.replace('\n', '').replace(' ', '')

        # Try alternative API
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            # Find sequence for the specific chain
            chain_found = False
            sequence_lines = []
            for line in lines:
                if line.startswith('>'):
                    if f"Chain {chain}" in line or f"_{chain}" in line:
                        chain_found = True
                    else:
                        chain_found = False
                elif chain_found:
                    sequence_lines.append(line.strip())
            if sequence_lines:
                return ''.join(sequence_lines)

        return None
    except Exception as e:
        print(f"  Error fetching SEQRES for {pdb_id}:{chain}: {e}")
        return None

def fetch_pdb_atom_sequence(pdb_id, chain):
    """
    Fetch ATOM sequence from PDB (observed residues only).
    Downloads PDB file and extracts CA atoms.
    """
    try:
        # Download PDB file
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return None

        # Parse PDB file for CA atoms
        residues = {}
        for line in response.text.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    atom_chain = line[21].strip()
                    if atom_chain == chain:
                        res_num = int(line[22:26].strip())
                        res_name = line[17:20].strip()
                        residues[res_num] = res_name
                except:
                    continue

        # Convert 3-letter codes to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }

        if residues:
            sequence = ''.join([aa_map.get(residues[i], 'X') for i in sorted(residues.keys())])
            return sequence

        return None

    except Exception as e:
        print(f"  Error fetching ATOM for {pdb_id}:{chain}: {e}")
        return None

def main():
    print("="*80)
    print("KLIFS + PDB SEQUENCE EXTRACTION")
    print("="*80)

    # Load target proteins
    print("\n1. Loading target proteins...")
    target_names = load_target_proteins()
    print(f"   Target proteins: {len(target_names)}")

    # Load existing KLIFS data to get available names
    print("\n2. Loading KLIFS kinase information...")
    client = setup_klifs_client()
    info_get = client.Information.get_kinase_information

    info_out = safe_call(info_get)
    if isinstance(info_out, tuple) and info_out[0] == "ERROR":
        print(f"   ERROR: Failed to get kinase information: {info_out[2]}")
        return

    import pandas as pd
    info_df = pd.DataFrame(info_out)

    # Get column names
    kid_col = next((c for c in info_df.columns if c in ("kinase.klifs_id","kinase_ID","kinase.id")), None)
    unp_col = next((c for c in info_df.columns if "uniprot" in c.lower()), None)
    species_col = next((c for c in info_df.columns if "species" in c.lower()), None)
    name_col = next((c for c in info_df.columns if c.strip().lower() in ("kinase","kinase_name","name")), None)

    info_df = info_df[[kid_col, name_col, unp_col, species_col]].copy()
    info_df.columns = ["kinase_klifs_id", "kinase_name", "uniprot_id", "species"]

    # Filter for human
    info_human = info_df[info_df["species"].astype(str).str.contains("Human", case=False, na=False)]
    klifs_names = set(info_human["kinase_name"].unique())
    print(f"   KLIFS human kinases: {len(klifs_names)}")

    # Create name mapping
    print("\n3. Mapping target names to KLIFS names...")
    name_map = create_name_mapping(target_names, klifs_names)
    print(f"   Matched proteins: {len(name_map)}")
    print(f"   Unmatched proteins: {len(target_names) - len(name_map)}")

    if len(name_map) < len(target_names):
        unmatched = target_names - set(name_map.keys())
        print(f"\n   Unmatched proteins (first 20):")
        for name in sorted(unmatched)[:20]:
            print(f"     - {name}")

    print(f"\n   Matched proteins (first 20):")
    for target, klifs in sorted(name_map.items())[:20]:
        if target != klifs:
            print(f"     {target:20s} -> {klifs}")
        else:
            print(f"     {target}")

    # Get KLIFS structures for matched proteins
    print(f"\n4. Extracting KLIFS structures for {len(name_map)} proteins...")
    structures_list_op = client.Structures.get_structures_list

    all_records = []
    errors = []

    for i, (target_name, klifs_name) in enumerate(sorted(name_map.items()), 1):
        # Get kinase ID
        kinase_rows = info_human[info_human["kinase_name"] == klifs_name]
        if kinase_rows.empty:
            print(f"   [{i}/{len(name_map)}] SKIP: {target_name} -> {klifs_name} (not found in KLIFS)")
            continue

        kid = int(kinase_rows.iloc[0]["kinase_klifs_id"])
        uniprot_id = kinase_rows.iloc[0]["uniprot_id"]

        # Get structures
        S_out = safe_call(structures_list_op, kinase_ID=kid)
        if isinstance(S_out, tuple) and S_out[0] == "ERROR":
            errors.append((target_name, f"structures_list_failed:{S_out[2]}"))
            print(f"   [{i}/{len(name_map)}] ERROR: {target_name} ({klifs_name})")
            continue

        S = pd.DataFrame(S_out)
        if S.empty:
            print(f"   [{i}/{len(name_map)}] SKIP: {target_name} ({klifs_name}) - no structures")
            continue

        # Normalize columns
        if "structure_ID" not in S.columns:
            for c in ("structure.klifs_id", "klifs_id", "id"):
                if c in S.columns:
                    S = S.rename(columns={c: "structure_ID"})
                    break

        # Filter human and wild-type
        if "species" in S.columns:
            S = S[S["species"].astype(str).str.contains("Human", case=False, na=False)]

        wt_structures = []
        for sid in S["structure_ID"].astype(int):
            if is_wild_type(client, sid):
                wt_structures.append(sid)

        S = S[S["structure_ID"].astype(int).isin(wt_structures)]

        if S.empty:
            print(f"   [{i}/{len(name_map)}] SKIP: {target_name} ({klifs_name}) - no WT human structures")
            continue

        # Store records
        for _, row in S.iterrows():
            all_records.append({
                "target_name": target_name,
                "klifs_name": klifs_name,
                "uniprot_id": uniprot_id,
                "kinase_klifs_id": kid,
                "structure_id": int(row["structure_ID"]),
                "pdb_id": row.get("pdb", ""),
                "chain": row.get("chain", ""),
                "alt_loc": row.get("alt", ""),
                "resolution": row.get("resolution", None),
                "quality_score": row.get("quality_score", None),
                "DFG_state": row.get("DFG", ""),
                "pocket_85aa": row.get("pocket", ""),
            })

        print(f"   [{i}/{len(name_map)}] ✓ {target_name} ({klifs_name}): {len(S)} WT structures")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save KLIFS data
    print(f"\n5. Saving KLIFS data...")
    df_klifs = pd.DataFrame(all_records)
    df_klifs.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved {len(df_klifs)} structures to {OUTPUT_CSV}")

    # Download PDB sequences
    print(f"\n6. Downloading PDB sequences...")
    seqres_records = []
    atom_records = []

    unique_pdbs = df_klifs[['pdb_id', 'chain']].drop_duplicates()
    print(f"   Unique PDB chains to fetch: {len(unique_pdbs)}")

    for idx, (_, row) in enumerate(unique_pdbs.iterrows(), 1):
        pdb_id = row['pdb_id']
        chain = row['chain']

        if not pdb_id or not chain:
            continue

        print(f"   [{idx}/{len(unique_pdbs)}] Fetching {pdb_id}:{chain}...")

        # Fetch SEQRES
        seqres = fetch_pdb_seqres(pdb_id, chain)
        if seqres:
            seqres_records.append({
                'pdb_id': pdb_id,
                'chain': chain,
                'sequence_type': 'SEQRES',
                'sequence': seqres,
                'length': len(seqres)
            })

        # Fetch ATOM
        atom_seq = fetch_pdb_atom_sequence(pdb_id, chain)
        if atom_seq:
            atom_records.append({
                'pdb_id': pdb_id,
                'chain': chain,
                'sequence_type': 'ATOM',
                'sequence': atom_seq,
                'length': len(atom_seq)
            })

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save PDB sequences
    if seqres_records:
        df_seqres = pd.DataFrame(seqres_records)
        df_seqres.to_csv(OUTPUT_SEQRES, index=False)
        print(f"\n   Saved {len(df_seqres)} SEQRES sequences to {OUTPUT_SEQRES}")

    if atom_records:
        df_atom = pd.DataFrame(atom_records)
        df_atom.to_csv(OUTPUT_ATOM, index=False)
        print(f"   Saved {len(df_atom)} ATOM sequences to {OUTPUT_ATOM}")

    # Save errors
    if errors:
        pd.DataFrame(errors, columns=["protein", "error"]).to_csv(ERRORS_CSV, index=False)
        print(f"\n   Logged {len(errors)} errors to {ERRORS_CSV}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Target proteins:        {len(target_names)}")
    print(f"Matched proteins:       {len(name_map)}")
    print(f"KLIFS structures:       {len(all_records)}")
    print(f"SEQRES sequences:       {len(seqres_records)}")
    print(f"ATOM sequences:         {len(atom_records)}")
    print("="*80)

if __name__ == "__main__":
    main()
