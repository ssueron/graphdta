#!/usr/bin/env python3
"""
Improved KLIFS + PDB extraction using UniProt-based matching.
Matches proteins through: Target -> Excel (HGNC) -> KLIFS (UniProt)
"""

import json
import csv
import time
import pandas as pd
import requests

# =========================
# CONFIG
# =========================
TARGET_FILE = "data/chembl_pkis2/proteins_klifs.txt"
EXCEL_MAPPING = "data/human_kinase_pocket_sequences.xlsx"
KLIFS_CSV = "klifs_pocket_sequences_human_wt_from_uniprot.csv"

OUTPUT_KLIFS = "klifs_filtered_sequences.csv"
OUTPUT_SEQRES = "pdb_seqres_sequences.csv"
OUTPUT_ATOM = "pdb_atom_sequences.csv"
OUTPUT_MATCHING = "matching_summary.csv"

SLEEP_BETWEEN_PDB = 0.2  # seconds


def load_target_proteins():
    """Load target protein list."""
    with open(TARGET_FILE, 'r') as f:
        return set(json.load(f).keys())


def load_excel_mapping():
    """Load Excel file with HGNC -> UniProt mapping."""
    df = pd.read_excel(EXCEL_MAPPING)
    return df


def load_klifs_data():
    """Load KLIFS CSV and index by UniProt."""
    klifs_by_uniprot = {}
    all_records = []

    with open(KLIFS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot = row['uniprot_id']
            all_records.append(row)

            if uniprot not in klifs_by_uniprot:
                klifs_by_uniprot[uniprot] = {
                    'kinase_name': row['kinase_name'],
                    'kinase_klifs_id': row['kinase_klifs_id'],
                    'structures': []
                }

            klifs_by_uniprot[uniprot]['structures'].append(row)

    return klifs_by_uniprot, all_records


def match_proteins(target_proteins, excel_df, klifs_by_uniprot):
    """
    Match target proteins to KLIFS data using Excel as bridge.
    Returns: (matched_list, unmatched_list)
    """
    matched = []
    unmatched = []

    for target_name in sorted(target_proteins):
        # Step 1: Find in Excel by HGNC or kinase_name
        excel_match = excel_df[
            (excel_df['HGNC'] == target_name) |
            (excel_df['kinase_name'] == target_name)
        ]

        if excel_match.empty:
            unmatched.append({
                'target_name': target_name,
                'reason': 'Not found in Excel mapping file'
            })
            continue

        # Get Excel info
        row = excel_match.iloc[0]
        uniprot = row['UniProt']
        excel_name = row['kinase_name']
        hgnc = row['HGNC']

        # Step 2: Find in KLIFS by UniProt
        if uniprot not in klifs_by_uniprot:
            unmatched.append({
                'target_name': target_name,
                'hgnc': hgnc,
                'excel_name': excel_name,
                'uniprot': uniprot,
                'reason': 'UniProt not found in KLIFS data'
            })
            continue

        # Successfully matched!
        klifs_info = klifs_by_uniprot[uniprot]
        matched.append({
            'target_name': target_name,
            'hgnc': hgnc,
            'excel_name': excel_name,
            'klifs_name': klifs_info['kinase_name'],
            'uniprot': uniprot,
            'klifs_id': klifs_info['kinase_klifs_id'],
            'num_structures': len(klifs_info['structures'])
        })

    return matched, unmatched


def fetch_pdb_seqres(pdb_id, chain):
    """Fetch SEQRES sequence from PDB (full biological sequence)."""
    try:
        # Try primary API
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{chain}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            sequence = data.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', '')
            if sequence:
                return sequence.replace('\n', '').replace(' ', '')

        # Fallback: FASTA API
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')
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
        print(f"    Error fetching SEQRES for {pdb_id}:{chain}: {e}")
        return None


def fetch_pdb_atom_sequence(pdb_id, chain):
    """Fetch ATOM sequence (observed residues with coordinates)."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return None

        # Parse CA atoms only
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

        # Convert 3-letter to 1-letter codes
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }

        if residues:
            sequence = ''.join([aa_map.get(residues[i], 'X')
                              for i in sorted(residues.keys())])
            return sequence

        return None

    except Exception as e:
        print(f"    Error fetching ATOM for {pdb_id}:{chain}: {e}")
        return None


def main():
    print("="*80)
    print("IMPROVED KLIFS + PDB EXTRACTION")
    print("Using UniProt-based matching through Excel mapping file")
    print("="*80)

    # Step 1: Load all data sources
    print("\n[1/6] Loading data sources...")
    target_proteins = load_target_proteins()
    excel_df = load_excel_mapping()
    klifs_by_uniprot, klifs_all_records = load_klifs_data()

    print(f"  ✓ Target proteins: {len(target_proteins)}")
    print(f"  ✓ Excel mappings: {len(excel_df)}")
    print(f"  ✓ KLIFS UniProts: {len(klifs_by_uniprot)}")
    print(f"  ✓ KLIFS structures: {len(klifs_all_records)}")

    # Step 2: Match proteins
    print("\n[2/6] Matching proteins using UniProt...")
    matched, unmatched = match_proteins(target_proteins, excel_df, klifs_by_uniprot)

    total_structures = sum(m['num_structures'] for m in matched)

    print(f"  ✓ MATCHED: {len(matched)} proteins")
    print(f"  ✗ UNMATCHED: {len(unmatched)} proteins")
    print(f"  → Total structures: {total_structures}")

    # Save matching summary
    df_matching = pd.DataFrame(matched)
    df_matching.to_csv(OUTPUT_MATCHING, index=False)
    print(f"  ✓ Saved matching summary to {OUTPUT_MATCHING}")

    # Step 3: Extract KLIFS structures
    print(f"\n[3/6] Extracting KLIFS data for {len(matched)} proteins...")

    klifs_filtered = []
    for m in matched:
        uniprot = m['uniprot']
        structures = klifs_by_uniprot[uniprot]['structures']

        for struct in structures:
            klifs_filtered.append({
                'target_name': m['target_name'],
                'hgnc': m['hgnc'],
                'klifs_name': m['klifs_name'],
                'uniprot_id': uniprot,
                'kinase_klifs_id': struct['kinase_klifs_id'],
                'structure_id': struct['structure_id'],
                'pdb_id': struct['pdb_id'],
                'chain': struct['chain'],
                'alt_loc': struct.get('alt_loc', ''),
                'resolution': struct.get('resolution', ''),
                'quality_score': struct.get('quality_score', ''),
                'DFG_state': struct.get('DFG_state', ''),
                'alphaC_state': struct.get('alphaC_state', ''),
                'pocket_85aa': struct['pocket_85aa']
            })

    df_klifs = pd.DataFrame(klifs_filtered)
    df_klifs.to_csv(OUTPUT_KLIFS, index=False)
    print(f"  ✓ Extracted {len(df_klifs)} structures")
    print(f"  ✓ Saved to {OUTPUT_KLIFS}")

    # Step 4: Get unique PDB chains
    print(f"\n[4/6] Identifying unique PDB chains...")
    unique_pdbs = df_klifs[['pdb_id', 'chain']].drop_duplicates()
    unique_pdbs = unique_pdbs[unique_pdbs['pdb_id'].notna() & (unique_pdbs['pdb_id'] != '')]
    print(f"  ✓ Found {len(unique_pdbs)} unique PDB chains to fetch")

    # Step 5: Download SEQRES sequences
    print(f"\n[5/6] Downloading SEQRES sequences...")
    seqres_records = []

    for idx, (_, row) in enumerate(unique_pdbs.iterrows(), 1):
        pdb_id = row['pdb_id']
        chain = row['chain']

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(unique_pdbs)}")

        seqres = fetch_pdb_seqres(pdb_id, chain)
        if seqres:
            seqres_records.append({
                'pdb_id': pdb_id,
                'chain': chain,
                'sequence': seqres,
                'length': len(seqres)
            })

        time.sleep(SLEEP_BETWEEN_PDB)

    df_seqres = pd.DataFrame(seqres_records)
    df_seqres.to_csv(OUTPUT_SEQRES, index=False)
    print(f"  ✓ Downloaded {len(df_seqres)}/{len(unique_pdbs)} SEQRES sequences")
    print(f"  ✓ Saved to {OUTPUT_SEQRES}")

    # Step 6: Download ATOM sequences
    print(f"\n[6/6] Downloading ATOM sequences...")
    atom_records = []

    for idx, (_, row) in enumerate(unique_pdbs.iterrows(), 1):
        pdb_id = row['pdb_id']
        chain = row['chain']

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(unique_pdbs)}")

        atom_seq = fetch_pdb_atom_sequence(pdb_id, chain)
        if atom_seq:
            atom_records.append({
                'pdb_id': pdb_id,
                'chain': chain,
                'sequence': atom_seq,
                'length': len(atom_seq)
            })

        time.sleep(SLEEP_BETWEEN_PDB)

    df_atom = pd.DataFrame(atom_records)
    df_atom.to_csv(OUTPUT_ATOM, index=False)
    print(f"  ✓ Downloaded {len(df_atom)}/{len(unique_pdbs)} ATOM sequences")
    print(f"  ✓ Saved to {OUTPUT_ATOM}")

    # Final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nMatching:")
    print(f"  • Target proteins:     {len(target_proteins)}")
    print(f"  • Matched proteins:    {len(matched)} ({len(matched)*100/len(target_proteins):.1f}%)")
    print(f"  • Unmatched proteins:  {len(unmatched)}")

    print(f"\nData extracted:")
    print(f"  • KLIFS structures:    {len(df_klifs)}")
    print(f"  • SEQRES sequences:    {len(df_seqres)}")
    print(f"  • ATOM sequences:      {len(df_atom)}")

    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_MATCHING}")
    print(f"  2. {OUTPUT_KLIFS}")
    print(f"  3. {OUTPUT_SEQRES}")
    print(f"  4. {OUTPUT_ATOM}")

    if unmatched:
        print(f"\nUnmatched proteins (first 20):")
        for u in unmatched[:20]:
            print(f"  - {u['target_name']}: {u['reason']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
