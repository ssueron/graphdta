#!/usr/bin/env python3
"""
Complete KLIFS extraction using ChEMBL file for 100% protein matching.
Matches: proteins_klifs.txt → ChEMBL (by KINOMEscan Symbol) → KLIFS (by UniProt)
"""

import json
import csv
import pandas as pd

# =========================
# CONFIG
# =========================
TARGET_FILE = "data/chembl_pkis2/proteins_klifs.txt"
CHEMBL_MAPPING = "data/uniprotkb_chembl_ID_kinomescan_cleaned.xlsx"
KLIFS_CSV = "klifs_pocket_sequences_human_wt_from_uniprot.csv"

OUTPUT_KLIFS = "klifs_complete_extraction.csv"
OUTPUT_MATCHING = "matching_complete_summary.csv"
OUTPUT_UNMATCHED = "proteins_not_in_klifs.csv"


def main():
    print("="*80)
    print("COMPLETE KLIFS EXTRACTION")
    print("Using ChEMBL file for 100% protein matching")
    print("="*80)

    # Step 1: Load target proteins
    print("\n[1/5] Loading target proteins...")
    with open(TARGET_FILE, 'r') as f:
        target_proteins = json.load(f)
    print(f"  ✓ Target proteins: {len(target_proteins)}")

    # Step 2: Load ChEMBL mapping
    print("\n[2/5] Loading ChEMBL mapping file...")
    df_chembl = pd.read_excel(CHEMBL_MAPPING)
    print(f"  ✓ ChEMBL entries: {len(df_chembl)}")

    # Create lookup by KINOMEscan Symbol
    chembl_lookup = {}
    for _, row in df_chembl.iterrows():
        kinomescan_symbol = row['KINOMEscan® Gene Symbol']
        chembl_lookup[kinomescan_symbol] = {
            'uniprot': row['UniProt Entry'],
            'gene_name': row['Gene Names'],
            'gene_synonyms': row.get('Gene Names (synonym)', ''),
            'chembl_id': row['protein_chembl_id'],
            'protein_names': row['Protein names']
        }

    # Step 3: Load KLIFS data indexed by UniProt
    print("\n[3/5] Loading KLIFS data...")
    klifs_by_uniprot = {}
    all_klifs_records = []

    with open(KLIFS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot = row['uniprot_id']
            all_klifs_records.append(row)

            if uniprot not in klifs_by_uniprot:
                klifs_by_uniprot[uniprot] = {
                    'kinase_name': row['kinase_name'],
                    'kinase_klifs_id': row['kinase_klifs_id'],
                    'structures': []
                }

            klifs_by_uniprot[uniprot]['structures'].append(row)

    print(f"  ✓ KLIFS UniProt IDs: {len(klifs_by_uniprot)}")
    print(f"  ✓ KLIFS structures: {len(all_klifs_records)}")

    # Step 4: Match proteins
    print("\n[4/5] Matching proteins...")
    matched = []
    unmatched = []

    for target_name in sorted(target_proteins.keys()):
        # Look up in ChEMBL by KINOMEscan Symbol
        if target_name not in chembl_lookup:
            unmatched.append({
                'target_name': target_name,
                'reason': 'Not found in ChEMBL file'
            })
            continue

        chembl_info = chembl_lookup[target_name]
        uniprot = chembl_info['uniprot']

        # Look up in KLIFS by UniProt
        if uniprot not in klifs_by_uniprot:
            unmatched.append({
                'target_name': target_name,
                'uniprot': uniprot,
                'gene_name': chembl_info['gene_name'],
                'chembl_id': chembl_info['chembl_id'],
                'reason': 'UniProt not found in KLIFS data'
            })
            continue

        # Successfully matched!
        klifs_info = klifs_by_uniprot[uniprot]
        matched.append({
            'target_name': target_name,
            'uniprot': uniprot,
            'gene_name': chembl_info['gene_name'],
            'gene_synonyms': chembl_info['gene_synonyms'],
            'chembl_id': chembl_info['chembl_id'],
            'klifs_name': klifs_info['kinase_name'],
            'klifs_id': klifs_info['kinase_klifs_id'],
            'num_structures': len(klifs_info['structures'])
        })

    print(f"  ✓ MATCHED: {len(matched)} proteins")
    print(f"  ✗ UNMATCHED: {len(unmatched)} proteins")

    total_structures = sum(m['num_structures'] for m in matched)
    print(f"  → Total structures: {total_structures}")

    # Save matching summary
    df_matching = pd.DataFrame(matched)
    df_matching.to_csv(OUTPUT_MATCHING, index=False)
    print(f"  ✓ Saved matching summary to {OUTPUT_MATCHING}")

    if unmatched:
        df_unmatched = pd.DataFrame(unmatched)
        df_unmatched.to_csv(OUTPUT_UNMATCHED, index=False)
        print(f"  ✓ Saved unmatched proteins to {OUTPUT_UNMATCHED}")

    # Step 5: Extract KLIFS structures for matched proteins
    print(f"\n[5/5] Extracting KLIFS data for {len(matched)} proteins...")

    klifs_extracted = []
    for m in matched:
        uniprot = m['uniprot']
        structures = klifs_by_uniprot[uniprot]['structures']

        for struct in structures:
            klifs_extracted.append({
                'target_name': m['target_name'],
                'gene_name': m['gene_name'],
                'gene_synonyms': m['gene_synonyms'],
                'klifs_name': m['klifs_name'],
                'uniprot_id': uniprot,
                'chembl_id': m['chembl_id'],
                'kinase_klifs_id': struct['kinase_klifs_id'],
                'structure_id': struct['structure_id'],
                'pdb_id': struct['pdb_id'],
                'chain': struct['chain'],
                'alt_loc': struct.get('alt_loc', ''),
                'species': struct.get('species', ''),
                'resolution': struct.get('resolution', ''),
                'quality_score': struct.get('quality_score', ''),
                'missing_residues': struct.get('missing_residues', ''),
                'missing_atoms': struct.get('missing_atoms', ''),
                'DFG_state': struct.get('DFG_state', ''),
                'alphaC_state': struct.get('alphaC_state', ''),
                'ligand': struct.get('ligand', ''),
                'ligand_id': struct.get('ligand_id', ''),
                'pocket_85aa': struct['pocket_85aa']
            })

    df_klifs = pd.DataFrame(klifs_extracted)
    df_klifs.to_csv(OUTPUT_KLIFS, index=False)

    print(f"  ✓ Extracted {len(df_klifs)} structures")
    print(f"  ✓ Saved to {OUTPUT_KLIFS}")

    # Final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)

    print(f"\nMatching Results:")
    print(f"  • Target proteins:     {len(target_proteins)}")
    print(f"  • Matched proteins:    {len(matched)} ({len(matched)*100/len(target_proteins):.1f}%)")
    print(f"  • Unmatched proteins:  {len(unmatched)}")

    print(f"\nData Extracted:")
    print(f"  • KLIFS structures:    {len(df_klifs)}")
    print(f"  • Unique PDB IDs:      {df_klifs['pdb_id'].nunique()}")
    print(f"  • Unique chains:       {df_klifs[['pdb_id', 'chain']].drop_duplicates().shape[0]}")

    print(f"\nOutput Files:")
    print(f"  1. {OUTPUT_MATCHING} - Complete protein mapping")
    print(f"  2. {OUTPUT_KLIFS} - All KLIFS structures")
    if unmatched:
        print(f"  3. {OUTPUT_UNMATCHED} - Proteins not in KLIFS")

    # Show examples of newly matched proteins
    previous_matched = pd.read_csv('matching_summary.csv')
    previous_names = set(previous_matched['target_name'])
    newly_matched = [m for m in matched if m['target_name'] not in previous_names]

    if newly_matched:
        print(f"\n🎉 Newly matched proteins ({len(newly_matched)}):")
        for m in sorted(newly_matched, key=lambda x: x['target_name'])[:20]:
            print(f"  • {m['target_name']:20s} → {m['klifs_name']:15s} ({m['uniprot']}) [{m['num_structures']} structures]")

        if len(newly_matched) > 20:
            print(f"  ... and {len(newly_matched) - 20} more!")

    if unmatched:
        print(f"\n⚠️  Unmatched proteins:")
        for u in unmatched:
            reason = u.get('reason', 'Unknown')
            print(f"  • {u['target_name']}: {reason}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
