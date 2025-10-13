#!/usr/bin/env python3
"""
Reconstruct complete protein sequences from KLIFS pocket sequences.
This script analyzes 85aa pocket sequences from different PDB structures
and reconstructs complete sequences by overlaying them.
"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def analyze_sequences(input_file: str):
    """
    Analyze sequences and categorize them into:
    1. Complete sequences (no gaps)
    2. Sequences that can be fully reconstructed
    3. Sequences that can be partially reconstructed
    4. Sequences that cannot be reconstructed
    """

    # Read all sequences grouped by uniprot_id
    sequences_by_protein = defaultdict(list)

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot_id = row['uniprot_id']
            sequences_by_protein[uniprot_id].append({
                'uniprot_id': row['uniprot_id'],
                'kinase_name': row['kinase_name'],
                'kinase_klifs_id': row['kinase_klifs_id'],
                'structure_id': row['structure_id'],
                'pdb_id': row['pdb_id'],
                'pocket_85aa': row['pocket_85aa']
            })

    # Results storage
    complete_sequences = []
    fully_reconstructed = []
    partially_reconstructed = []
    unreconstructable = []

    # Process each protein
    for uniprot_id, entries in sequences_by_protein.items():
        # Get basic info from first entry
        first_entry = entries[0]
        kinase_name = first_entry['kinase_name']
        kinase_klifs_id = first_entry['kinase_klifs_id']

        # Find complete sequences (no gaps)
        complete_seqs = [e for e in entries if '_' not in e['pocket_85aa']]

        if complete_seqs:
            # Use the first complete sequence found
            complete_sequences.append({
                'uniprot_id': uniprot_id,
                'kinase_name': kinase_name,
                'kinase_klifs_id': kinase_klifs_id,
                'sequence_85aa': complete_seqs[0]['pocket_85aa'],
                'source_pdb': complete_seqs[0]['pdb_id'],
                'num_complete_structures': len(complete_seqs)
            })
        else:
            # Try to reconstruct from multiple sequences
            reconstruction_result = reconstruct_sequence(entries)

            if reconstruction_result['status'] == 'fully_reconstructed':
                fully_reconstructed.append({
                    'uniprot_id': uniprot_id,
                    'kinase_name': kinase_name,
                    'kinase_klifs_id': kinase_klifs_id,
                    'sequence_85aa': reconstruction_result['sequence'],
                    'num_structures_used': len(entries),
                    'source_pdbs': reconstruction_result['source_pdbs'],
                    'variations': reconstruction_result['variations']
                })
            elif reconstruction_result['status'] == 'partially_reconstructed':
                partially_reconstructed.append({
                    'uniprot_id': uniprot_id,
                    'kinase_name': kinase_name,
                    'kinase_klifs_id': kinase_klifs_id,
                    'sequence_85aa': reconstruction_result['sequence'],
                    'num_structures_used': len(entries),
                    'missing_positions': reconstruction_result['missing_positions'],
                    'source_pdbs': reconstruction_result['source_pdbs'],
                    'variations': reconstruction_result['variations']
                })
            else:  # unreconstructable
                unreconstructable.append({
                    'uniprot_id': uniprot_id,
                    'kinase_name': kinase_name,
                    'kinase_klifs_id': kinase_klifs_id,
                    'num_structures': len(entries),
                    'all_sequences': [e['pocket_85aa'] for e in entries],
                    'reason': 'All structures have gaps at all positions'
                })

    return {
        'complete': complete_sequences,
        'fully_reconstructed': fully_reconstructed,
        'partially_reconstructed': partially_reconstructed,
        'unreconstructable': unreconstructable
    }


def reconstruct_sequence(entries: List[Dict]) -> Dict:
    """
    Reconstruct a sequence by overlaying multiple PDB structures.
    Returns the reconstructed sequence and metadata about variations.
    """
    sequences = [e['pocket_85aa'] for e in entries]

    # Find the maximum length (should be 85 for most)
    seq_length = max(len(seq) for seq in sequences)

    # For each position, collect all non-gap amino acids
    position_data = []
    for pos in range(seq_length):
        amino_acids = []
        source_pdbs = []

        for idx, seq in enumerate(sequences):
            # Handle sequences shorter than the expected length
            if pos < len(seq):
                aa = seq[pos]
                if aa != '_':
                    amino_acids.append(aa)
                    source_pdbs.append(entries[idx]['pdb_id'])

        position_data.append({
            'amino_acids': amino_acids,
            'source_pdbs': source_pdbs
        })

    # Build reconstructed sequence
    reconstructed = []
    missing_positions = []
    variations = []
    all_source_pdbs = set()

    for pos, data in enumerate(position_data):
        if not data['amino_acids']:
            # No amino acid found at this position
            reconstructed.append('_')
            missing_positions.append(pos)
        else:
            # Check for variations (different amino acids at same position)
            unique_aas = set(data['amino_acids'])

            if len(unique_aas) > 1:
                # Variation detected - this is unusual and should be reported
                variations.append({
                    'position': pos,
                    'amino_acids': list(unique_aas),
                    'counts': {aa: data['amino_acids'].count(aa) for aa in unique_aas},
                    'source_pdbs': data['source_pdbs']
                })
                # Use the most common amino acid
                most_common = max(unique_aas, key=lambda aa: data['amino_acids'].count(aa))
                reconstructed.append(most_common)
            else:
                # All agree on the same amino acid
                reconstructed.append(data['amino_acids'][0])

            all_source_pdbs.update(data['source_pdbs'])

    reconstructed_seq = ''.join(reconstructed)

    # Determine status
    if not missing_positions:
        status = 'fully_reconstructed'
    elif len(missing_positions) == seq_length:
        status = 'unreconstructable'
    else:
        status = 'partially_reconstructed'

    return {
        'status': status,
        'sequence': reconstructed_seq,
        'missing_positions': missing_positions,
        'source_pdbs': ','.join(sorted(all_source_pdbs)),
        'variations': format_variations(variations)
    }


def format_variations(variations: List[Dict]) -> str:
    """Format variation information as a readable string."""
    if not variations:
        return 'None'

    var_strings = []
    for var in variations:
        aa_info = ','.join([f"{aa}({var['counts'][aa]})" for aa in sorted(var['amino_acids'])])
        var_strings.append(f"pos{var['position']}:[{aa_info}]")

    return '; '.join(var_strings)


def save_results(results: Dict, output_prefix: str):
    """Save results to CSV files."""

    # 1. Complete sequences
    with open(f'{output_prefix}_complete.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uniprot_id', 'kinase_name', 'kinase_klifs_id', 'sequence_85aa',
            'source_pdb', 'num_complete_structures'
        ])
        writer.writeheader()
        writer.writerows(results['complete'])

    # 2. Fully reconstructed sequences
    with open(f'{output_prefix}_fully_reconstructed.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uniprot_id', 'kinase_name', 'kinase_klifs_id', 'sequence_85aa',
            'num_structures_used', 'source_pdbs', 'variations'
        ])
        writer.writeheader()
        writer.writerows(results['fully_reconstructed'])

    # 3. Partially reconstructed sequences
    with open(f'{output_prefix}_partially_reconstructed.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uniprot_id', 'kinase_name', 'kinase_klifs_id', 'sequence_85aa',
            'num_structures_used', 'num_missing_positions', 'missing_positions',
            'source_pdbs', 'variations'
        ])
        writer.writeheader()
        for row in results['partially_reconstructed']:
            row['num_missing_positions'] = len(row['missing_positions'])
            row['missing_positions'] = ','.join(map(str, row['missing_positions']))
        writer.writerows(results['partially_reconstructed'])

    # 4. Unreconstructable sequences
    with open(f'{output_prefix}_unreconstructable.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uniprot_id', 'kinase_name', 'kinase_klifs_id', 'num_structures', 'reason'
        ])
        writer.writeheader()
        for row in results['unreconstructable']:
            # Remove all_sequences from output (too verbose)
            row_output = {k: v for k, v in row.items() if k != 'all_sequences'}
            writer.writerow(row_output)

    # Print summary
    print("\n" + "="*80)
    print("SEQUENCE RECONSTRUCTION SUMMARY")
    print("="*80)
    print(f"\nComplete sequences (no gaps):          {len(results['complete']):4d}")
    print(f"Fully reconstructed sequences:         {len(results['fully_reconstructed']):4d}")
    print(f"Partially reconstructed sequences:     {len(results['partially_reconstructed']):4d}")
    print(f"Unreconstructable sequences:           {len(results['unreconstructable']):4d}")
    print(f"{'─'*80}")
    print(f"Total unique proteins:                 {sum(len(results[k]) for k in results):4d}")

    # Report variations
    total_variations = sum(1 for r in results['fully_reconstructed'] if r['variations'] != 'None')
    total_variations += sum(1 for r in results['partially_reconstructed'] if r['variations'] != 'None')

    if total_variations > 0:
        print(f"\n⚠️  WARNING: {total_variations} proteins have sequence variations (mutations)")
        print("   Check 'variations' column in output files for details.")

    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("="*80)
    print(f"  {output_prefix}_complete.csv")
    print(f"  {output_prefix}_fully_reconstructed.csv")
    print(f"  {output_prefix}_partially_reconstructed.csv")
    print(f"  {output_prefix}_unreconstructable.csv")
    print()


def main():
    input_file = 'klifs_pocket_sequences_human_wt_from_uniprot.csv'
    output_prefix = 'klifs_sequences'

    print(f"Reading sequences from: {input_file}")
    print("Processing...")

    results = analyze_sequences(input_file)
    save_results(results, output_prefix)

    print("Done!")


if __name__ == '__main__':
    main()
