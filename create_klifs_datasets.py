import json
import csv
import zipfile
from xml.etree import ElementTree as ET
import sys

def parse_excel_klifs(excel_file):
    zf = zipfile.ZipFile(excel_file)
    shared = ET.parse(zf.open('xl/sharedStrings.xml'))
    strings = []
    for s in shared.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
        t_elem = s.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
        strings.append(t_elem.text if t_elem is not None and t_elem.text else '')

    sheet = ET.parse(zf.open('xl/worksheets/sheet1.xml'))
    rows = sheet.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')

    klifs_dict = {}
    uniprot_to_klifs = {}

    for i, row in enumerate(rows[1:]):
        cells = row.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c')
        values = []
        for c in cells:
            v_elem = c.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
            if v_elem is not None and v_elem.text:
                if c.get('t') == 's':
                    values.append(strings[int(v_elem.text)])
                else:
                    values.append(v_elem.text)
            else:
                values.append('')

        if len(values) >= 4:
            kinase_name = values[0]
            pocket_seq = values[1]
            hgnc = values[2] if len(values) > 2 else ''
            uniprot = values[3] if len(values) > 3 else ''

            if kinase_name and pocket_seq:
                klifs_dict[kinase_name] = pocket_seq
                if uniprot:
                    uniprot_to_klifs[uniprot] = pocket_seq

    return klifs_dict, uniprot_to_klifs

def normalize_kinase_name(name):
    if '(' in name:
        return name.split('(')[0]
    if 'p' == name[-1] and name[:-1].isalnum():
        return name[:-1]
    return name

def create_klifs_dataset(input_csv, output_csv, protein_dict, klifs_dict, use_uniprot=False):
    matched = 0
    unmatched = 0
    unmatched_proteins = set()

    with open(input_csv, 'r') as fin, open(output_csv, 'w') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=['compound_iso_smiles', 'target_sequence', 'affinity'])
        writer.writeheader()

        for row in reader:
            smiles = row['compound_iso_smiles']
            full_seq = row['target_sequence']
            affinity = row['affinity']

            protein_id = None
            for pid, seq in protein_dict.items():
                if seq == full_seq:
                    protein_id = pid
                    break

            if protein_id:
                if use_uniprot:
                    klifs_seq = klifs_dict.get(protein_id)
                else:
                    normalized = normalize_kinase_name(protein_id)
                    klifs_seq = klifs_dict.get(protein_id) or klifs_dict.get(normalized)

                if klifs_seq:
                    writer.writerow({
                        'compound_iso_smiles': smiles,
                        'target_sequence': klifs_seq,
                        'affinity': affinity
                    })
                    matched += 1
                else:
                    unmatched += 1
                    unmatched_proteins.add(protein_id)

    return matched, unmatched, unmatched_proteins

print('Parsing Excel file...')
klifs_by_name, klifs_by_uniprot = parse_excel_klifs('data/human_kinase_pocket_sequences.xlsx')
print(f'Loaded {len(klifs_by_name)} KLIFS sequences by name')
print(f'Loaded {len(klifs_by_uniprot)} KLIFS sequences by UniProt ID')

print('\nLoading Davis proteins...')
with open('data/davis/proteins.txt', 'r') as f:
    davis_proteins = json.load(f)
print(f'Loaded {len(davis_proteins)} Davis proteins')

print('\nLoading KIBA proteins...')
with open('data/kiba/proteins.txt', 'r') as f:
    kiba_proteins = json.load(f)
print(f'Loaded {len(kiba_proteins)} KIBA proteins')

print('\n=== Processing Davis datasets ===')
matched, unmatched, unmatched_set = create_klifs_dataset(
    'data/davis_train.csv', 'data/davis_train_klifs.csv',
    davis_proteins, klifs_by_name, use_uniprot=False
)
print(f'Davis train: {matched} matched, {unmatched} unmatched')

matched, unmatched, unmatched_set_test = create_klifs_dataset(
    'data/davis_test.csv', 'data/davis_test_klifs.csv',
    davis_proteins, klifs_by_name, use_uniprot=False
)
print(f'Davis test: {matched} matched, {unmatched} unmatched')

all_unmatched_davis = unmatched_set.union(unmatched_set_test)
if all_unmatched_davis:
    print(f'\nUnmatched Davis proteins ({len(all_unmatched_davis)}):')
    print(', '.join(sorted(list(all_unmatched_davis))[:20]))
    if len(all_unmatched_davis) > 20:
        print(f'... and {len(all_unmatched_davis)-20} more')

print('\n=== Processing KIBA datasets ===')
matched, unmatched, unmatched_set = create_klifs_dataset(
    'data/kiba_train.csv', 'data/kiba_train_klifs.csv',
    kiba_proteins, klifs_by_uniprot, use_uniprot=True
)
print(f'KIBA train: {matched} matched, {unmatched} unmatched')

matched, unmatched, unmatched_set_test = create_klifs_dataset(
    'data/kiba_test.csv', 'data/kiba_test_klifs.csv',
    kiba_proteins, klifs_by_uniprot, use_uniprot=True
)
print(f'KIBA test: {matched} matched, {unmatched} unmatched')

all_unmatched_kiba = unmatched_set.union(unmatched_set_test)
if all_unmatched_kiba:
    print(f'\nUnmatched KIBA proteins ({len(all_unmatched_kiba)}):')
    print(', '.join(sorted(list(all_unmatched_kiba))[:20]))
    if len(all_unmatched_kiba) > 20:
        print(f'... and {len(all_unmatched_kiba)-20} more')

print('\nDone! Created KLIFS datasets.')
print('Next: Run create_data_chembl.py to generate .pt files')
