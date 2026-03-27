# Molecule Mention Report

This folder contains the inputs and outputs for `molecule_mention_report.py`.

The purpose of the script is simple: take a TSV of molecules, search for likely mentions of those molecules in papers and patents, and write the results in machine-readable files.

## What The Script Searches

The script searches for mentions of molecules represented by `SMILES` strings.

In this folder, the main input molecule lists are:

- `primary_aliphatic_amines.tsv`
- `secondary_aliphatic_amines.tsv`
- `aromatic_amines.tsv`
- `others.tsv`

Each row is one molecule. The minimum required input column is:

- `SMILES`

An optional `Label` column can also be present. If it exists, it is carried into the outputs and log messages.

## Where The Search Happens

The script searches in two external sources:

- Europe PMC for papers
- Google Patents for patents

This is a text-retrieval workflow. It is not doing:

- substructure search
- similarity search
- RDKit structure matching against external databases
- full document chemistry parsing

## How The Search Works

For each input molecule, `molecule_mention_report.py` does the following:

1. Read the molecule's `SMILES`.
2. Send that SMILES to PubChem.
3. Try to resolve identifier fields such as:
   - `CID`
   - `InChIKey`
   - `IUPACName`
   - canonical or connectivity SMILES
4. Build quoted text queries from those resolved identifiers.
5. Search Europe PMC and Google Patents with those queries.
6. Collect the returned records as candidate mentions.
7. Write JSONL, TSV, and summary outputs.

So the search is identifier-driven. The script tries to find literature and patent records that mention the same molecule textually.

## What The Script Looks For

The script tries to determine whether each molecule is:

- found in papers
- found in patents
- found in both
- not found

In the current implementation:

- paper hits come from Europe PMC search results
- patent hits come from Google Patents search results

## Search Queries Used

For each molecule, the script can build up to three query types:

1. `exact_identifier`
   - uses the PubChem `InChIKey`
2. `exact_name`
   - uses the PubChem `IUPACName`
3. `exact_smiles`
   - uses canonical/connectivity SMILES from PubChem, or the original input SMILES

These are quoted text queries. They are not proof that the compound was chemically verified in the source.

## What "Found" Means

For this workflow, a molecule is treated as found when the search returns one or more candidate mention records.

Current interpretation:

- `papers only`: at least one paper mention was recorded, and no patent mention was recorded
- `patents only`: at least one patent mention was recorded, and no paper mention was recorded
- `both`: at least one paper mention and at least one patent mention were recorded
- `not found`: no mention records were recorded for that molecule

## Important Limitations

This is a first-pass discovery workflow, not a final validation system.

Important limits:

- no synonym expansion beyond PubChem-resolved names
- no fuzzy name matching
- no full-text article verification
- no patent body verification
- no chemistry-aware equivalence checking beyond PubChem normalization
- `exact_smiles` matches are relatively weak because textual SMILES usage is inconsistent

Patent hits especially should be treated as leads for review, not automatic confirmation.

## Output Layout In This Folder

The top-level TSV files are molecule input sets.

The `*_scan/` folders hold the outputs from `molecule_mention_report.py`.

Example layout:

```text
out/amine_split/
├── primary_aliphatic_amines.tsv
├── secondary_aliphatic_amines.tsv
├── aromatic_amines.tsv
├── others.tsv
├── primary_scan/
│   ├── primary_aliphatic_amines.jsonl
│   ├── primary_aliphatic_amines.tsv
│   └── primary_aliphatic_amines_summary.txt
├── secondary_scan/
├── aromatic_scan/
└── others_scan/
```

## Meaning Of The Output Files

For a scan prefix such as `primary_aliphatic_amines`, the script writes:

- `primary_aliphatic_amines.jsonl`
- `primary_aliphatic_amines.tsv`
- `primary_aliphatic_amines_summary.txt`

### `.jsonl`

One JSON record per input molecule.

Each record includes:

- the original input row
- PubChem identity fields
- the collected mention records

This is the most structured output if you want to post-process results programmatically.

### `.tsv`

Flattened tabular output.

Important detail:

- if a molecule has multiple mentions, the TSV can contain multiple rows for that same molecule
- if a molecule has no mentions, the TSV still includes a base row with `MentionCount = 0`

This is the main file used by `plot_tsv.py`.

### `_summary.txt`

Human-readable running summary of the scan, including:

- how many input rows were processed
- how many rows had mentions
- how many PubChem resolutions failed
- counts by source type
- counts by database

## Plotting The Scan Results

The plotting script at `/Users/ahmadlutfi/Downloads/Xiaoyi/chempa/github/chempa/plot_tsv.py` reads the flattened scan TSV and collapses it back to one result per unique molecule.

It then groups molecules into:

- `Papers`
- `Patents`
- `Both`
- `Not found`

It also writes two text files:

- one listing the found molecules
- one listing the not-found molecules

Those files are generated from the scan TSV content, not from the raw input TSV.

## How To Run The Search

Example:

```bash
python3 molecule_mention_report.py \
  --input primary_aliphatic_amines.tsv \
  --out-prefix primary_scan/primary_aliphatic_amines
```

Useful options:

- `--limit`
- `--workers`
- `--timeout`
- `--retries`
- `--per-source-limit`
- `--no-papers`
- `--no-patents`

Help:

```bash
python3 molecule_mention_report.py --help
```

## How To Plot The Search Results

Example for one scan TSV:

```bash
python3 plot_tsv.py \
  --tsv ./out/amine_split/primary_scan/primary_aliphatic_amines.tsv
```

Example for all scan TSVs in this folder:

```bash
python3 plot_tsv.py \
  --input-dir ./out/amine_split \
  --output-dir ./out/plots/scan_audit
```
