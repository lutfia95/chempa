# Molecule Mention Report

This folder contains a small pipeline for scanning a TSV of molecules and producing a report of likely literature and patent mentions.

The main script is [molecule_mention_report.py](./molecule_mention_report.py).

## What The Script Does

For each input row:

1. Read the `SMILES` string from the TSV.
2. Resolve that SMILES through PubChem.
3. Extract PubChem identity fields:
   - `CID`
   - `InChIKey`
   - `IUPACName`
   - canonical or connectivity SMILES
4. Build search queries from those identity fields.
5. Search:
   - Europe PMC for paper hits
   - Google Patents for patent hits
6. Write structured outputs as:
   - JSONL
   - TSV
   - summary text
7. Print runtime progress to stderr while running.

## Input Format

The input file must be a tab-separated file with at least:

- `SMILES`

It can also include:

- `Label`

If `Label` is present, it is carried through into the output and progress logs.

## Search Strategy

The search is not based on substructure chemistry or RDKit matching. It is text-based retrieval built from PubChem-resolved identifiers.

For each molecule, the script builds up to three quoted query types:

1. `exact_identifier`
   - Query: the PubChem `InChIKey`
   - Confidence label: `high`
2. `exact_name`
   - Query: the PubChem `IUPACName`
   - Confidence label: `medium`
3. `exact_smiles`
   - Query: PubChem canonical/connectivity SMILES, or the original input SMILES
   - Confidence label: `low`

These labels are query-strength labels. They are not proof of chemical identity in the source text.

## What "Match" Means Here

This script does not currently implement a formal `complete match` vs `partial match` classifier.

What it really does is:

- send exact quoted text queries to the external search systems
- collect the returned records
- tag the result with the query type that produced it

So the current meaning is:

- `exact_identifier`: strongest signal because `InChIKey` is specific
- `exact_name`: moderate signal because names can vary
- `exact_smiles`: weaker signal because SMILES strings are brittle in text and can vary by normalization

There is currently no explicit:

- fuzzy name match
- synonym expansion
- alias resolution
- partial structure match
- substructure search
- stereochemistry-aware equivalence check across text mentions

## Paper Search Behavior

Paper search uses Europe PMC.

For each quoted query, the script calls the Europe PMC search API and accepts returned records as candidate mentions. It stores metadata such as:

- title
- year
- authors
- journal/source
- DOI/PMID URL when available
- abstract snippet

This means the script depends on Europe PMC search behavior. It does not fetch and verify the full article body itself.

## Patent Search Behavior

Patent search uses Google Patents search result pages.

For each quoted query, the script:

1. fetches the search results page
2. parses patent links from the HTML
3. records those patents as candidate mentions

Important limitation:

Current patent hits are mostly "search-result leads", not body-verified patent mentions.

In other words, a patent record in the output currently means:

- Google Patents returned this patent for the query

It does not yet mean:

- the exact identifier/name/SMILES was confirmed in the patent body, abstract, claims, or examples

So patents should be treated as review targets unless the script is extended with body-text verification.

## Runtime Logging

While running, the script prints progress information to stderr.

There are two log styles:

1. Per-row summary
   - example:
   - `[73/619] info label=WX604420 smiles=COC(=O)c1ccc(C2(N)CC2)cc1 pubchem_cid=45140209 found=Europe PMC=1`
2. Per-hit detail
   - includes source, database, confidence, match mode, year, reference id, query, and title

The first tells you that the row produced one or more hits.
The second tells you which records were found.

## Output Files

Given:

- `--out-prefix mention_report`

The script writes:

- `mention_report.jsonl`
- `mention_report.tsv`
- `mention_report_summary.txt`

### JSONL

One record per input molecule, including:

- original input row
- resolved PubChem identity
- list of mention records

### TSV

Flattened tabular output with one row per mention.

If a molecule has no mentions, a base row is still written with identity information and `MentionCount = 0`.

### Summary

A rolling summary of:

- total input rows processed
- rows with mentions
- PubChem resolve errors
- databases searched
- counts by source type
- counts by database

## Deduplication

Mentions are deduplicated by:

- `(database, id)`
- or `(database, url)` if `id` is missing

If the same mention is found via multiple query types, the script keeps the best confidence in this order:

1. `high`
2. `medium`
3. `low`

## Current Strengths

- Simple and easy to run
- Good for first-pass triage
- Strongest when `InChIKey` is available
- Produces structured outputs for later review
- Supports concurrent processing with rate limiting

## Current Weaknesses

- No synonym expansion beyond PubChem `IUPACName`
- No partial/fuzzy matching logic
- No article full-text verification
- No patent body verification
- No chemistry-aware structure equivalence beyond PubChem normalization
- `exact_smiles` is often fragile for literature text search

## How To Run

Example:

```bash
python3 molecule_mention_report.py \
  --input aromatic_amines.tsv \
  --out-prefix aromatic_amines_mentions
```

Useful options:

- `--limit 20`
- `--workers 8`
- `--timeout 30`
- `--retries 2`
- `--per-source-limit 5`
- `--no-papers`
- `--no-patents`

See full CLI help:

```bash
python3 molecule_mention_report.py --help
```

## Recommended Interpretation

Use this script as a discovery and prioritization tool.

Interpret results this way:

- Europe PMC `exact_identifier`: usually strong evidence worth reviewing first
- Europe PMC `exact_name`: useful but can still be ambiguous
- Europe PMC `exact_smiles`: weak and should be checked manually
- Google Patents hits: leads only unless manually verified

## Recommended Next Improvement

The most important upgrade would be patent body verification:

- fetch each patent page
- extract visible text
- confirm the identifier/name/SMILES is actually present
- distinguish:
  - `search_result_only`
  - `body_verified`

That would make the patent side much more reliable.
