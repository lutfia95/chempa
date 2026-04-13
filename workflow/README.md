# Workflow

This package implements the clustering workflow discussed earlier:

`RDKit standardization -> MoLFormer embeddings -> PCA -> UMAP -> HDBSCAN -> chemical summaries`

The entrypoint is:

```bash
python -m workflow.main --config workflow/config.toml
```

Setup:

Conda is the recommended route because RDKit is a hard dependency and is usually simpler to install from `conda-forge`.

```bash
conda env create -f workflow/environment.yml
conda activate chempa-workflow
python -m workflow.main --config workflow/config.toml
```

If you already have RDKit installed in another environment, you can install the remaining Python packages with:

```bash
pip install -r workflow/requirements.txt
```

Run:

```bash
python -m workflow.main --config workflow/config.toml
```

The important knobs live in `workflow/config.toml`:

- input TSV path
- SMILES column name
- standardization parent strategy
- embedding backend and model name
- PCA / UMAP parameters
- HDBSCAN parameters
- output location

Notes:

- `InputSMILES` is the SMILES column for `out/amine_split/primary_amins.tsv`.
- `backend = "molformer"` is the intended primary mode.
- `backend = "morgan_bits"` is available as a fallback for local debugging.
- RDKit is required for the chemistry stages.
- UMAP is imported after setting a writable numba cache directory.
- The first `molformer` run may need to download model weights unless you set `local_files_only = true` and have them cached already.
- The workflow is pinned to `transformers==4.48.3` because the MoLFormer custom code can fail on incompatible `transformers` builds with errors such as `ModuleNotFoundError: No module named 'transformers.onnx'`.

## Analysis Steps

The workflow processes the input table in a fixed order. The default input for this project is:

- `out/amine_split/primary_amins.tsv`
- SMILES column: `InputSMILES`

### 1. Load the input table

The workflow reads the TSV defined in `workflow/config.toml` and checks that the configured SMILES column exists.

What happens:

- the input file is loaded as text
- a stable `row_id` column is added
- no chemistry is done yet at this point

Why it matters:

- later files use `row_id` so rows can be traced across outputs

### 2. Parse and standardize molecules with RDKit

Each `InputSMILES` string is parsed with RDKit and then converted to one consistent parent form.

What happens:

- empty SMILES are marked invalid
- RDKit parsing is attempted
- the configured parent strategy is applied:
  - `fragment`
  - `charge`
  - `tautomer`
  - `super`
  - `none`
- the molecule is converted back to canonical SMILES
- the canonical SMILES is reparsed to stabilize RDKit caches
- ring information is recomputed before Murcko scaffold extraction

What is computed here:

- `canonical_smiles`
- `standardized_smiles`
- `murcko_scaffold`
- RDKit descriptors:
  - `MW`
  - `logP`
  - `HBD`
  - `HBA`
  - `TPSA`
  - `RB`
  - `Rings`
  - `HeavyAtoms`
  - `NumAtoms`

What can be filtered here:

- invalid molecules
- duplicate molecules after canonicalization

Important output:

- `standardized.tsv`

### 3. Deduplicate standardized molecules

After standardization, duplicate canonical structures are identified.

What happens:

- each unique `canonical_smiles` is counted
- the first occurrence is kept
- later occurrences are marked as duplicates
- if `deduplicate = true`, duplicates are removed from downstream analysis

Why it matters:

- clustering should reflect unique chemistry, not repeated rows

### 4. Compute Morgan fingerprints

Morgan fingerprints are computed for every retained molecule.

What happens:

- RDKit generates circular fingerprints using the configured:
  - `radius`
  - `n_bits`
- the workflow stores both dense and sparse fingerprint representations

Why it matters:

- fingerprints provide a classical cheminformatics baseline
- they are used for fingerprint nearest-neighbor analysis
- they can also be used as a fallback embedding backend

Important output:

- `morgan_fingerprints.npz`

### 5. Build molecular embeddings

The workflow converts each standardized molecule into a numeric vector.

Available backends:

- `molformer`
- `morgan_bits`

Default behavior:

- `molformer` is the primary backend
- standardized SMILES are tokenized
- the pretrained MoLFormer model is run
- token embeddings are pooled into one vector per molecule
- vectors are normalized if `normalize = true`

Fallback behavior:

- if `backend = "morgan_bits"`, the fingerprint bit vectors are used directly as embeddings

Important output:

- `embeddings.npy`

### 6. Reduce dimensionality with PCA

The high-dimensional embeddings are first compressed with PCA.

What happens:

- the embedding matrix is reduced to `pca_components`
- this produces a more stable intermediate space before UMAP

Why it matters:

- PCA removes some noise and redundancy
- clustering is more stable than using raw high-dimensional vectors directly

Important output:

- `pca_50.npy`

### 7. Build UMAP spaces

The workflow runs UMAP twice.

What happens:

- one UMAP embedding is generated for clustering
- one UMAP embedding is generated for 2D plotting

Default intent:

- cluster on the higher-dimensional UMAP space
- visualize on the 2D UMAP space

Why it matters:

- 2D UMAP is for inspection
- the clustering space should preserve more structure than a flat plot

Important outputs:

- `umap_cluster.npy`
- `umap_2d.tsv`

### 8. Cluster molecules with HDBSCAN

The clustering step is run on the UMAP clustering space.

What happens:

- HDBSCAN groups molecules by local density
- some molecules may be labeled as `-1`, meaning noise or uncertain assignment
- the workflow computes basic cluster metrics

Metrics include:

- number of clusters
- number of noise points
- non-noise count
- silhouette score when it is meaningful to compute
- cluster size distribution

Important outputs:

- `clusters.tsv`
- `cluster_metrics.json`

### 9. Choose cluster representatives

For each non-noise cluster, the workflow chooses a representative molecule.

What happens:

- the cluster centroid is computed in clustering space
- the closest molecule to that centroid is selected

Why it matters:

- this gives one interpretable molecule per cluster for inspection

Important output:

- `cluster_representatives.tsv`

### 10. Summarize chemotypes with Murcko scaffolds

The workflow counts the most common Murcko scaffolds in each cluster.

What happens:

- molecules are grouped by `cluster_id`
- scaffolds are counted inside each cluster
- the top scaffolds are exported

Why it matters:

- this helps distinguish coherent chemotypes from mixed or noisy clusters

Important output:

- `cluster_scaffolds.tsv`

### 11. Summarize descriptor ranges per cluster

For each non-noise cluster, the workflow summarizes descriptor values.

What happens:

- mean, min, and max are computed for configured descriptors
- descriptors come from the RDKit standardization stage

Why it matters:

- this shows whether a cluster is structurally coherent but also whether it is biased by size, polarity, or lipophilicity

Important output:

- `cluster_descriptors.tsv`

### 12. Compute nearest neighbors in two ways

The workflow exports neighborhood relationships using both learned and classical similarity.

What happens:

- cosine similarity is computed on embeddings
- Tanimoto similarity is computed on Morgan fingerprints
- top neighbors are exported for each molecule

Why it matters:

- embedding neighbors show representation-space proximity
- fingerprint neighbors show classical structural similarity
- agreement between the two usually makes a local region more trustworthy

Important outputs:

- `nearest_neighbors_embedding.tsv`
- `nearest_neighbors_fingerprint.tsv`

### 13. Generate UMAP plots

Three plots are generated from the 2D UMAP space.

Plots:

- clusters colored by `cluster_id`
- molecules colored by `MW`
- molecules colored by `logP`

Why it matters:

- the cluster plot shows group separation and outliers
- the descriptor plots help you see whether the space is organized mainly by scaffold or by simple bulk properties

Important outputs:

- `umap_clusters.png`
- `umap_mw.png`
- `umap_logp.png`

### 14. Write run metadata

The workflow writes one metadata file that records the run context.

What happens:

- the resolved config is saved
- the standardization summary is saved
- clustering metrics are saved
- output file paths are recorded

Important output:

- `run_metadata.json`

## Output Files

The main output directory is:

- `workflow/results/<run_name>/`

Default run name:

- `primary_amins`

Typical files you will inspect first:

- `standardized.tsv`
- `clusters.tsv`
- `library_overview.json`
- `executive_summary.txt`
- `chemical_landscape_report.txt`
- `super_regions.tsv`
- `functional_group_summary.tsv`
- `scaffold_family_summary.tsv`
- `cluster_representatives.tsv`
- `cluster_scaffolds.tsv`
- `cluster_descriptors.tsv`
- `cluster_metrics.json`
- `umap_clusters.png`

## Recommended Reading Order

If you want to interpret one run step by step, read the outputs in this order:

1. `standardized.tsv`
2. `executive_summary.txt`
3. `chemical_landscape_report.txt`
4. `library_overview.json`
5. `property_landscape.json`
6. `super_regions.tsv`
7. `functional_group_summary.tsv`
8. `scaffold_family_summary.tsv`
9. `cluster_metrics.json`
10. `cluster_representatives.tsv`
11. `cluster_scaffolds.tsv`
12. `cluster_descriptors.tsv`
13. `cluster_interpretation.tsv`
14. `outlier_summary.tsv`
15. `nearest_neighbors_embedding.tsv`
16. `nearest_neighbors_fingerprint.tsv`
17. `umap_clusters.png`
18. `umap_mw.png`
19. `umap_logp.png`
