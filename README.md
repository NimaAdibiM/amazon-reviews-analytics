# Amazon Reviews 2023 — Data Processing and Feature Representation

Python implementation for preparing product-level review data and constructing numerical and text representations for downstream analysis. The pipeline targets four Amazon Reviews 2023 categories: Electronics, Beauty_and_Personal_Care, Appliances, and All_Beauty.

## Project structure

### `data_preparation/`

| File | Responsibility |
| --- | --- |
| `scan_reviews.py` | Entry point for scanning the four review categories. |
| `ingest.py` | Constructs product-level review cohorts using bounded, disk-backed indexing. |
| `metadata.py` | Reads and audits product metadata separately from prediction-time features. |
| `features.py` | Computes product features from the first five observed reviews. |
| `split_scale.py` | Assigns product splits and fits numerical scaling on training data. |
| `prepare.py` | Coordinates feature generation, table exports, and data-quality summaries. |
| `verify.py` | Checks prepared outputs, joins, and recorded file hashes. |

### `feature_representation/`

| File | Responsibility |
| --- | --- |
| `__main__.py` | Command-line interface for running and inspecting the representation pipeline. |
| `pipeline.py` | Orchestrates reading, fitting, transforming, and artifact publication. |
| `contracts.py` | Validates input schemas and exports aligned early-review text. |
| `partner.py` | Loads feature definitions from the data-preparation modules. |
| `classical.py` | Builds TF-IDF features and TruncatedSVD projections. |
| `semantic.py` | Implements BGE/E5 sentence-embedding backends and review chunking. |
| `aggregation.py` | Combines five review embeddings into product-level representations. |
| `numeric.py` | Prepares numeric feature groups and PCA representations. |
| `sampling.py` | Extracts bounded samples from prepared product-level outputs. |
| `experiments.py` | Supplies language, dimensionality, and paired-comparison diagnostics. |
| `artifacts.py` | Persists and validates transactional representation artifacts. |
| `artifacts_v1.py` | Reads artifacts produced with the earlier storage format. |
| `token_stats.py` | Computes bounded-memory token-length statistics. |
| `plots.py` | Generates representation diagnostics and projections. |
| `common.py` | Shared constants, file readers, hashes, and environment details. |
| `__init__.py` | Package metadata. |




