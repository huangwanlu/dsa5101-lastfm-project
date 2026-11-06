# Friendship Communities and Music Recommendation

This DSA5101 project investigates whether friendship-graph communities improve artist recommendation beyond a listening-history model, particularly for users with sparse observed histories.

## Dataset

The project uses the publicly available HetRec 2011 Last.fm 2K dataset, released for the HetRec 2011 workshop. The working archive contains aggregate user-artist listening counts and a reciprocal friendship graph. It has no listening-event timestamps, so the evaluation uses random within-user holdout rather than temporal forecasting.

The notebook downloads the archive from the [GroupLens HetRec 2011 dataset page](https://grouplens.org/datasets/hetrec-2011/) when it is not already available at:

```text
data/raw/hetrec2011-lastfm-2k.zip
```

It creates the directory as needed, reuses an existing archive, and validates the required archive members before reading them. Raw data are intentionally not included in this repository or submission ZIP.

## Methods and results

Two algorithm classes are applied:

- **Community detection:** Louvain clustering of the undirected friendship graph, followed by community-level artist popularity with shrinkage toward global popularity.
- **Matrix factorization:** implicit alternating least squares (ALS) fitted to user-artist listening counts.

Global popularity is used as a baseline, and a normalized hybrid combines ALS scores with community-based scores. On the primary test evaluation, the hybrid achieved NDCG@10 of 0.240868 versus 0.239310 for ALS. The paired mean gain was 0.001558, with a 95% bootstrap interval of [0.000129, 0.003037], meeting the planned criterion for a small improvement over ALS. Community-only recommendation improved over global popularity but remained weaker than ALS.

## Repository structure

```text
report.tex              LaTeX source for the report
report.pdf              Compiled report
analysis.ipynb          Executed analysis notebook
requirements.txt        Python dependencies and versions
pyproject.toml          Project configuration
uv.lock                 Locked environment dependencies
report_assets/          Figures and the NUS logo used by the report
docs/                   Data contract and research protocol
src/lastfm_project/     Reusable project modules
tests/                  Automated tests for project modules
```

## Reproduction

Install the pinned dependencies with:

```bash
uv sync --locked
```

Alternatively, install the packages listed in `requirements.txt` in a Python 3.11 environment. Restart the notebook kernel and run `analysis.ipynb` from beginning to end; the dataset is downloaded automatically if missing. The notebook reconstructs the data checks, split, community detection, model selection, final fitting, evaluation, inference, and diagnostics. The analysis uses seed 5101 and records the relevant software and numerical settings.

Run the automated tests with:

```bash
uv run --locked pytest -q
```

## Submission archive

`A0237821E.zip` is the submission archive. It contains exactly these three top-level files, as required by the project instructions:

```text
report.pdf
analysis.ipynb
requirements.txt
```
