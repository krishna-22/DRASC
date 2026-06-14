# DRASC: Density-Ratio Adaptive Spectral Clustering

This repository holds the benchmark code,  and the result files
for DRASC, a one-parameter change to spectral clustering. DRASC multiplies the
adaptive Gaussian affinity by a density-ratio gate raised to an exponent gamma.
Setting gamma to zero recovers self-tuning spectral clustering, so the effect of
the gate can be measured on its own.

## What is in this repository

- The main Python script, which runs the full benchmark and writes every result file.
- The generated results: a table in CSV form, a plain-text summary, an Excel
  workbook with one sheet per table

## Requirements

- Python 3.9 or newer
- numpy, scipy, pandas, scikit-learn
- openpyxl, for the Excel summary
- matplotlib, only for the figures

Optional, each enabling an extra baseline or view:

- hdbscan, via: pip install hdbscan
- finch-clust, via: pip install finch-clust
- umap-learn, only if you want UMAP figures in place of t-SNE

Install the core set with pip:

    pip install numpy scipy pandas scikit-learn openpyxl matplotlib

## How to run

Run the main script with Python 3. With no arguments it reproduces the full
study: the main benchmark, the label-free selection study, the robustness
sweeps, the scalability run on MNIST, the summary export, and the figures.

Useful flags:

- --smoke, a fast subset for a quick check that everything works
- --all, only the main benchmark across all datasets
- --robustness, only the noise and imbalance sweeps
- --sensitivity, only the k and gamma grid
- --mnist, only the scalability run on the full MNIST set
- --viz, only the figures
- --viz-method tsne | umap, choose the projection used for the figures
- --no-images, skip the large image datasets
- --summary, rebuild the summary files from results that already exist

Quick check:

    python drasc_benchmark.py --smoke



## Datasets

All datasets load automatically. The small tabular sets and the Olivetti faces
come through scikit-learn and OpenML. MNIST is downloaded from a public mirror
on first use. The image sets are standardized and reduced to fifty principal
components before clustering. You need an internet connection on the first run
so the downloads and the OpenML fetch can finish. Results are cached afterward.

## Outputs

Everything is written to an output folder created in the working directory:

- a results table in CSV form, one row per dataset, method, and protocol
- a plain-text summary with the per-table pivots, mean ranks, and the tests
- an Excel workbook with one sheet per table
- per-dataset figures in PNG and PDF form
- a short record of the run environment, with Python and library versions

## Reproducing the paper tables

The CSV holds every value behind the tables in the paper. The summary text and
the Excel workbook are derived from it. Mean ranks, the Friedman test, and the
pairwise Wilcoxon tests are recomputed during the export step, so you can check
them straight from the CSV.

## Citation

If you use this code, please cite the DRASC paper. 

## License

Released under the MIT License.
