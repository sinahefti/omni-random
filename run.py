# Adapted from sta426hs2026/omni-pipelines-py (run.py)
# Original author: Ilaria Billato (https://github.com/billila)
# Modified by: Tom Kitak (Omnibenchmark compatibility)
# License: MIT
# This version replaces the scanpy clustering with random cluster labels.

import os
import argparse
import json

import numpy as np
import pandas as pd
import scanpy as sc

# Parse command line arguments
parser = argparse.ArgumentParser(description="Random clustering baseline")
parser.add_argument(
    "--output_dir", "-o", type=str, required=True, help="Output directory"
)
parser.add_argument("--name", "-n", type=str, required=True, help="Dataset name")
parser.add_argument(
    "--data.h5ad", dest="data_h5ad", type=str, required=True, help="Input h5ad file"
)
parser.add_argument(
    "--data.clusters_truth_num",
    dest="clusters_truth_num_path",
    type=str,
    required=True,
    help="input true number of clusters path",
)
parser.add_argument("--method_name", type=str, default="random", help="Method to run")
parser.add_argument(
    "--d_cluster",
    type=int,
    default=0,
    help="delta number of clusters with respect to true number of clusters",
)
parser.add_argument("--n_comp", type=int, default=50, help="number of PCA components")
parser.add_argument("--n_neig", type=int, default=15, help="unused, kept for compatibility")
parser.add_argument("--n_hvg", type=int, default=1000, help="number of highly variable genes")
parser.add_argument("--filter", type=str, default="manual", help="unused, kept for compatibility")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args, _ = parser.parse_known_args()

# time object to store time involved (in seconds) in each step
timings = {
    "gpu_load": None,
    "find_mit_gene": None,
    "filter": None,
    "normalization": None,
    "hvg": None,
    "scaling": None,
    "pca": None,
    "t_sne": None,
    "umap": None,
    "louvain": None,
    "leiden": None,
}

clustering_info = {
    "resolutions": {"louvain": None, "leiden": None},
    "num_runs": {"louvain": None, "leiden": None},
}

# data ####
adata = sc.read_h5ad(args.data_h5ad)
adata.var_names_make_unique()
with open(args.clusters_truth_num_path, "r") as f:
    clusters_truth_num = int(f.read())
n_cluster = clusters_truth_num + args.d_cluster

# the "method": random cluster labels, reproducible through the seed ####
np.random.seed(args.seed)

labels = np.random.randint(1, n_cluster + 1, size=adata.n_obs).astype(str)
adata.obs["louvain"] = labels
adata.obs["leiden"] = labels

# the code below expects PCA and highly variable genes to exist
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
sc.pp.pca(adata, n_comps=args.n_comp, use_highly_variable=True)

# Save timings as JSON
with open(os.path.join(args.output_dir, f"{args.name}.timings.json"), "w") as f:
    json.dump(timings, f, indent=2)

# Save clustering metadata as JSON
with open(os.path.join(args.output_dir, f"{args.name}.clustering_info.json"), "w") as f:
    json.dump(clustering_info, f, indent=2)

# Save cluster assignments as TSV
adata.obs[["louvain", "leiden"]].reset_index(names="cell_id").to_csv(
    os.path.join(args.output_dir, f"{args.name}.clusters.tsv"),
    sep="\t",
    index=False,
)

hvg_list = adata.var.highly_variable.index.to_series()
hvg_list.to_csv(
    os.path.join(args.output_dir, f"{args.name}.hvgs.tsv"),
    sep="\t",
    index=False,
    header=False,
)

# Save PCA coordinates as TSV
pca_df = pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[f"PC{i + 1}" for i in range(adata.obsm["X_pca"].shape[1])],
)
pca_df.to_csv(
    os.path.join(args.output_dir, f"{args.name}.pca.tsv"),
    sep="\t",
    index=True,
    index_label="cell_id",
)
