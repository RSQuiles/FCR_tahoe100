import json
import matplotlib.pyplot as plt
import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import argparse
import scanpy as sc
import re
import math

def compute_latents(trained_model, datasets, adata):
    print("Computing latent representations...")
    ZXs = []
    ZTs = []
    ZXTs = []
    for data in datasets["loader_tr"]:
        (genes, perts, cf_genes, cf_perts, covariates) = (
                data[0], data[1], data[2], data[3], data[4:])

        ZX, ZXT, ZT = trained_model.get_latent_presentation(genes, perts, covariates, sample=False)
        ZXs.extend(ZX)
        ZTs.extend(ZT)
        ZXTs.extend(ZXT)

    ZXs = [e.detach().cpu().numpy() for e in ZXs]
    ZXs = np.array(ZXs)
    print("ZX mean:", ZXs.mean(), "ZX std:", ZXs.std())
    ZXTs = [e.detach().cpu().numpy() for e in ZXTs]
    ZXTs = np.array(ZXTs)
    print("ZXT mean:", ZXTs.mean(), "ZXT std:", ZXTs.std())
    ZTs = [e.detach().cpu().numpy() for e in ZTs]
    ZTs = np.array(ZTs)
    print("ZT mean:", ZTs.mean(), "ZT std:", ZTs.std())

    # Append to adata
    adata.obsm["ZXs"] = ZXs
    adata.obsm["ZTs"] = ZTs
    adata.obsm["ZXTs"] = ZXTs

    # Export to avoid computing again
    # adata.write(args["data_path"])    

    return

def raw_umap(adata,
             feature,
             n_comps=50,
             n_neighbors=15,
             min_dist=0.3,
             size=30,
             n_pcs=30,
             ax=None,
             return_fig=True
             ):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    #PCA
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30)
    sc.tl.umap(adata, min_dist=min_dist)

    # UMAP colored by feature
    figure = sc.pl.umap(
        adata,
        color=feature,
        frameon=False,
        palette="Set3",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False,
        ax=ax,
        return_fig=return_fig
    )

    return figure

def umap(adata,
         rep,
         return_fig,
         ax=None,
         n_neighbors=10,
         metric="cosine", 
         min_dist=0.05,   
         size=10,
         color=["cell_name"],
         palette="Set3",
         legend_loc="on data",
         title=None
         ):
    
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(adata, min_dist=min_dist)

    figure = sc.pl.umap(
        adata,
        color=color,
        title=title,
        frameon=False,
        palette=palette,
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc=legend_loc,
        color_map="Blues",
        show=False,
        ax=ax,  # allows plotting into an existing axis
        return_fig=return_fig,  # Return if we are using ax = None
    )

    return figure


def plot_umaps(model_dir, target_epoch=None):
    from ..fcr import get_model

    args, model, datasets= get_model(model_dir, target_epoch)

    output_dir = str(os.path.join(model_dir, "umaps"))
    os.makedirs(output_dir, exist_ok=True)

    ####################################################
    ################# PLOT UMAP RESULTS ################
    ####################################################

    adata = sc.read(args["data_path"])

    # Append latents to adata
    compute_latents(model, datasets, adata)

    # Plot ZX
    print("Plotting ZX UMAP...")
    fig = umap(adata, rep="ZXs", return_fig=True)
    fig.savefig(os.path.join(output_dir,"UMAP_ZXs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # PLot ZXT
    print("Plotting ZXT UMAP...")
    fig = umap(adata, rep="ZXTs", color=["cell_name", "Agg_Treatment"], return_fig=True)
    fig.savefig(os.path.join(output_dir,"UMAP_ZXTs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot ZT
    print("Plotting ZT UMAP...")
    fig = umap(adata, rep="ZTs", color=["cell_name", "Agg_Treatment"], return_fig=True)
    fig.savefig(os.path.join(output_dir,"UMAP_ZTs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot before FCR
    fig = raw_umap(adata, feature="cell_name")
    fig.savefig(os.path.join(output_dir,"UMAP_cell_name.png"), dpi=300, bbox_inches="tight")

    fig = raw_umap(adata, feature="Agg_Treatment")
    fig.savefig(os.path.join(output_dir,"UMAP_treatment.png"), dpi=300, bbox_inches="tight")


def plot_progression(model_dir, rep, feature, last_epoch=None, freq=50, n_cols=5):
    from ..fcr import fetch_latest
    from ..fcr import get_model

    # Initialize target epoch
    target_epoch = 0

    # Determine max epoch
    if last_epoch is None:
        latest_model_path = fetch_latest(model_dir)
        match = re.search(r"epoch=(\d+)", latest_model_path)
        last_epoch = int(match.group(1))

    # Initialize the plot accordingly
    n_subplots = math.floor(last_epoch / freq) + 2 # before FCR and epoch=0
    n_rows = math.ceil(n_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    axes = axes.flatten()
    # print(f"Axes: {axes}")

    # Set output directory
    output_dir = str(os.path.join(model_dir, "umaps"))
    os.makedirs(output_dir, exist_ok=True)

    filename = f"UMAP_progression_{rep}.png"
    save_path = str(os.path.join(output_dir, filename))

    # Import AnnData
    args, model, datasets = get_model(model_dir)
    adata = sc.read(args["data_path"])

    for i, ax in enumerate(axes):
        if target_epoch > last_epoch:
            break

        # Plot state before FCR
        if i == 0:
            print("Plotting UMAP before FCR...")
            f = raw_umap(adata,feature=feature, ax=ax, return_fig=False)

        else:
            retrieved = get_model(model_dir, target_epoch)
            args, model, datasets  = retrieved[0], retrieved[1], retrieved[2]
            compute_latents(model, datasets, adata)

            f = umap(adata, rep=rep, return_fig=False, ax=ax, title=f"Epoch {target_epoch}")

            target_epoch += freq
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return


