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

from ..fcr import get_model
import matplotlib.colors as mcolors
import seaborn as sns
from glasbey import create_palette

from ..dataset.dataset import prepare_dataset

def compute_latents(trained_model, datasets, adata, batch_size, sample=False, max_samples=50_000, only_latents=False):
    print("Computing latent representations...")
    indices = []
    ZXs = []
    ZTs = []
    ZXTs = []

    trained_model.eval()
    with torch.no_grad():
        for i, data in enumerate(datasets["loader"]):
            if batch_size * i < max_samples:
                (genes, perts, cf_genes, idx, covariates) = (
                        data[0], data[1], data[2], data[3], data[4:])

                ZX, ZXT, ZT = trained_model.get_latent_presentation(genes, perts, covariates, sample=sample)
                ZXs.extend(ZX)
                ZTs.extend(ZT)
                ZXTs.extend(ZXT)
                indices.extend(idx.tolist())

            else:
                break

    ZXs = [e.detach().cpu().numpy() for e in ZXs]
    ZXs = np.array(ZXs)
    # print("ZX mean:", ZXs.mean(), "ZX std:", ZXs.std())
    ZXTs = [e.detach().cpu().numpy() for e in ZXTs]
    ZXTs = np.array(ZXTs)
    # print("ZXT mean:", ZXTs.mean(), "ZXT std:", ZXTs.std())
    ZTs = [e.detach().cpu().numpy() for e in ZTs]
    ZTs = np.array(ZTs)
    # print("ZT mean:", ZTs.mean(), "ZT std:", ZTs.std())

    if only_latents:
        latent_dic = {
            "ZX": ZXs,
            "ZT": ZTs,
            "ZXT": ZXTs}
        return latent_dic

    # Subset AnnData according to indices
    subset_obs = adata.obs.iloc[indices]
    subset_X = adata.X[indices]
    adata_subset = sc.AnnData(
    X=subset_X,
    obs=subset_obs,
    var=adata.var)

    adata_subset.obsm["ZXs"] = ZXs
    adata_subset.obsm["ZTs"] = ZTs
    adata_subset.obsm["ZXTs"] = ZXTs

    # Export to avoid computing again
    # adata.write(args["data_path"])    

    return adata_subset


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

    norm = mcolors.SymLogNorm(linthresh=0.01, vmin=0, vmax=5)

    # UMAP colored by feature
    figure = sc.pl.umap(
        adata,
        color=feature,
        frameon=False,
        palette="Set3",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "right margin",
        #vcenter=0.01,
        norm=norm,
        show=False,
        ax=ax,
        return_fig=return_fig
    )

    return figure


def umap(adata,
         rep,
         return_fig,
         ax=None,
         n_neighbors=15,
         metric="euclidean", 
         min_dist=0.1,   
         size=10,
         color=["cell_name"],
         palette="Set3",
         legend_loc="right margin",
         title=None
         ):
    
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors, metric=metric)
    sc.tl.umap(adata, min_dist=min_dist)

    norm = mcolors.SymLogNorm(linthresh=0.01, vmin=0, vmax=5)

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
        show=False,
        ax=ax,  # allows plotting into an existing axis
        return_fig=return_fig,  # Return if we are using ax = None
        #vcenter=0.01,
        norm=norm
    )

    return figure


def filter_adata(adata, filter_dict):
    """
    Filter an AnnData object based on a filter dictionary.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to filter.
    filter_dict : dict
        Dictionary specifying filters to apply on the data (e.g., {"drug": "DMSO_TF"}).
        
    Returns
    -------
    AnnData
        Filtered AnnData object.
    """
    mask = np.ones(adata.n_obs, dtype=bool)
    filter_suffix = ""
    
    for key, value in filter_dict.items():
        if key not in adata.obs.columns:
            print(f"Warning: '{key}' not found in adata.obs. Skipping this filter.")
            continue
        
        # Handle single value or list of values
        if isinstance(value, (list, tuple)):
            mask &= adata.obs[key].isin(value)
            filter_suffix += f"_{'_'.join(str(v) for v in value)}"
        else:
            mask &= (adata.obs[key] == value)
            filter_suffix += f"_{value}"

    # Subset adata
    n_before = adata.n_obs
    adata_sub = adata[mask].copy()
    n_after = adata_sub.n_obs
    print(f"Applied filters: {filter_dict}")
    print(f"Samples: {n_before} → {n_after} ({n_after/n_before*100:.1f}%)")
    
    return [adata_sub, filter_suffix]


def plot_umaps(model_dir, 
               n_checkpoint=None, 
               plot_raw=False,
               plot_all=False,
               plot_zx=False,
               plot_zxt=False,
               plot_zt=False, 
               target_epoch=None, 
               # drug_dict=None, 
               # cell_dict=None,
               filter_dict=None, 
               all_drugs=False, 
               sample=False, 
               show_figs=False):
    """
    Plot UMAPs for FCR latent representations.
    
    Parameters
    ----------
    model_dir : str
        Path to model directory
    target_epoch : int, optional
        Specific epoch to load
    filter_dict : dict, optional
        Dictionary of filters to apply to adata before plotting.
        Usage: mainly to select drugs if there are several
        Format: {column_name: value} or {column_name: [list_of_values]}
        Examples:
            {'Agg_Treatment': 'Trametinib'}  # Only Trametinib samples
            {'Agg_Treatment': ['Trametinib', 'Dabrafenib']}  # Multiple drugs
            {'cell_name': 'A375', 'dose': 10}  # Multiple conditions
    """

    # Define plotting mode
    if not plot_zx and not plot_zxt and not plot_zt:
        plot_all = True

    args, model, datasets= get_model(model_dir, target_epoch)
    splits = datasets[0]
    adata = datasets[1].adata

    # Set output directory
    # Can specify model checkpoint index
    if n_checkpoint is not None:
        output_dir = str(os.path.join(model_dir, "umaps", f"checkpoint_{n_checkpoint}"))
    else:
        output_dir = str(os.path.join(model_dir, "umaps"))
    os.makedirs(output_dir, exist_ok=True)

    # Set batch size to determine number of samples to process
    batch_size = 256
    try:
        batch_size = model.batch_size
    except:
        pass

    if filter_dict is None:
        adata = compute_latents(model, splits, adata, batch_size, sample=sample)
    # If we are filtering, do not cap the amount of samples
    else:
        adata = compute_latents(model, splits, adata, batch_size, sample=sample, max_samples=np.inf)

    # Apply filters if provided
    """
    if drug_dict is not None:
        adata_drug, drug_suffix = filter_adata(adata, drug_dict)
    else:
        drug_suffix = ""
        adata_drug = adata

    if cell_dict is not None:
        adata_cell, cell_suffix = filter_adata(adata, cell_dict)
    else:
        cell_suffix = ""
        adata_cell = adata
    """
    if filter_dict is not None:
        adata_sub, filter_suffix = filter_adata(adata, filter_dict)
        # The following does not work for perturbation_input="ohe", because the size of
        # the treatment embeddings depends on the treatments present in the dataset 
        #new_datasets = prepare_dataset(
        #    args,
        #    data_path=None,
        #    split_name="all",
        #    adata=adata_sub
        #)
        #new_splits = new_datasets[0]
    else:
        adata_sub = adata
        #new_splits = splits
        filter_suffix = ""

    # Generate palette
    n = adata.obs["cell_name"].nunique()
    palette = create_palette(palette_size=n)

    # Plot ZX
    if plot_all or plot_zx:
        print("Plotting ZX UMAP...")
        fig = umap(adata_sub, rep="ZXs", color=["cell_name", "Agg_Treatment", "dose"], palette=palette, return_fig=True)
        if show_figs:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir,f"UMAP_ZXs{filter_suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Plot ZXT
    if plot_all or plot_zxt:
        print("Plotting ZXT UMAP...")
        fig = umap(adata_sub, rep="ZXTs", color=["cell_name", "Agg_Treatment", "dose"], return_fig=True)
        if show_figs:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir,f"UMAP_ZXTs{filter_suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Plot ZT
    if plot_all or plot_zt:
        if not all_drugs:
            print("Plotting ZT UMAP...")
            fig = umap(adata_sub, rep="ZTs", color=["dose", "Agg_Treatment", "cell_name"], return_fig=True)
            if show_figs:
                plt.show()
            else:
                fig.savefig(os.path.join(output_dir,f"UMAP_ZTs{filter_suffix}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
        # Option: print ZT against all drugs
        else:
            drugs_plot = [drug for drug in adata.obs["Agg_Treatment"].unique() if drug != "DMSO_TF"]
            for drug in drugs_plot:
                print(f"Plotting ZT UMAP for drug: {drug}...")
                filter_dict = {"Agg_Treatment": [drug, "DMSO_TF"]}
                adata_sub, filter_suffix = filter_adata(adata, filter_dict)
                fig = umap(adata_sub, rep="ZTs", color=["dose", "Agg_Treatment", "cell_name"], return_fig=True)
                if show_figs:
                    plt.show()
                else:
                    fig.savefig(os.path.join(output_dir,f"UMAP_ZTs_{filter_suffix}.png"), dpi=300, bbox_inches="tight")
                plt.close()

    # Plot before FCR
    if plot_raw:
        print("Plotting raw UMAPs...")
        fig = raw_umap(adata_sub, feature="cell_name")
        if show_figs:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir,f"UMAP_cell_name{filter_suffix}.png"), dpi=300, bbox_inches="tight")

        fig = raw_umap(adata_sub, feature="Agg_Treatment")
        if show_figs:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir,f"UMAP_treatment{filter_suffix}.png"), dpi=300, bbox_inches="tight")

        fig = raw_umap(adata_sub, feature="dose")
        if show_figs:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir,f"UMAP_dose{filter_suffix}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_progression(model_dir, rep, feature, sample=False, last_epoch=None, freq=50, n_cols=5):
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
    args, model, datasets = get_model(model_dir, target_epoch=None)
    splits = datasets[0]
    adata = datasets[1].adata
    # adata = sc.read(args["data_path"], backed="r") # Backed mode to avoid memory issues

    for i, ax in enumerate(axes):
        if target_epoch > last_epoch:
            break

        # Plot state before FCR
        if i == 0:
            print("Plotting UMAP before FCR...")
            f = raw_umap(adata,feature=feature, ax=ax, return_fig=False)

        else:
            retrieved = get_model(model_dir, target_epoch)
            model  = retrieved[1]
            # Set batch size to determine number of samples to process
            batch_size = 256
            try:
                batch_size = model.batch_size
            except:
                pass

            adata = compute_latents(model, splits, adata, batch_size, sample=sample)

            f = umap(adata, rep=rep, return_fig=False, ax=ax, title=f"Epoch {target_epoch}")

            target_epoch += freq
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return


