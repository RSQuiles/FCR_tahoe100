import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import scanpy as sc
import numpy as np
from scipy import sparse
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad
import os

from typing import Optional, Union

def select_dataset(cell_lines: Union[str, list], 
                   drugs: Union[str, list], 
                   balance_controls: bool = True,
                   balance_lines: bool = False,
                   metadata_path: str = "/cluster/work/bewi/members/rquiles/experiments/datasets/obs_metadata.parquet",
                   data_dir: str = "/cluster/work/bewi/data/tahoe100/h5ad/",
                   ) -> sc.AnnData:
    
    """
    Select and prepare dataset based on provided configuration:
    - USAGE CASE: select specific cell lines and drugs from Tahoe100 dataset
    - Optionally balance controls and cell lines
    """

    # Read arguments
    lines_select = cell_lines if isinstance(cell_lines, list) else [cell_lines]
    control_name = "DMSO_TF"
    drugs_select = drugs if isinstance(drugs, list) else [drugs]
    drugs_select.append(control_name)

    # Use metadata to filter the plates

    print("Inspecting metadata...")
    metadata = pd.read_parquet(metadata_path)

    select = metadata[(metadata["cell_name"].isin(lines_select)) & (metadata["drug"].isin(drugs_select))]
    print(f"Selected {select.shape[0]} samples from {len(lines_select)} cell lines and {len(drugs_select)} drugs (plus controls)")

    ## SUBSET TO BALANCE THE NUMBER OF CONTROLS
    if balance_controls:
        keep_rows = []

        for cell_line in lines_select:
            # Treated (non-control) rows for this cell line
            treated_mask = (
                (select["cell_name"] == cell_line) & (select["drug"] != control_name)
            )
            treated_idx = select.index[treated_mask]
            n_treated = len(treated_idx)
            if n_treated > 0:
                keep_rows.extend(treated_idx)

            # Controls: keep 1/3 of treated count (at most)
            n_controls = n_treated // 3
            control_mask = (
                (select["cell_name"] == cell_line) & (select["drug"] == control_name)
            )
            ctrl_idx = select.index[control_mask][:n_controls]
            if len(ctrl_idx) > 0:
                keep_rows.extend(ctrl_idx)

        # Randomize row order and subset
        random.shuffle(keep_rows)
        select = select.loc[keep_rows]

    ## SUBSET TO BALANCE THE NUMBER OF SAMPLES PER CELL LINE
    if balance_lines:
        n_samples = []
        for cell_line in lines_select:
            mask = select["cell_name"] == cell_line
            row_indexes = select[mask].index
            n_samples.append(len(row_indexes))

        # Groups can be at most 1.5 the size of the smallest group
        min_samples = round(min(n_samples) * 1.5)
        print(f"N_samples per cell line: {n_samples}")
        print(f"Minimum samples per cell line: {min_samples}")

        keep_rows = []
        # Sample according to the smallest group
        for cell_line in lines_select:
            mask = select["cell_name"] == cell_line
            row_indexes = select[mask].index
            row_indexes = row_indexes[:min_samples]
            if len(row_indexes) > 0:
                keep_rows.extend(row_indexes)
                print(f"Length of {cell_line}: {len(row_indexes)}")
            
        # Randomize row order and subset by row indices (DataFrame)
        random.shuffle(keep_rows)
        select = select.loc[keep_rows]

    # Determine which plates to inspect
    plates = np.unique(select["plate"])

    # Loop through plates and append the matching rows
    final_adata = None

    for i, plate in enumerate(plates):
        print(f"Loading {plate}...")
        data_path = f"/cluster/work/bewi/data/tahoe100/h5ad/{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
        adata = sc.read_h5ad(data_path, backed="r")
        print(f"Subsetting {plate}...")
        
        ids = select[select["plate"] == plate]["BARCODE_SUB_LIB_ID"].values
        subset = adata[adata.obs_names.isin(ids), :].to_memory()
        # Initialize or append final AnnData
        if final_adata is None:
            final_adata = subset
        else:
            final_adata = ad.concat([final_adata, subset], join="outer")

    # Ensure we found at least some cells across selected plates
    if final_adata is None or final_adata.n_obs == 0:
        raise RuntimeError(
            "No matching cells were found for the requested cell lines/drugs across the selected plates. "
            "Check that BARCODE_SUB_LIB_ID values exist in the plate .h5ad files and that filters are not too strict."
        )

    # Randomize the rows in final_adata
    n = final_adata.n_obs  # number of cells
    perm = np.random.permutation(n)
    adata = final_adata[perm, :]

    return adata


def preprocess_adata(adata: sc.AnnData,
                        n_genes: int = 5000,
                        tahoe_plate: bool = True,
                        normalize_total: bool = True,
                        log_transform: bool = True,
                        scale_data: bool = False,
                        ) -> sc.AnnData:
    """
    Preprocess AnnData:
    - Modify columns for FCR method compatibility
    - Normalize total counts per cell
    - Log-transform data
    - Scale data to zero mean and unit variance
    - Select highly variable genes
    """

    ## UPDATE COLUMNS
    if tahoe_plate:
        control_name = "DMSO_TF"
        adata.obs["control"] = adata.obs["drug"] == control_name
        adata.obs["control"] = adata.obs["control"].astype(int)
        adata.obs["dose"] = adata.obs["drugname_drugconc"].str.split(",").str[1].astype(float)

    # Normalize total counts per cell
    if normalize_total:
        print("Normalizing total counts per cell...")
        sc.pp.normalize_total(adata, target_sum=1e4)

    # Log-transform data
    if log_transform:
        print("Log-transforming data...")
        sc.pp.log1p(adata)

    # Identify highly variable genes
    print(f"Selecting top {n_genes} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=True, flavor="seurat_v3")

    # Scale data to zero mean and unit variance
    if scale_data:
        print("Scaling data to zero mean and unit variance...")
        sc.pp.scale(adata, max_value=10)

    return adata


def write_adata(export_dir: str,
                export_name: str,
                adata: sc.AnnData) -> None:
    """
    Write AnnData to specified directory with given name.
    Ensures the export directory exists.
    """
    os.makedirs(export_dir, exist_ok=True)
    if not export_name.endswith(".h5ad"):
        export_name = f"{export_name}.h5ad"
    export_path = os.path.join(export_dir, export_name)
    print(f"Writing AnnData to {export_path}...")
    adata.write_h5ad(export_path)