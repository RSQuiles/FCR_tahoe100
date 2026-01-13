from typing import Union
import scipy
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path
import time

import torch

from ..utils.general_utils import unique_ind
from ..utils.data_utils import rank_genes_groups

import warnings
warnings.filterwarnings("ignore")

import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

def get_dataset_features(data_path: str,
            perturbation_key="perturbation",
            perturbation_input="ohe",
            control_key="control",
            covariate_keys="covariates",
            ):

    # Load AnnData in backed mode
    adata = sc.read_h5ad(data_path, backed='r')

    control_names = np.unique(adata[adata.obs[control_key] == 1].obs[perturbation_key])
    var_names = adata.var_names
    pert_unique = np.array(get_unique_perts(adata, perturbation_key))

    # num_treatments := size of the perturbation embedding
    if perturbation_input == "ohe":
        num_treatments = len(pert_unique)
    elif perturbation_input in ["chemberta", "morgan", "maccs"]:
        # Load drug metadata to get embedding size
        drug_metadata_path = "/cluster/work/bewi/data/tahoe100/metadata/drug_metadata.parquet"
        drug_df = pd.read_parquet(drug_metadata_path)
        sample_drug = pert_unique[0]
        if sample_drug not in drug_df["drug"].values:
            raise ValueError(f"Drug {sample_drug} not found in drug metadata for embedding size inference.")
        if perturbation_input == "chemberta":
            embedding_size = len(drug_df.loc[drug_df["drug"] == sample_drug, "chemberta"].values[0])
        elif perturbation_input == "morgan":
            embedding_size = len(drug_df.loc[drug_df["drug"] == sample_drug, "morgan_fp"].values[0])
        elif perturbation_input == "maccs":
            embedding_size = len(drug_df.loc[drug_df["drug"] == sample_drug, "maccs_fp"].values[0])
        num_treatments = embedding_size

    num_outcomes = adata.n_vars

    if not isinstance(covariate_keys, list):
        covariate_keys = [covariate_keys]

    if covariate_keys is not None:
        if not len(covariate_keys) == len(set(covariate_keys)):
            raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
        covars_dict = {}
        num_covariates = []
        cov_names = []
        for cov in covariate_keys:
            values = adata.obs[cov].astype(str).values
            cov_names.append(values)

            names = np.unique(values)
            num_covariates.append(len(names))

            names_idx = torch.arange(len(names)).unsqueeze(-1)
            covars_dict[cov] = dict(
                zip(list(names), names_idx)
            )
        cov_names = pd.Series(["_".join(c) for c in zip(*cov_names)]).astype(str).values
        cov_names_unique = np.unique(cov_names)
    else:
        raise NotImplementedError("Covariate keys are required for counterfactual sampling")

    controls = adata.obs[control_key].astype(str).values
    cov_control = cov_names + "_" + controls
    cov_control_idx = unique_ind(cov_control)

    # Counterfactual gene indices
    cf_genemap = {} # map from covariate names to genes
    cf_pert_dose_name = control_names[0]
    control_vals = '1'

    for covariate_name in cov_names_unique:
        cf_name = covariate_name + f"_{control_vals}"

        # Get counterfactual genes (from control)
        cf_inds = cov_control_idx[cf_name]
        cf_i = np.random.choice(cf_inds)
        row = adata.X[cf_i, :]           # load 1 row from disk
        if scipy.sparse.issparse(row):
            row = row.toarray().squeeze()
        cf_genes = torch.from_numpy(row).float()

        # Add to dictionary
        cf_genemap[covariate_name] = cf_genes

    return [control_names,
            var_names,
            pert_unique,
            covars_dict,
            cf_genemap,
            num_treatments,
            num_outcomes,
            num_covariates
           ]


def get_unique_perts(adata, perturbation_key="perturbation"):
    all_perts = adata.obs[perturbation_key].values
    perts = [i for p in all_perts for i in p.split("+")]
    return list(dict.fromkeys(perts))

    

class Dataset:
    def __init__(
        self,
        data,
        args,
        control_names,
        var_names,
        pert_unique,
        covars_dict,
        cf_genemap,
        perturbation_key="perturbation",
        control_key="control",
        dose_key="dose",
        covariate_keys="covariates",
        split_key="split",
        test_ratio=0.2,
        random_state=42,
        sample_cf=False,
        cf_samples=20,
        perturbation_input="ohe",
        control_name= None,
        embedded_dose= None,
        drug_metadata_path = "/cluster/work/bewi/data/tahoe100/metadata/drug_metadata.parquet"
    ):
        
        # Measure time to load dataset
        start = time.time()

        # Load AnnData
        data_path = Path(data) 
        # print("Reading AnnData...")
        self.adata = sc.read(data_path)

        self.sample_cf = sample_cf
        self.cf_samples = cf_samples

        # MODIFIED: INCREASE FLEXIBILITY AND ADAPT TO HUGE DATASETS (millions of samples)

        # Fields
        # perturbation
        assert perturbation_key in self.adata.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"

        # control
        if control_key not in self.adata.obs.columns:
            if control_name is not None:
                print(f"Adding control column based on control name: {control_name}...")
                self.adata.obs[control_key] = (self.adata.obs[perturbation_key] == control_name).astype(int)
            else:
                raise ValueError(f"Control {control_key} is missing in the provided adata and no control_name was given.")
        
        # dose
        if dose_key is None:
            print("Adding a dummy dose...")
            self.adata.obs["dummy_dose"] = 1.0
            dose_key = "dummy_dose"
        elif dose_key not in self.adata.obs.columns:
            if embedded_dose is not None:
                self.adata.obs[dose_key] = self.adata.obs[embedded_dose].str.split(",").str[1].astype(float)
            else:
                raise ValueError(f"Dose {dose_key} is missing in the provided adata and no embedded_dose column was given.")

        # covariates
        if covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            self.adata.obs["dummy_covar"] = "dummy-covar"
            covariate_keys = ["dummy_covar"]
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in self.adata.obs.columns, f"Covariate {key} is missing in the provided adata"

        # split
        if split_key is None or split_key not in self.adata.obs.columns:
            # print(f"Performing automatic train-test split with {test_ratio} ratio.")
            from sklearn.model_selection import train_test_split

            self.adata.obs["split"] = "train"
            idx_train, idx_test = train_test_split(
                self.adata.obs_names, test_size=test_ratio, random_state=random_state
            )
            self.adata.obs["split"].loc[idx_train] = "train"
            self.adata.obs["split"].loc[idx_test] = "test"
            split_key = "split"
        else:
            assert split_key in self.adata.obs.columns, f"Split {split_key} is missing in the provided adata"

        # Store keys
        self.perturbation_key = perturbation_key
        self.perturbation_input = perturbation_input
        self.control_key = control_key
        self.dose_key = dose_key
        self.covariate_keys = covariate_keys
        self.split_key = split_key

        # Vectorized categorical conversion for metadata
        keys = [perturbation_key, dose_key, control_key, split_key]
        keys.extend(covariate_keys if covariate_keys is not None else [])
        for key in keys:
            if key in self.adata.obs.columns:
                self.adata.obs[key] = self.adata.obs[key].astype("category")

        # Precompute useful columns as NumPy arrays
        self.pert_names = self.adata.obs[perturbation_key].astype(str).values
        self.doses = self.adata.obs[dose_key].astype(str).values
        self.controls = self.adata.obs[control_key].astype(str).values
        self.control_names = control_names

        self.var_names = var_names

        n = len(self.adata.obs)

        self.indices = {
            "all": np.arange(n),
            "control": np.where(self.adata.obs[control_key].cat.codes == 1)[0],
            "treated": np.where(self.adata.obs[control_key].cat.codes != 1)[0],
            "train": np.where(self.adata.obs[split_key] == "train")[0],
            "test": np.where(self.adata.obs[split_key] == "test")[0],
            "ood": np.where(self.adata.obs[split_key] == "ood")[0],
        }

        # PERTURBATIONS

        if self.perturbation_input == "ohe":
            # Using custom OHE for perturbations
            # store as attribute for molecular featurisation
            pert_unique_onehot = torch.eye(len(pert_unique))
            self.perts_dict = dict(
                zip(pert_unique, pert_unique_onehot)
            )
            # get perturbation combinations
            perturbations = []
            for i, comb in enumerate(self.pert_names):
                perturbation_combos = [self.perts_dict[p] for p in comb.split("+")]
                dose_combos = str(self.adata.obs[dose_key].values[i]).split("+")
                perturbation_ohe = []
                for j, d in enumerate(dose_combos):
                    perturbation_ohe.append(float(d) * perturbation_combos[j])
                perturbations.append(sum(perturbation_ohe))

            self.perturbations = torch.stack(perturbations)

        elif self.perturbation_input == "chemberta":
            print("Using ChemBERTa embeddings for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "chemberta"].values[0])
                                        for pert in self.pert_names])
            # Multiply by dose to generate unique embeddings for each drug-dosage combination
            doses = pd.to_numeric(self.adata.obs[dose_key], errors="coerce").fillna(0).values.astype(np.float32)
            dose_tensor = torch.from_numpy(doses).float()
            self.perturbations = perturbations * dose_tensor.unsqueeze(1)
            
        elif self.perturbation_input == "morgan":
            print("Using Morgan fingerprints for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "morgan_fp"].values[0])
                                        for pert in self.pert_names])
            # Multiply by dose to generate unique embeddings for each drug-dosage combination
            doses = pd.to_numeric(self.adata.obs[dose_key], errors="coerce").fillna(0).values.astype(np.float32)
            dose_tensor = torch.from_numpy(doses).float()
            self.perturbations = perturbations * dose_tensor.unsqueeze(1)

        elif self.perturbation_input == "maccs":
            print("Using MACCS keys for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "maccs_fp"].values[0])
                                        for pert in self.pert_names])
            # Multiply by dose to generate unique embeddings for each drug-dosage combination
            doses = pd.to_numeric(self.adata.obs[dose_key], errors="coerce").fillna(0).values.astype(np.float32)
            dose_tensor = torch.from_numpy(doses).float()
            self.perturbations = perturbations * dose_tensor.unsqueeze(1)
                        
        else:
            raise NotImplementedError("Unmatched input mode for treatments")


        # COVARIATES
        self.covars_dict = covars_dict

        if covariate_keys is not None:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            cov_names = []
            self.covariates = []
            self.num_covariates = []
            for cov in covariate_keys:
                values = self.adata.obs[cov].astype(str).values
                cov_names.append(values)

                self.covariates.append(
                    torch.stack([self.covars_dict[cov][v] for v in values])
                )
            self.cov_names = pd.Series(["_".join(c) for c in zip(*cov_names)]).astype(str).values
        else:
            self.cov_names = np.array([""] * len(data), dtype=str)
            self.covariates = None

        # GENES
        # Directly access gene expression from AnnData object
        genes = self.adata.X
        if scipy.sparse.issparse(genes):
            genes = genes.toarray().squeeze()
        self.genes = torch.from_numpy(genes).float()

        self.n_obs = self.adata.n_obs

        self.pert_dose = self.pert_names + "_" + self.doses
        self.cov_pert = self.cov_names + "_" + self.pert_names
        self.cov_pert_dose = self.cov_names + "_" + self.pert_dose
        self.cov_control = self.cov_names + "_" + self.controls

        self.num_treatments = args["num_treatments"]
        self.num_covariates = args["num_covariates"]
        self.num_covariates = args["num_covariates"]

        self.cf_genemap = cf_genemap

        # Time to load dataset
        # print(f"Dataset load has elapsed: {start - time.time():.2f} seconds.")

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __len__(self):
        return self.n_obs


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    We just save a reference to the original Dataset for more speed
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset  # Keep reference to parent
        self.indices = indices  # Store which indices this subset uses
        self.n_obs = len(indices)

        self.control_vals = '1'
        
        if self.dataset.sample_cf:
            self.cov_pert_dose_idx = unique_ind(self.dataset.cov_pert_dose)
            # self.cov_control_idx = unique_ind(self.dataset.cov_control)

    def subset_condition(self, control=True):
        raise NotImplementedError("Not implemented yet.")

    def __getitem__(self, i):

        ### get gene activations
        parent_idx = self.indices[i]
        genes = self.dataset.genes[parent_idx]
        
        ### get the control activations for this sample
        covariate_name = indx(self.dataset.cov_names, parent_idx)
        cf_genes = self.dataset.cf_genemap[covariate_name]

        return (
            genes,
            indx(self.dataset.perturbations, parent_idx),
            cf_genes,
            parent_idx, # ADDED: AnnData row indexes corresponding to each sample
            *[indx(cov, parent_idx) for cov in self.dataset.covariates]
        )

    def __len__(self):
        return self.n_obs
    
    
# LEGACY CODE: not used in currrent implementationM
def load_dataset_splits(
    data_path: str,
    perturbation_key: str = "perturbation",
    control_key: str = "control",
    dose_key: str = "dose",
    covariate_keys: Union[list, str] = "covariates",
    split_key: str = "split",
    sample_cf: bool = False,
    return_dataset: bool = False,
):

    dataset = Dataset(
        data_path, perturbation_key, control_key, dose_key, covariate_keys, split_key, 
        sample_cf=sample_cf
    )

    splits = {
        "train": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("ood", "all"),
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits
    
    
def load_dataset_train_test(
    data_path: str,
    args: dict,
    features: dict,
    perturbation_key: str = "Agg_Treatment",
    perturbation_input: str = "ohe",
    control_key: str = "control",
    dose_key: str = "dose",
    covariate_keys: Union[list, str] = "covariates",
    split_key: str = "split",
    control_name: str = None,
    embedded_dose: str = None,
    sample_cf: bool = False,
    return_dataset: bool = False,
):

    dataset = Dataset(
        data_path,
        args,
        features["control_names"],
        features["var_names"],
        features["pert_unique"],
        features["covars_dict"],
        features["cf_genemap"],
        perturbation_key, 
        control_key, 
        dose_key, 
        covariate_keys, 
        split_key, 
        sample_cf=sample_cf, 
        control_name=control_name, 
        embedded_dose=embedded_dose,
        perturbation_input=perturbation_input, 
    )

    start_split = time.time()
    splits = {
        "train": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("test", "all"),
        ## modified: returns whole dataset for testing and visualization
        "all": dataset.subset("all","all")
    }
    # print(f"Dataset split has elapsed: {time.time() - start_split} seconds.")

    if return_dataset:
        return splits, dataset
    else:
        return splits        

indx = lambda a, i: a[i] if a is not None else None
