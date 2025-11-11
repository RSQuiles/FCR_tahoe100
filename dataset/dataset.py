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


class Dataset:
    def __init__(
        self,
        data,
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
        args = None,
        drug_metadata_path = "/cluster/work/bewi/data/tahoe100/metadata/drug_metadata.parquet",
    ):
        
        # Measure time to load dataset
        start = time.time()

        if type(data) == str:
            # MODIFIED: Only load metadata (light) and read expression data on demand during training
            data_path = Path(data) 
            print("Reading AnnData...")
            self.adata = sc.read(data_path, backed="r")
            # Cache references
            self.obs = self.adata.obs

        self.sample_cf = sample_cf
        self.cf_samples = cf_samples

        # MODIFIED: INCREASE FLEXIBILITY AND ADAPT TO HUGE DATASETS (millions of samples)

        # Fields
        # perturbation
        assert perturbation_key in self.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"

        # control
        if control_key not in self.obs.columns:
            if control_name is not None:
                print(f"Adding control column based on control name: {control_name}...")
                self.obs[control_key] = (self.obs[perturbation_key] == control_name).astype(int)
            else:
                raise ValueError(f"Control {control_key} is missing in the provided adata and no control_name was given.")
        
        # dose
        if dose_key is None:
            print("Adding a dummy dose...")
            self.obs["dummy_dose"] = 1.0
            dose_key = "dummy_dose"
        elif dose_key not in self.obs.columns:
            if embedded_dose is not None:
                self.obs[dose_key] = self.obs[embedded_dose].str.split(",").str[1].astype(float)
            else:
                raise ValueError(f"Dose {dose_key} is missing in the provided adata and no embedded_dose column was given.")

        # covariates
        if covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            self.obs["dummy_covar"] = "dummy-covar"
            covariate_keys = ["dummy_covar"]
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in self.obs.columns, f"Covariate {key} is missing in the provided adata"

        # split
        if split_key is None or split_key not in self.obs.columns:
            print(f"Performing automatic train-test split with {test_ratio} ratio.")
            from sklearn.model_selection import train_test_split

            self.obs["split"] = "train"
            idx_train, idx_test = train_test_split(
                self.adata.obs_names, test_size=test_ratio, random_state=random_state
            )
            self.obs["split"].loc[idx_train] = "train"
            self.obs["split"].loc[idx_test] = "test"
            split_key = "split"
        else:
            assert split_key in self.obs.columns, f"Split {split_key} is missing in the provided adata"

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
            if key in self.obs.columns:
                self.obs[key] = self.obs[key].astype("category")

        # Precompute useful columns as NumPy arrays
        self.pert_names = self.obs[perturbation_key].astype(str).values
        self.doses = self.obs[dose_key].astype(str).values
        self.controls = self.obs[control_key].astype(str).values
        self.control_names = np.unique(
            self.adata[self.obs[self.control_key] == 1].obs[self.perturbation_key]
        )

        self.var_names = self.adata.var_names

        n = len(self.obs)

        self.indices = {
            "all": np.arange(n),
            "control": np.where(self.obs[control_key].cat.codes == 1)[0],
            "treated": np.where(self.obs[control_key].cat.codes != 1)[0],
            "train": np.where(self.obs[split_key] == "train")[0],
            "test": np.where(self.obs[split_key] == "test")[0],
            "ood": np.where(self.obs[split_key] == "ood")[0],
        }
        
        # GENES
        # Modified: read gene expression from precomputed .npy file
        genes_path = data_path.with_name("genes.npy")
        self.genes = np.load(genes_path, mmap_mode='r')
        # self.genes = np.load(genes_path) # load into memory
        print("Finished loading genes.npy file...")

        # PERTURBATIONS
        # get unique perturbations
        pert_unique = np.array(self.get_unique_perts())

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
            self.num_treatments = len(pert_unique) # treatment input dimension

        elif self.perturbation_input == "chemberta":
            print("Using ChemBERTa embeddings for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            self.perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "chemberta"].values[0])
                                        for pert in self.pert_names])
            
        elif self.perturbation_input == "morgan":
            print("Using Morgan fingerprints for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            self.perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "morgan_fp"].values[0])
                                        for pert in self.pert_names])
            
        elif self.perturbation_input == "maccs":
            print("Using MACCS keys for perturbations!")
            drug_df = pd.read_parquet(drug_metadata_path)
            self.perturbations = torch.stack([torch.tensor(drug_df.loc[drug_df["drug"] == pert, "maccs_fp"].values[0])
                                        for pert in self.pert_names])
            
        else:
            raise NotImplementedError("Unmatched input mode for treatments")


        # COVARIATES
        if covariate_keys is not None:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            cov_names = []
            self.covars_dict = {}
            self.covariates = []
            self.num_covariates = []
            for cov in covariate_keys:
                values = self.obs[cov].astype(str).values
                cov_names.append(values)

                names = np.unique(values)
                self.num_covariates.append(len(names))

                names_idx = torch.arange(len(names)).unsqueeze(-1)
                self.covars_dict[cov] = dict(
                    zip(list(names), names_idx)
                )

                self.covariates.append(
                    torch.stack([self.covars_dict[cov][v] for v in values])
                )
            # self.cov_names = np.array(["_".join(c) for c in zip(*cov_names)], dtype=str)
            self.cov_names = pd.Series(["_".join(c) for c in zip(*cov_names)]).astype(str).values
        else:
            self.cov_names = np.array([""] * len(data), dtype=str)
            self.covars_dict = None
            self.covariates = None
            self.num_covariates = None

        self.num_outcomes = self.adata.n_vars
        self.n_obs = self.adata.n_obs

        self.pert_dose = self.pert_names + "_" + self.doses
        self.cov_pert = self.cov_names + "_" + self.pert_names
        self.cov_pert_dose = self.cov_names + "_" + self.pert_dose
        self.cov_control = self.cov_names + "_" + self.controls

        # Time to load dataset
        print(f"Dataset load has elapsed: {start - time.time():.2f} seconds.")

    def get_unique_perts(self, all_perts=None):
        if all_perts is None:
            all_perts = self.pert_names
        perts = [i for p in all_perts for i in p.split("+")]
        return list(dict.fromkeys(perts))

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        # return SubDataset(self, idx)
        return SubDataset_Pair(self, idx)

    def __len__(self):
        return self.n_obs


class SubDataset_Pair:
    """
    Memory-efficient subset of Dataset.
    It does NOT copy any data — it only stores index references.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset  # Keep reference to parent
        self.indices = np.array(indices, dtype=np.int64)  # store as numpy array
        self.sample_cf = dataset.sample_cf
        self.cf_samples = dataset.cf_samples

        self.perturbation_key = dataset.perturbation_key
        self.perturbation_input = dataset.perturbation_input
        self.control_key = dataset.control_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys

        self.control_names = dataset.control_names

        if self.perturbation_input == "ohe":
            self.perts_dict = dataset.perts_dict
        self.covars_dict = dataset.covars_dict

        # Keep references to parent dataset
        self.genes = dataset.genes
        self.perturbations = dataset.perturbations
        self.controls = dataset.controls
        self.covariates = dataset.covariates
        self.pert_names = dataset.pert_names
        self.doses = dataset.doses
        self.cov_names = dataset.cov_names
        self.cov_pert = dataset.cov_pert
        self.pert_dose = dataset.pert_dose
        self.cov_pert_dose = dataset.cov_pert_dose
        self.cov_control = dataset.cov_control
        self.control_vals = '1'
        self.var_names = dataset.var_names
        self.num_covariates = dataset.num_covariates
        self.num_outcomes = dataset.num_outcomes
        self.num_treatments = dataset.num_treatments

        if self.sample_cf:
            self.cov_control_idx = unique_ind(self.cov_control)

        self.n_obs = len(self.indices)

    # Standard PyTorch Dataset interface
    def __len__(self):
        return self.n_obs

    def __getitem__(self, i):

        # Genes loaded on demand from disk

        # Lookup real index in parent dataset
        parent_idx = self.indices[i]

        # Fetch treated genes
        genes = torch.as_tensor(self.genes[parent_idx])

        cf_genes = None
        cf_i = 0

        # Genes loaded in RAM
        if self.sample_cf:
            cov_name = self.cov_names[parent_idx]
            cf_name = f"{cov_name}_{self.control_vals}"
            if cf_name in self.cov_control_idx:
                cf_candidates = self.cov_control_idx[cf_name]
                cf_i = np.random.choice(cf_candidates)
                cf_genes = torch.as_tensor(self.genes[cf_i])

        if self.sample_cf:
            cov_name = self.cov_names[parent_idx]
            cf_name = f"{cov_name}_{self.control_vals}"
            if cf_name in self.cov_control_idx:
                cf_candidates = self.cov_control_idx[cf_name]
                cf_i = np.random.choice(cf_candidates)
                cf_genes = torch.as_tensor(self.genes[cf_i])

        return (
            genes,
            self.perturbations[parent_idx],
            cf_genes,
            parent_idx,
            cf_i,
            *[cov[parent_idx] for cov in self.covariates]
        )

    # Subset to return SubDataset
    def subset_condition(self, control=True):
        if control is None:
            return self
        else:
            idx = np.where(self.controls == control)[0].tolist()
            return SubDataset(self, idx)

class SubDataset:
    """
    Subsets a `SubDatasetPair` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.indices = np.array(indices, dtype=np.int64)
        self.parent_dataset = dataset

        self.sample_cf = dataset.sample_cf
        self.cf_samples = dataset.cf_samples

        self.perturbation_key = dataset.perturbation_key
        self.perturbation_input = dataset.perturbation_input
        self.control_key = dataset.control_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.control_names = dataset.control_names
        self.covars_dict = dataset.covars_dict

        if self.perturbation_input == "ohe":
            self.perts_dict = dataset.perts_dict

        # Obtain gene expression data
        adata_indices = np.asarray(self.parent_dataset.indices)[indices]
        adata_X = self.parent_dataset.dataset.adata.X
        genes = adata_X[adata_indices, :]
        if scipy.sparse.issparse(genes):
            genes = genes.toarray().squeeze()

        genes = torch.from_numpy(genes)
        self.genes = dataset.genes[indices]

        self.perturbations = indx(dataset.perturbations, indices)
        self.controls = dataset.controls[indices]
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.pert_names = indx(dataset.pert_names, indices)
        self.doses = indx(dataset.doses, indices)

        self.cov_names = indx(dataset.cov_names, indices)
        self.cov_pert = indx(dataset.cov_pert, indices)
        self.pert_dose = indx(dataset.pert_dose, indices)
        self.cov_pert_dose = indx(dataset.cov_pert_dose, indices)
        # self.cov_control = indx(dataset.cov_control, indices)

        self.var_names = dataset.var_names
        self.n_obs = len(indices)
        ## modified: commented out if de_genes not used downstream
        #self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_outcomes = dataset.num_outcomes
        self.num_treatments = dataset.num_treatments

        if self.sample_cf:
            self.cov_pert_dose_idx = unique_ind(self.cov_pert_dose)

    def subset_condition(self, control=True):
        if control is None:
            return self
        else:
            idx = np.where(self.controls == control)[0].tolist()
            return SubDataset(self, idx)

    def __getitem__(self, i):
        cf_pert_dose_name = self.control_names[0]
        while any(c in cf_pert_dose_name for c in self.control_names):
            cf_i = np.random.choice(len(self.pert_dose))
            cf_pert_dose_name = self.pert_dose[cf_i]

        #cf_genes = None
        if self.sample_cf:
            covariate_name = indx(self.cov_names, i)
            cf_name = covariate_name + f"_{cf_pert_dose_name}"

            if cf_name in self.cov_pert_dose_idx:
                cf_inds = self.cov_pert_dose_idx[cf_name]
                cf_i = np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))
                #parent_cf_i = self.indices[cf_i]
                #cf_genes = self.parent_dataset.dataset.adata.X[parent_cf_i,:]
                #if scipy.sparse.issparse(cf_genes):
                #    cf_genes = cf_genes.toarray().squeeze()
                #cf_genes = torch.from_numpy(cf_genes)

        return (
            self.genes[i],
            indx(self.perturbations, i),
            self.genes[cf_i],
            indx(self.perturbations, cf_i),
            *[indx(cov, i) for cov in self.covariates]
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
    args = None,
):

    dataset = Dataset(
        data_path, perturbation_key, control_key, dose_key, covariate_keys, split_key, 
        sample_cf=sample_cf, control_name=control_name, embedded_dose=embedded_dose,
        perturbation_input=perturbation_input, args=args
    )

    start_split = time.time()
    splits = {
        "train": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("test", "all"),
        ## modified: returns whole dataset for testing and visualization
        "all": dataset.subset("all","all")
    }
    print(f"Dataset split has elapsed: {time.time() - start_split} seconds.")

    if return_dataset:
        return splits, dataset
    else:
        return splits        

indx = lambda a, i: a[i] if a is not None else None
