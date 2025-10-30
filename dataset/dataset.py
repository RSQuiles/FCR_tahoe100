from typing import Union
import scipy
import numpy as np
import scanpy as sc
import pandas as pd

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
    ):
        if type(data) == str:
            print("Reading AnnData...")
            data = sc.read(data)

        self.sample_cf = sample_cf
        self.cf_samples = cf_samples

        # MODIFIED INCREASE FLEXIBILITY AND ADAPT TO TAHOE100 Dataset
        # Fields
        # perturbation
        #if perturbation_key in data.uns["fields"]:
        #    perturbation_key = data.uns["fields"][perturbation_key]
        #else:
        assert perturbation_key in data.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"

        # control
        #if control_key in data.uns["fields"]:
        #    control_key = data.uns["fields"][control_key]
        #else:
        if control_key not in data.obs.columns:
            if control_name is not None:
                print(f"Adding control column based on control name: {control_name}...")
                data.obs[control_key] = (data.obs[perturbation_key] == control_name).astype(int)
            else:
                raise ValueError(f"Control {control_key} is missing in the provided adata and no control_name was given.")
        
        # dose
        #if dose_key in data.uns["fields"]:
        #    dose_key = data.uns["fields"][dose_key]
        if dose_key is None:
            print("Adding a dummy dose...")
            data.obs["dummy_dose"] = 1.0
            dose_key = "dummy_dose"
        elif dose_key not in data.obs.columns:
            if embedded_dose is not None:
                data.obs[dose_key] = data.obs[embedded_dose].str.split(",").str[1].astype(float)
            else:
                raise ValueError(f"Dose {dose_key} is missing in the provided adata and no embedded_dose column was given.")

        # covariates
        #if isinstance(covariate_keys, str) and covariate_keys in data.uns["fields"]:
        #    covariate_keys = list(data.uns["fields"][covariate_keys])
        if covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            data.obs["dummy_covar"] = "dummy-covar"
            covariate_keys = ["dummy_covar"]
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in data.obs.columns, f"Covariate {key} is missing in the provided adata"
        # split
        #if split_key in data.uns["fields"]:
        #    split_key = data.uns["fields"][split_key]
        if split_key is None:
            print(f"Performing automatic train-test split with {test_ratio} ratio.")
            from sklearn.model_selection import train_test_split

            data.obs["split"] = "train"
            idx_train, idx_test = train_test_split(
                data.obs_names, test_size=test_ratio, random_state=random_state
            )
            data.obs["split"].loc[idx_train] = "train"
            data.obs["split"].loc[idx_test] = "test"
            split_key = "split"
        else:
            assert split_key in data.obs.columns, f"Split {split_key} is missing in the provided adata"

        self.indices = {
            "all": list(range(len(data.obs))),
            "control": np.where(data.obs[control_key] == 1)[0].tolist(),
            "treated": np.where(data.obs[control_key] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

        self.perturbation_key = perturbation_key
        self.perturbation_input = perturbation_input
        self.control_key = control_key
        self.dose_key = dose_key
        self.covariate_keys = covariate_keys

        self.control_names = np.unique(
            data[data.obs[self.control_key] == 1].obs[self.perturbation_key]
        )

        if scipy.sparse.issparse(data.X):
            self.genes = torch.Tensor(data.X.A)
        else:
            self.genes = torch.Tensor(data.X) # data.layers["counts"]

        self.var_names = data.var_names

        self.pert_names = np.array(data.obs[perturbation_key].values)
        self.doses = np.array(data.obs[dose_key].values)

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
                dose_combos = str(data.obs[dose_key].values[i]).split("+")
                perturbation_ohe = []
                for j, d in enumerate(dose_combos):
                    perturbation_ohe.append(float(d) * perturbation_combos[j])
                perturbations.append(sum(perturbation_ohe))

            self.perturbations = torch.stack(perturbations)
            self.num_treatments = len(pert_unique) # treatment input dimension

        elif self.perturbation_input == "chemberta":
            print("Using ChemBERTa embeddings for perturbations!")
            from transformers import AutoTokenizer, AutoModel
            MODEL_ID = "DeepChem/ChemBERTa-100M-MLM"

            # select device
            device = (
                "cuda:" + str(args["gpu"])
                    if (not args["cpu"]) 
                        and torch.cuda.is_available() 
                    else 
                "cpu"
            )

            # Load the tokenizer and model
            print("Loading ChemBERTa model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

            # Load drug metadata
            print("Generating perturbation SMILES...")
            drug_metadata_path = "/cluster/work/bewi/members/rquiles/experiments/datasets/drug_metadata.parquet"
            drug_metadata = pd.read_parquet(drug_metadata_path)
            self.pert_smiles = np.array([drug_metadata[drug_metadata["drug"] == name]["canonical_smiles"].astype(str).iloc[0] for name in self.pert_names]).tolist()

            # Generate embeddings
            emb_list = [] 
            batch_size = 1024 # tune depending on GPU memory

            # Iterate through pert_smiles in batches
            print("Generating embeddings...")
            for i in range(0, len(self.pert_smiles), batch_size):
                batch_smiles = self.pert_smiles[i:i + batch_size]

                # Tokenize batch
                batch = tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)

                # Forward pass
                with torch.no_grad():
                    outputs = model(**batch)
                    hidden_states = outputs.last_hidden_state

                    # Mask-aware mean pooling
                    attn = batch["attention_mask"].unsqueeze(-1).float()
                    emb = (hidden_states * attn).sum(1) / attn.sum(1)

                # Move embeddings to CPU and store
                emb_list.append(emb.cpu())

            # Concatenate all batches into one big tensor
            self.perturbations = torch.cat(emb_list, dim=0)
            self.num_treatments = self.perturbations.shape[1] # treatment input dimension
            # print(f"Perturbation embeddings shape: {self.perturbations.shape}")
            
        else:
            raise NotImplementedError("Unmatched input mode for treatments")
        
        self.controls = data.obs[self.control_key].values.astype(bool)
        
        if covariate_keys is not None:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            cov_names = []
            self.covars_dict = {}
            self.covariates = []
            self.num_covariates = []
            for cov in covariate_keys:
                values = np.array(data.obs[cov].values)
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
            self.cov_names = np.array(["_".join(c) for c in zip(*cov_names)])
        else:
            self.cov_names = np.array([""] * len(data))
            self.covars_dict = None
            self.covariates = None
            self.num_covariates = None

        self.num_outcomes = self.genes.shape[1]

        self.cov_pert = np.array([
            f"{self.cov_names[i]}_"
            f"{data.obs[perturbation_key].values[i]}"
            for i in range(len(data))
        ])
        self.cov_control = np.array([
            f"{self.cov_names[i]}_"
            f"{data.obs[control_key].values[i]}"
            for i in range(len(data))
        ])
        
        self.pert_dose = np.array([
            f"{data.obs[perturbation_key].values[i]}"
            f"_{data.obs[dose_key].values[i]}"
            for i in range(len(data))
        ])
        self.cov_pert_dose = np.array([
            f"{self.cov_names[i]}_{self.pert_dose[i]}"
            for i in range(len(data))
        ])

        ## modified: commented out if de_genes not used downstream
        """
        if not ("rank_genes_groups_cov" in data.uns):
            data.obs["cov_name"] = self.cov_names
            data.obs["cov_pert_name"] = self.cov_pert
            print("Ranking genes for DE genes...")
            rank_genes_groups(data,
                groupby="cov_pert_name", 
                reference="cov_name",
                control_key=control_key)
        self.de_genes = data.uns["rank_genes_groups_cov"]
        """

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
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
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

        cf_genes = None
        if self.sample_cf:
            covariate_name = indx(self.cov_names, i)
            cf_name = covariate_name + f"_{cf_pert_dose_name}"

            if cf_name in self.cov_pert_dose_idx:
                cf_inds = self.cov_pert_dose_idx[cf_name]
                cf_genes = self.genes[np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))]

        return (
            self.genes[i],
            indx(self.perturbations, i),
            cf_genes,
            indx(self.perturbations, cf_i),
            *[indx(cov, i) for cov in self.covariates]
        )

    def __len__(self):
        return len(self.genes)

 
class SubDataset_Pair:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
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
        self.cov_control = indx(dataset.cov_control, indices)
        self.control_vals = '1'
        
        self.var_names = dataset.var_names
        ## modified: commented out if de_genes not used downstream
        #self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_outcomes = dataset.num_outcomes
        self.num_treatments = dataset.num_treatments
        # print(f"num_treatments in SubDataset_Pair: {self.num_treatments}")

        if self.sample_cf:
            self.cov_pert_dose_idx = unique_ind(self.cov_pert_dose)
            self.cov_control_idx = unique_ind(self.cov_control)


    def subset_condition(self, control=True):
        if control is None:
            return self
        else:
            idx = np.where(self.controls == control)[0].tolist()
            return SubDataset(self, idx)

    def __getitem__(self, i):
        
        
        ### get the control sample
    
        cf_pert_dose_name = self.control_names[0]
        cf_genes = None
        cf_i=0
        covariate_name = indx(self.cov_names, i)
        cf_name = covariate_name + f"_{self.control_vals}"
        # print("cf_pert_dose_name {}".format(cf_pert_dose_name))

        if cf_name in self.cov_control:
            cf_inds = self.cov_control_idx[cf_name]
            cf_i = np.random.choice(cf_inds)
            cf_genes = self.genes[cf_i]
                        
        return (
            self.genes[i],
            indx(self.perturbations, i),
            cf_genes,
            cf_i,
            *[indx(cov, i) for cov in self.covariates]
        )

    def __len__(self):
        return len(self.genes)
    
    
    
# LEGACY CODE: not used in currrent implementation
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

    splits = {
        "train": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("test", "all"),
        ## modified: returns whole dataset for testing and visualization
        "all": dataset.subset("all","all")
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits        

indx = lambda a, i: a[i] if a is not None else None
