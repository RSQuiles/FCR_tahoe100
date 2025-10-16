"""
The aim of this script is to compare the extracted latents against the raw data as inputs of the same
(simple) ML model to predict e.g. cell lines or treatments
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from typing import Dict, Tuple, Optional, Union, Any

# from ..fcr import get_model, FCR_sim
from ..utils.data_utils import data_collate
from .view_results import compute_latents


def compare_latent_vs_raw_predictions(
    model_dir: str,
    feature_name: str,
    target_epoch: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    classifier: str = 'logistic',
    # sample_latent: bool = False,
    return_classifiers: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load a trained FCR model, extract latents, and compare how well they predict
    a given feature in adata.obs compared to using raw data in adata.X.
    
    Parameters
    ----------
    model_dir : str
        Path to the directory containing the trained model checkpoints and config
    feature_name : str
        Name of the feature in adata.obs to predict (e.g., 'cell_type', 'perturbation')
    target_epoch : int, optional
        Specific epoch checkpoint to load. If None, loads the latest checkpoint
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    classifier : str, default='logistic'
        Type of classifier to use: 'logistic' or 'random_forest'
    sample_latent : bool, default=False
        Whether to sample from latent distributions (True) or use means (False)
    return_classifiers : bool, default=False
        Whether to return the trained classifiers
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'raw_metrics': metrics using raw data (accuracy, f1, etc.)
        - 'latent_metrics': metrics using latent representations
        - 'zx_metrics': metrics using only ZX latents
        - 'zt_metrics': metrics using only ZT latents
        - 'zxt_metrics': metrics using only ZXT latents
        - 'model': the loaded FCR model
        - 'adata': the loaded AnnData object
        - 'latents': dict with 'ZX', 'ZXT', 'ZT' tensors
        - 'classifiers': trained classifiers (if return_classifiers=True)
    """
    
    if verbose:
        print(f"Loading model from: {model_dir}")
        print(f"Target feature: {feature_name}")
    
    # Load model and data
    args, model, datasets = get_model(model_dir, target_epoch=target_epoch)
    
    # Get the AnnData object path from args
    data_path = args.get('data_path')
    if data_path is None:
        raise ValueError("Could not find 'data_path' in model arguments")
    
    if verbose:
        print(f"Loading AnnData from: {data_path}")
    
    adata = sc.read(data_path)
    
    # Check if feature exists
    if feature_name not in adata.obs.columns:
        raise ValueError(f"Feature '{feature_name}' not found in adata.obs. "
                        f"Available columns: {list(adata.obs.columns)}")
    
    # Get labels
    y = adata.obs[feature_name].values
    
    if verbose:
        print(f"Computing latents for {len(adata)} samples...")
    
    # Compute latents (and save them to AnnData)
    compute_latents(model, datasets, adata)
        
    # Convert to numpy for sklearn
    ZX_np = adata.obsm["ZXs"]
    ZXT_np = adata.obsm["ZXTs"]
    ZT_np = adata.obsm["ZTs"]

    # Concatenate all latents
    latents_combined = np.concatenate([ZX_np, ZXT_np, ZT_np], axis=1)
    
    if verbose:
        print(f"Latent dimensions - ZX: {ZX_np.shape[1]}, ZXT: {ZXT_np.shape[1]}, ZT: {ZT_np.shape[1]}")
        print(f"Total latent dimension: {latents_combined.shape[1]}")
    
    # Get raw data
    X_raw = adata.X.A if hasattr(adata.X, 'A') else adata.X
    
    # Split data
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Prepare datasets
    X_raw_train, X_raw_test = X_raw[train_idx], X_raw[test_idx]
    X_latent_train, X_latent_test = latents_combined[train_idx], latents_combined[test_idx]
    ZX_train, ZX_test = ZX_np[train_idx], ZX_np[test_idx]
    ZXT_train, ZXT_test = ZXT_np[train_idx], ZXT_np[test_idx]
    ZT_train, ZT_test = ZT_np[train_idx], ZT_np[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Initialize classifiers
    if classifier == 'logistic':
        clf_raw = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_latent = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_zx = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_zxt = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_zt = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == 'random_forest':
        clf_raw = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf_latent = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf_zx = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf_zxt = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf_zt = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}. Use 'logistic' or 'random_forest'")
    
    if verbose:
        print(f"\nTraining {classifier} classifiers...")
    
    # Train and evaluate classifiers
    # Raw data
    clf_raw.fit(X_raw_train, y_train)
    y_pred_raw = clf_raw.predict(X_raw_test)
    
    # Combined latents
    clf_latent.fit(X_latent_train, y_train)
    y_pred_latent = clf_latent.predict(X_latent_test)
    
    # Individual latent components
    clf_zx.fit(ZX_train, y_train)
    y_pred_zx = clf_zx.predict(ZX_test)
    
    clf_zxt.fit(ZXT_train, y_train)
    y_pred_zxt = clf_zxt.predict(ZXT_test)
    
    clf_zt.fit(ZT_train, y_train)
    y_pred_zt = clf_zt.predict(ZT_test)
    
    # Calculate metrics
    def get_metrics(y_true, y_pred, name):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except:
                pass
        
        if verbose:
            print(f"\n{name} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    raw_metrics = get_metrics(y_test, y_pred_raw, "Raw Data")
    latent_metrics = get_metrics(y_test, y_pred_latent, "Combined Latents")
    zx_metrics = get_metrics(y_test, y_pred_zx, "ZX Latents")
    zxt_metrics = get_metrics(y_test, y_pred_zxt, "ZXT Latents")
    zt_metrics = get_metrics(y_test, y_pred_zt, "ZT Latents")
    
    # Prepare results
    results = {
        'raw_metrics': raw_metrics,
        'latent_metrics': latent_metrics,
        'zx_metrics': zx_metrics,
        'zxt_metrics': zxt_metrics,
        'zt_metrics': zt_metrics,
        'model': model,
        'adata': adata,
        'latents': {
            'ZX': ZX_np,
            'ZXT': ZXT_np,
            'ZT': ZT_np,
            'combined': latents_combined
        },
        'test_indices': test_idx,
        'train_indices': train_idx,
        'predictions': {
            'raw': y_pred_raw,
            'latent': y_pred_latent,
            'zx': y_pred_zx,
            'zxt': y_pred_zxt,
            'zt': y_pred_zt
        },
        'y_test': y_test
    }
    
    if return_classifiers:
        results['classifiers'] = {
            'raw': clf_raw,
            'latent': clf_latent,
            'zx': clf_zx,
            'zxt': clf_zxt,
            'zt': clf_zt
        }
    
    if verbose:
        print("\n" + "="*60)
        print("Summary:")
        print(f"Raw data accuracy: {raw_metrics['accuracy']:.4f}")
        print(f"Latent accuracy: {latent_metrics['accuracy']:.4f}")
        improvement = ((latent_metrics['accuracy'] - raw_metrics['accuracy']) / 
                      raw_metrics['accuracy'] * 100)
        print(f"Improvement: {improvement:+.2f}%")
        print("="*60)
    
    return results


def plot_prediction_comparison(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create visualization comparing prediction performance between raw data and latents.
    
    Parameters
    ----------
    results : dict
        Output from compare_latent_vs_raw_predictions()
    save_path : str, optional
        Path to save the figure. If None, only displays
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract metrics
    methods = ['Raw Data', 'Combined\nLatents', 'ZX', 'ZXT', 'ZT']
    metric_keys = ['raw_metrics', 'latent_metrics', 'zx_metrics', 'zxt_metrics', 'zt_metrics']
    
    accuracies = [results[key]['accuracy'] for key in metric_keys]
    f1_scores = [results[key]['f1_macro'] for key in metric_keys]
    
    # Plot accuracy
    ax = axes[0]
    bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot F1 scores
    ax = axes[1]
    bars = ax.bar(methods, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('F1 Score (macro)', fontsize=12)
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def save_prediction_results(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = 'prediction_comparison'
) -> None:
    """
    Save prediction comparison results to files.
    
    Parameters
    ----------
    results : dict
        Output from compare_latent_vs_raw_predictions()
    output_dir : str
        Directory to save results
    prefix : str, default='prediction_comparison'
        Prefix for output filenames
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as CSV
    metrics_data = []
    for method, key in [('Raw Data', 'raw_metrics'), 
                        ('Combined Latents', 'latent_metrics'),
                        ('ZX Latents', 'zx_metrics'),
                        ('ZXT Latents', 'zxt_metrics'),
                        ('ZT Latents', 'zt_metrics')]:
        row = {'method': method}
        row.update(results[key])
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, f'{prefix}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save latents as NPZ
    latents_path = os.path.join(output_dir, f'{prefix}_latents.npz')
    np.savez(
        latents_path,
        ZX=results['latents']['ZX'],
        ZXT=results['latents']['ZXT'],
        ZT=results['latents']['ZT'],
        combined=results['latents']['combined'],
        test_indices=results['test_indices'],
        train_indices=results['train_indices']
    )
    print(f"Latents saved to: {latents_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': results['y_test'],
        'pred_raw': results['predictions']['raw'],
        'pred_latent': results['predictions']['latent'],
        'pred_zx': results['predictions']['zx'],
        'pred_zxt': results['predictions']['zxt'],
        'pred_zt': results['predictions']['zt']
    })
    predictions_path = os.path.join(output_dir, f'{prefix}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Create and save visualization
    fig = plot_prediction_comparison(results)
    fig_path = os.path.join(output_dir, f'{prefix}_plot.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {fig_path}")
    plt.close(fig)