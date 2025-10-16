"""
Example usage of the compare_latent_vs_raw_predictions function
"""

from fcr.validation.predictions import (
    compare_latent_vs_raw_predictions,
    plot_prediction_comparison,
    save_prediction_results
)

# Example 1: Basic usage - Compare predictions for a cell type feature
def example_basic():
    """
    Basic example: Load a model and compare latent vs raw predictions
    """
    results = compare_latent_vs_raw_predictions(
        model_dir='experiments/test_run/',
        feature_name='cell_type',
        classifier='logistic',
        verbose=True
    )
    
    print("\n=== Results ===")
    print(f"Raw Data Accuracy: {results['raw_metrics']['accuracy']:.4f}")
    print(f"Latent Accuracy: {results['latent_metrics']['accuracy']:.4f}")
    print(f"ZX Latent Accuracy: {results['zx_metrics']['accuracy']:.4f}")
    print(f"ZT Latent Accuracy: {results['zt_metrics']['accuracy']:.4f}")
    print(f"ZXT Latent Accuracy: {results['zxt_metrics']['accuracy']:.4f}")


# Example 2: Using Random Forest classifier and saving results
def example_with_saving():
    """
    Example with Random Forest classifier and saving all results
    """
    results = compare_latent_vs_raw_predictions(
        model_dir='experiments/test_run/',
        feature_name='perturbation',
        classifier='random_forest',
        test_size=0.3,
        random_state=123,
        verbose=True
    )
    
    # Save all results
    save_prediction_results(
        results=results,
        output_dir='results/prediction_comparison/',
        prefix='cell_type_rf'
    )
    
    # Also create a custom plot
    fig = plot_prediction_comparison(
        results=results,
        save_path='results/custom_comparison.png',
        figsize=(14, 7)
    )


# Example 3: Load specific epoch and use sampling
def example_specific_epoch():
    """
    Example loading a specific epoch and sampling from latent distributions
    """
    results = compare_latent_vs_raw_predictions(
        model_dir='experiments/test_run/',
        feature_name='cell_type',
        target_epoch=50,  # Load checkpoint from epoch 50
        classifier='logistic',
        sample_latent=True,  # Sample from latent distributions instead of using means
        return_classifiers=True,  # Return trained classifiers for further analysis
        verbose=True
    )
    
    # Access the trained classifiers
    clf_raw = results['classifiers']['raw']
    clf_latent = results['classifiers']['latent']
    
    # Access latent representations
    ZX = results['latents']['ZX']
    ZT = results['latents']['ZT']
    ZXT = results['latents']['ZXT']
    
    print(f"\nLatent shapes:")
    print(f"  ZX: {ZX.shape}")
    print(f"  ZT: {ZT.shape}")
    print(f"  ZXT: {ZXT.shape}")


# Example 4: Analyzing multiple features
def example_multiple_features():
    """
    Example analyzing predictions for multiple features
    """
    features = ['cell_type', 'perturbation', 'dose']
    
    all_results = {}
    for feature in features:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing feature: {feature}")
            print('='*60)
            
            results = compare_latent_vs_raw_predictions(
                model_dir='experiments/test_run/',
                feature_name=feature,
                classifier='logistic',
                verbose=False  # Less verbose when running multiple
            )
            
            all_results[feature] = results
            
            # Print summary
            print(f"\n{feature} - Summary:")
            print(f"  Raw Accuracy: {results['raw_metrics']['accuracy']:.4f}")
            print(f"  Latent Accuracy: {results['latent_metrics']['accuracy']:.4f}")
            improvement = ((results['latent_metrics']['accuracy'] - 
                          results['raw_metrics']['accuracy']) / 
                         results['raw_metrics']['accuracy'] * 100)
            print(f"  Improvement: {improvement:+.2f}%")
            
        except Exception as e:
            print(f"Error processing {feature}: {e}")
            continue
    
    return all_results


if __name__ == '__main__':
    # Run basic example
    print("Running basic example...")
    example_basic()
    
    # Uncomment to run other examples:
    # example_with_saving()
    # example_specific_epoch()
    # example_multiple_features()
