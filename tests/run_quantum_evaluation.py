import pytest
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from test_quantum_llm import TestQuantumLLM


def run_evaluation():
    """Run quantum LLM evaluation tests and save results."""
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run tests and collect results
    pytest.main([
        "test_quantum_llm.py",
        "-v",
        "--capture=tee-sys"
    ])
    
    # Load test results from TestQuantumLLM instance
    test_instance = None
    for item in pytest.main.session.items:
        if isinstance(item.instance, TestQuantumLLM):
            test_instance = item.instance
            break
    
    if test_instance and hasattr(test_instance, "performance_metrics"):
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = results_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, "w") as f:
            json.dump(test_instance.performance_metrics, f, indent=2)
        
        # Create visualization
        plot_results(test_instance.performance_metrics, results_dir, timestamp)


def plot_results(metrics, results_dir, timestamp):
    """Create visualizations of test results."""
    # Perplexity comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(
        ["Standard LLM", "Quantum LLM"],
        [
            metrics["standard"]["perplexity"],
            metrics["quantum"]["perplexity"]
        ]
    )
    plt.title("Model Perplexity Comparison")
    plt.ylabel("Perplexity (lower is better)")
    
    # Generation time comparison
    plt.subplot(1, 2, 2)
    plt.bar(
        ["Standard LLM", "Quantum LLM"],
        [
            metrics["standard"]["generation_time"],
            metrics["quantum"]["generation_time"]
        ]
    )
    plt.title("Generation Time Comparison")
    plt.ylabel("Time (ms)")
    
    plt.tight_layout()
    plt.savefig(results_dir / f"comparison_{timestamp}.png")
    plt.close()


if __name__ == "__main__":
    run_evaluation() 