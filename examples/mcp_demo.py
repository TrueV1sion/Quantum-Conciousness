import asyncio
import os
import torch
import numpy as np
from src.meta_cognitive_pipeline import MetaCognitivePipeline
from src.visualization.lattice_visualizer import LatticeVisualizer


async def demonstrate_mcp():
    """Demonstrate the Meta-Cognitive Pipeline."""
    print("\nMeta-Cognitive Pipeline Demo")
    print("===========================\n")
    
    # Initialize MCP
    plugins_dir = os.path.join(os.path.dirname(__file__), "../src/plugins")
    mcp = MetaCognitivePipeline(plugins_dir)
    
    # Configure cognitive engines
    config = {
        "CognitiveEngine_language": {
            "modality": "language",
            "model_name": "bert-base-uncased",
            "model_config": {
                "output_hidden_states": True,
                "output_attentions": True
            },
            "max_length": 512
        },
        "CognitiveEngine_numerical": {
            "modality": "numerical",
            "embedding_dim": 256
        },
        "CognitiveEngine_visual": {
            "modality": "visual",
            "model_name": "resnet50"
        },
        "CognitiveEngine_audio": {
            "modality": "audio",
            "sample_rate": 16000,
            "n_fft": 400,
            "n_mels": 128,
            "hop_length": 160
        }
    }
    
    # Initialize pipeline
    await mcp.initialize(config)
    
    # Create test inputs
    input_data = {
        # Language input
        "text": "The quantum nature of consciousness emerges from the "
                "interaction of multiple cognitive processes.",
        
        # Numerical input (simulated time series)
        "numerical": {
            "time_series": np.sin(np.linspace(0, 10, 100)) + \
                np.random.normal(0, 0.1, 100),
            "sampling_rate": 10
        },
        
        # Visual input (simulated image features)
        "visual": torch.randn(1, 3, 224, 224),
        
        # Audio input (simulated waveform)
        "audio": torch.randn(1, 16000),
        
        # Context
        "context": {
            "domain": "cognitive_science",
            "confidence": 0.9
        }
    }
    
    # Create consciousness field
    consciousness_field = torch.randn(1, 1024)  # [batch, consciousness_dim]
    
    # Process input
    print("\nProcessing input through cognitive engines...")
    results = await mcp.process_input(input_data, consciousness_field)
    
    # Print results from each engine
    for engine_name, engine_results in results.items():
        if engine_name != 'contradictions':
            print(f"\n{engine_name} Results:")
            print(f"- Context ID: {engine_results.get('context_id')}")
            print(f"- Number of Nodes: {len(engine_results.get('nodes', []))}")
            
            # Print node details
            nodes = engine_results.get('nodes', [])
            if nodes:
                print("\nSignificant Nodes:")
                for node in nodes:
                    if node.confidence > 0.7:
                        print(
                            f"- Node {node.id}: "
                            f"confidence={node.confidence:.3f}, "
                            f"connections={len(node.connections)}"
                        )
                        
                        # Print modality-specific details
                        if 'pattern_type' in node.metadata:
                            print(
                                f"  Pattern: {node.metadata['pattern_type']}"
                            )
                        elif 'level' in node.metadata:
                            print(
                                f"  Level: {node.metadata['level']}, "
                                f"Position: {node.metadata['position']}"
                            )
                        elif 'time_range' in node.metadata:
                            print(
                                f"  Time: {node.metadata['time_range']}, "
                                f"Freq: {node.metadata['frequency_bin']}"
                            )
    
    # Print any contradictions found
    if 'contradictions' in results:
        print("\nContradictions Found:")
        contradictions = results['contradictions']
        for id, resolution in contradictions.items():
            print(f"\nContradiction {id}:")
            print(f"- Status: {resolution['status']}")
            print(f"- Method: {resolution['resolution_method']}")
            print(f"- Confidence: {resolution['confidence']:.3f}")
            print(f"- Reasoning: {resolution['reasoning']}")
    
    # Print echoic heuristics
    print("\nEchoic Heuristics:")
    for echo in mcp.echoic_heuristics:
        print(f"\nEcho from {echo.source_engine}:")
        print(f"- Confidence: {echo.confidence:.3f}")
        print("- Resonances:")
        if echo.resonances:
            for engine, resonance in echo.resonances.items():
                print(f"  * {engine}: {resonance:.3f}")
        print(f"- Truth Candidate: {echo.is_truth_candidate}")
    
    # Print truth candidates
    print("\nTruth Candidates:")
    for truth in mcp.truth_candidates:
        print(f"\nTruth from {truth.source_engine}:")
        print(f"- Confidence: {truth.confidence:.3f}")
        print("- Supporting Engines:")
        if truth.resonances:
            strong_support = [
                engine for engine, res in truth.resonances.items()
                if res > 0.7
            ]
            print(f"  * {', '.join(strong_support)}")
    
    # Print cross-modality insights
    print("\nCross-Modality Insights:")
    all_nodes = []
    for engine_results in results.values():
        if isinstance(engine_results, dict):
            all_nodes.extend(engine_results.get('nodes', []))
    
    # Find connections between different modalities
    cross_modal_connections = []
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            node1, node2 = all_nodes[i], all_nodes[j]
            if (
                node1.modality != node2.modality
                and node2.id in node1.connections
            ):
                cross_modal_connections.append((node1, node2))
    
    if cross_modal_connections:
        for node1, node2 in cross_modal_connections:
            print(
                f"\nConnection between {node1.modality} and {node2.modality}:"
            )
            print(f"- Node {node1.id} ({node1.confidence:.3f}) <-> "
                  f"Node {node2.id} ({node2.confidence:.3f})")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = LatticeVisualizer()
    
    # Plot context lattice
    print("\nContext Lattice:")
    visualizer.visualize_lattice(
        mcp,
        show_labels=True,
        show_confidence=True,
        highlight_resonances=True,
        min_confidence=0.5
    )
    
    # Plot resonance matrix
    print("\nResonance Matrix:")
    visualizer.plot_resonance_matrix(mcp)
    
    # Plot confidence distributions
    print("\nConfidence Distributions:")
    visualizer.plot_confidence_distribution(mcp)


if __name__ == "__main__":
    asyncio.run(demonstrate_mcp()) 