"""
Test file for Quantum-Inspired NLP Model.

This module demonstrates the quantum NLP model on a sentiment analysis task.
"""

import torch
import torch.nn as nn
from quantum_applications import QuantumNLPModel


class SentimentAnalysis(nn.Module):
    """Quantum-inspired sentiment analysis model."""
    
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 64,
        num_qubits: int = 4,
        num_classes: int = 2
    ):
        super().__init__()
        self.quantum_nlp = QuantumNLPModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_qubits=num_qubits
        )
        self.classifier = nn.Linear(vocab_size, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process text and classify sentiment."""
        # Get quantum-processed features
        features = self.quantum_nlp(input_ids, attention_mask)
        
        # Average pooling over sequence length
        pooled = features.mean(dim=1)
        
        # Classify
        return self.classifier(pooled)


def create_vocabulary(
    texts: list[str],
    max_size: int = 5000
) -> dict[str, int]:
    """Create vocabulary from texts."""
    word_freq = {}
    
    # Count word frequencies
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and limit size
    sorted_words = sorted(
        word_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )
    vocab = {
        word: idx
        for idx, (word, _) in enumerate(sorted_words[:max_size-2])
    }
    
    # Add special tokens
    vocab['[PAD]'] = len(vocab)
    vocab['[UNK]'] = len(vocab)
    
    return vocab


def tokenize(
    text: str,
    vocab: dict[str, int],
    max_length: int = 64
) -> list[int]:
    """Convert text to token IDs."""
    # Split and convert to lowercase
    words = text.lower().split()
    
    # Convert words to IDs
    ids = [
        vocab.get(word, vocab['[UNK]'])
        for word in words[:max_length]
    ]
    
    # Pad sequence
    if len(ids) < max_length:
        ids.extend([vocab['[PAD]']] * (max_length - len(ids)))
    
    return ids


def test_sentiment_analysis():
    """Test quantum NLP model on sentiment analysis."""
    # Sample data
    texts = [
        "This movie was absolutely fantastic!",
        "The acting was terrible and the plot made no sense.",
        "I really enjoyed watching this film.",
        "What a waste of time and money.",
        "Great performances by all the actors.",
        "The special effects were amazing but the story was weak.",
        "I would highly recommend this movie to everyone.",
        "This has to be one of the worst films ever made."
    ]
    
    # 1 for positive, 0 for negative
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    
    # Create vocabulary
    vocab = create_vocabulary(texts)
    
    # Tokenize texts
    token_ids = torch.tensor([
        tokenize(text, vocab)
        for text in texts
    ])
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (token_ids != vocab['[PAD]']).float()
    
    # Initialize model
    model = SentimentAnalysis(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_qubits=4,
        num_classes=2
    )
    
    # Forward pass
    logits = model(token_ids, attention_mask)
    predictions = torch.argmax(logits, dim=1)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Text: {text}")
        print(f"Predicted sentiment: {sentiment}\n")
    
    # Calculate accuracy
    accuracy = (predictions == labels).float().mean()
    print(f"Accuracy: {accuracy.item():.2%}")


if __name__ == "__main__":
    test_sentiment_analysis() 