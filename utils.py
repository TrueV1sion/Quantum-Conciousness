# utils.py

import torch
import numpy as np
from scipy.linalg import expm
from typing import Any, Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

def matrix_exponential(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix exponential for quantum operations.
    
    Args:
        matrix: Input tensor to compute exponential for
        
    Returns:
        Tensor containing matrix exponential
    """
    try:
        if len(matrix.shape) == 3:  # Batched input
            matrix_np = matrix.detach().cpu().numpy()
            exp_matrices = np.array([expm(m) for m in matrix_np])
            return torch.from_numpy(exp_matrices).to(matrix.device)
        else:  # Single matrix
            matrix_np = matrix.detach().cpu().numpy()
            exp_matrix = expm(matrix_np)
            return torch.from_numpy(exp_matrix).to(matrix.device)
    except Exception as e:
        logger.error(f"Matrix exponential calculation failed: {str(e)}")
        raise

def calculate_entropy(coeffs: torch.Tensor) -> float:
    """
    Calculate Shannon entropy of wavelet coefficients.
    
    Args:
        coeffs: Wavelet coefficients tensor
        
    Returns:
        Float value representing entropy
    """
    try:
        coeffs_abs = torch.abs(coeffs)
        probs = coeffs_abs / (torch.sum(coeffs_abs, dim=-1, keepdim=True) + 1e-10)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return float(entropy.mean().item())
    except Exception as e:
        logger.error(f"Entropy calculation failed: {str(e)}")
        raise

def find_dominant_frequency(pattern: torch.Tensor) -> float:
    """
    Find the dominant frequency in a given pattern using FFT.
    
    Args:
        pattern: Input tensor pattern
        
    Returns:
        Float value of dominant frequency
    """
    try:
        if len(pattern.shape) > 1:  # Batched input
            pattern = pattern.mean(dim=0)  # Average across batch
        fft_result = torch.fft.fft(pattern)
        amplitudes = torch.abs(fft_result)
        dominant_idx = torch.argmax(amplitudes)
        freqs = torch.fft.fftfreq(len(pattern))
        dominant_freq = abs(freqs[dominant_idx].item())
        return dominant_freq
    except Exception as e:
        logger.error(f"Dominant frequency calculation failed: {str(e)}")
        raise

def calculate_coherence(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """
    Calculate coherence between two quantum states.
    
    Args:
        state1: First quantum state tensor
        state2: Second quantum state tensor
        
    Returns:
        Float value representing coherence
    """
    try:
        # Handle batched input
        if len(state1.shape) > 1:
            state1 = state1.mean(dim=0)
        if len(state2.shape) > 1:
            state2 = state2.mean(dim=0)
            
        state1_norm = state1 / (torch.norm(state1) + 1e-10)
        state2_norm = state2 / (torch.norm(state2) + 1e-10)
        coherence = torch.abs(torch.dot(state1_norm, state2_norm)).item()
        return coherence
    except Exception as e:
        logger.error(f"Coherence calculation failed: {str(e)}")
        raise

def create_resonance_matrix(frequencies: List[float], dim: int) -> torch.Tensor:
    """
    Create a resonance matrix from given frequencies.
    
    Args:
        frequencies: List of resonance frequencies
        dim: Dimension of the output matrix
        
    Returns:
        Tensor containing resonance matrix
    """
    try:
        t = torch.linspace(0, 1, dim)
        matrix = torch.zeros((dim, dim))
        for freq in frequencies:
            phase = 2 * np.pi * freq * t
            matrix += torch.outer(torch.sin(phase), torch.sin(phase))
        return matrix / len(frequencies)
    except Exception as e:
        logger.error(f"Resonance matrix creation failed: {str(e)}")
        raise

def encode_information(data: Any, encoding_dim: int) -> torch.Tensor:
    """
    Encode arbitrary information into quantum-compatible tensor.
    
    Args:
        data: Input data to encode
        encoding_dim: Dimension of the encoding
        
    Returns:
        Tensor containing encoded information
    """
    try:
        if isinstance(data, str):
            # Encode string using character ASCII values
            ascii_vals = [ord(c) for c in data[:encoding_dim]]
            while len(ascii_vals) < encoding_dim:
                ascii_vals.append(0)
            tensor = torch.tensor(ascii_vals, dtype=torch.float32)
            return tensor / 255.0  # Normalize
        elif isinstance(data, (int, float)):
            # Encode number using binary representation
            tensor = torch.zeros(encoding_dim)
            tensor[0] = float(data)
            return tensor / (abs(float(data)) + 1)  # Normalize
        elif isinstance(data, torch.Tensor):
            # Handle batched input
            if len(data.shape) > 1:
                if data.shape[-1] > encoding_dim:
                    return data[..., :encoding_dim]
                elif data.shape[-1] < encoding_dim:
                    padded = torch.zeros(*data.shape[:-1], encoding_dim)
                    padded[..., :data.shape[-1]] = data
                    return padded
                return data
            else:
                # Single tensor
                if data.numel() > encoding_dim:
                    return data[:encoding_dim]
                elif data.numel() < encoding_dim:
                    padded = torch.zeros(encoding_dim)
                    padded[:data.numel()] = data
                    return padded
                return data
        else:
            raise ValueError(f"Unsupported data type for encoding: {type(data)}")
    except Exception as e:
        logger.error(f"Information encoding failed: {str(e)}")
        raise

def decode_information(encoded_tensor: torch.Tensor, original_type: type) -> Any:
    """
    Decode quantum tensor back to original information type.
    
    Args:
        encoded_tensor: Encoded tensor to decode
        original_type: Original type of the information
        
    Returns:
        Decoded information in original type
    """
    try:
        # Handle batched input
        if len(encoded_tensor.shape) > 1:
            encoded_tensor = encoded_tensor.mean(dim=0)  # Average across batch
            
        if original_type == str:
            # Decode string from ASCII values
            ascii_vals = (encoded_tensor * 255).round().int()
            chars = [chr(val) for val in ascii_vals if val > 0]
            return ''.join(chars)
        elif original_type in (int, float):
            # Decode number from first tensor value
            return original_type(encoded_tensor[0].item())
        elif original_type == torch.Tensor:
            return encoded_tensor
        else:
            raise ValueError(f"Unsupported type for decoding: {original_type}")
    except Exception as e:
        logger.error(f"Information decoding failed: {str(e)}")
        raise

class MovingAverage:
    """Utility class for calculating moving averages of values."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def update(self, value: float) -> float:
        """
        Update moving average with new value.
        
        Args:
            value: New value to add
            
        Returns:
            Current moving average
        """
        self.values.append(value)
        return self.average
    
    @property
    def average(self) -> float:
        """Calculate current moving average."""
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    @property
    def std(self) -> float:
        """Calculate current standard deviation."""
        if not self.values:
            return 0.0
        avg = self.average
        variance = sum((x - avg) ** 2 for x in self.values) / len(self.values)
        return variance ** 0.5
