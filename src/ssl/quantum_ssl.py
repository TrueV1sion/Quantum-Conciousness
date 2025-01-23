"""Quantum Self-Supervised Learning Components.

This module provides self-supervised learning capabilities for quantum consciousness models,
including:
1. Masked Quantum Autoencoding
2. Quantum Contrastive Learning
3. Quantum State Autoencoders
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..processors import UnifiedState
from ..quantum_pqc import ParameterizedQuantumCircuit, PQCConfig
from ..quantum_bridge_google import GoogleQuantumBridge


@dataclass
class QuantumSSLConfig:
    """Configuration for quantum self-supervised learning."""
    # Masking parameters
    mask_ratio: float = 0.8
    mask_noise_scale: float = 0.1
    
    # Contrastive learning parameters
    temperature: float = 0.07
    num_negatives: int = 16
    projection_dim: int = 128
    
    # Autoencoder parameters
    latent_dim: int = 64
    reconstruction_weight: float = 1.0
    quantum_weight: float = 0.5


class MaskedQuantumAutoencoder(nn.Module):
    """Implements masked quantum autoencoding for self-supervised learning."""
    
    def __init__(
        self,
        config: QuantumSSLConfig,
        pqc: ParameterizedQuantumCircuit
    ):
        super().__init__()
        self.config = config
        self.pqc = pqc
        
        # Projection layers
        self.encoder = nn.Sequential(
            nn.Linear(pqc.config.num_qubits, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, pqc.config.num_qubits)
        )
    
    def _mask_input(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to input quantum state."""
        if mask is None:
            mask = torch.rand(
                x.shape[0],
                x.shape[1],
                device=x.device
            ) < self.config.mask_ratio
        
        # Add noise to masked regions
        noise = torch.randn_like(x) * self.config.mask_noise_scale
        masked_x = x * (1 - mask.unsqueeze(-1)) + noise * mask.unsqueeze(-1)
        
        return masked_x, mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with masking and reconstruction."""
        # Apply masking
        masked_x, mask = self._mask_input(x, mask)
        
        # Encode
        latent = self.encoder(masked_x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Calculate reconstruction metrics
        metrics = {
            'mask': mask,
            'latent': latent.detach(),
            'reconstruction_error': torch.nn.functional.mse_loss(
                reconstructed,
                x
            ).item()
        }
        
        return reconstructed, metrics


class QuantumContrastiveLearner(nn.Module):
    """Implements quantum contrastive learning."""
    
    def __init__(
        self,
        config: QuantumSSLConfig,
        pqc: ParameterizedQuantumCircuit
    ):
        super().__init__()
        self.config = config
        self.pqc = pqc
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(pqc.config.num_qubits, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )
    
    def _augment_state(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply quantum state augmentation."""
        # Add random noise
        noise = torch.randn_like(x) * self.config.mask_noise_scale
        augmented = x + noise
        
        # Normalize
        augmented = augmented / torch.norm(augmented, dim=-1, keepdim=True)
        
        return augmented
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss."""
        # Normalize embeddings
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.T) / self.config.temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(z1.shape[0], device=z1.device)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(similarity, labels)
        
        return loss
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with contrastive learning."""
        # Generate augmented view
        x_aug = self._augment_state(x)
        
        # Get projections
        z1 = self.projector(x)
        z2 = self.projector(x_aug)
        
        # Compute loss
        loss = self.contrastive_loss(z1, z2)
        
        metrics = {
            'contrastive_loss': loss.item(),
            'z1': z1.detach(),
            'z2': z2.detach()
        }
        
        return z1, z2, metrics


class QuantumStateAutoencoder(nn.Module):
    """Implements quantum state autoencoding."""
    
    def __init__(
        self,
        config: QuantumSSLConfig,
        pqc: ParameterizedQuantumCircuit
    ):
        super().__init__()
        self.config = config
        self.pqc = pqc
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(pqc.config.num_qubits, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, pqc.config.num_qubits)
        )
        
        # Quantum bridge for state verification
        self.quantum_verifier = nn.Linear(
            config.latent_dim,
            pqc.config.num_parameters
        )
    
    def _quantum_fidelity_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum fidelity between states."""
        # Normalize states
        original = original / torch.norm(
            original, dim=-1, keepdim=True
        )
        reconstructed = reconstructed / torch.norm(
            reconstructed, dim=-1, keepdim=True
        )
        
        # Compute fidelity
        fidelity = torch.abs(
            torch.sum(original.conj() * reconstructed, dim=-1)
        )
        
        return -torch.mean(fidelity)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with quantum state autoencoding."""
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Quantum verification parameters (used for state preparation)
        quantum_verification = self.quantum_verifier(latent)
        
        # Compute losses
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, x)
        quantum_loss = self._quantum_fidelity_loss(x, reconstructed)
        
        # Combined loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.quantum_weight * quantum_loss
        )
        
        metrics = {
            'reconstruction_loss': reconstruction_loss.item(),
            'quantum_loss': quantum_loss.item(),
            'total_loss': total_loss.item(),
            'latent': latent.detach(),
            'quantum_verification': quantum_verification.detach()
        }
        
        return reconstructed, metrics


class QuantumSSLManager:
    """Manages quantum self-supervised learning components."""
    
    def __init__(
        self,
        config: QuantumSSLConfig,
        pqc: ParameterizedQuantumCircuit
    ):
        self.config = config
        self.pqc = pqc
        
        # Initialize components
        self.masked_autoencoder = MaskedQuantumAutoencoder(config, pqc)
        self.contrastive_learner = QuantumContrastiveLearner(config, pqc)
        self.state_autoencoder = QuantumStateAutoencoder(config, pqc)
    
    async def pretrain(
        self,
        states: List[UnifiedState[Any]],
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Pretrain all SSL components."""
        # Convert states to tensors
        quantum_fields = torch.stack([
            state.quantum_field for state in states
        ])
        
        # Optimizers
        mae_optimizer = torch.optim.Adam(
            self.masked_autoencoder.parameters(),
            lr=learning_rate
        )
        cl_optimizer = torch.optim.Adam(
            self.contrastive_learner.parameters(),
            lr=learning_rate
        )
        ae_optimizer = torch.optim.Adam(
            self.state_autoencoder.parameters(),
            lr=learning_rate
        )
        
        training_history: Dict[str, List[float]] = {
            'mae_loss': [],
            'contrastive_loss': [],
            'autoencoder_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(len(quantum_fields))
            
            for i in range(0, len(quantum_fields), batch_size):
                indices = perm[i:i + batch_size]
                batch = quantum_fields[indices]
                
                # 1. Masked Autoencoding
                mae_optimizer.zero_grad()
                reconstructed, mae_metrics = self.masked_autoencoder(batch)
                mae_loss = mae_metrics['reconstruction_error']
                mae_loss = torch.tensor(mae_loss, requires_grad=True)
                mae_loss.backward()
                mae_optimizer.step()
                
                # 2. Contrastive Learning
                cl_optimizer.zero_grad()
                _, _, cl_metrics = self.contrastive_learner(batch)
                cl_loss = cl_metrics['contrastive_loss']
                cl_loss = torch.tensor(cl_loss, requires_grad=True)
                cl_loss.backward()
                cl_optimizer.step()
                
                # 3. State Autoencoding
                ae_optimizer.zero_grad()
                _, ae_metrics = self.state_autoencoder(batch)
                ae_loss = ae_metrics['total_loss']
                ae_loss = torch.tensor(ae_loss, requires_grad=True)
                ae_loss.backward()
                ae_optimizer.step()
            
            # Record metrics
            training_history['mae_loss'].append(mae_loss.item())
            training_history['contrastive_loss'].append(cl_loss.item())
            training_history['autoencoder_loss'].append(ae_loss.item())
        
        return training_history
    
    def encode_state(
        self,
        state: UnifiedState[Any]
    ) -> Dict[str, torch.Tensor]:
        """Encode a quantum state using all SSL components."""
        with torch.no_grad():
            # Get quantum field
            quantum_field = state.quantum_field.unsqueeze(0)
            
            # Get encodings from each component
            _, mae_metrics = self.masked_autoencoder(quantum_field)
            z1, _, cl_metrics = self.contrastive_learner(quantum_field)
            _, ae_metrics = self.state_autoencoder(quantum_field)
            
            encodings = {
                'masked_autoencoder': mae_metrics['latent'],
                'contrastive': z1,
                'autoencoder': ae_metrics['latent']
            }
            
            return encodings 