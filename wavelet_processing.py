# wavelet_processing.py

import pywt
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from enum import Enum
from config import UnifiedState
from utils import calculate_entropy, find_dominant_frequency
from exceptions import WaveletProcessingError
import asyncio

class WaveletType(Enum):
    """Supported wavelet types."""
    DAUBECHIES = 'db4'
    SYMLET = 'sym8'
    COIFLET = 'coif3'
    MEXICAN_HAT = 'mexh'
    MORLET = 'morl'
    CONSCIOUSNESS = 'consciousness'  # Custom wavelet for consciousness fields

@dataclass
class WaveletConfig:
    """Configuration for wavelet processing."""
    wavelet_type: WaveletType
    max_level: int
    threshold_rule: str = 'soft'
    mode: str = 'symmetric'
    consciousness_parameters: Optional[Dict[str, float]] = None

class FourierProcessor:
    """Process fields using Fourier analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process field using Fourier transforms.
        
        Args:
            field: Input tensor field
            
        Returns:
            Dictionary containing FFT results and spectra
        """
        try:
            # Ensure field is properly shaped for FFT
            if len(field.shape) == 1:
                field = field.unsqueeze(0)
            
            fft_result = torch.fft.fft(field)
            power_spectrum = torch.abs(fft_result) ** 2
            phase_spectrum = torch.angle(fft_result)
            return {
                'fft': fft_result,
                'power_spectrum': power_spectrum,
                'phase_spectrum': phase_spectrum
            }
        except Exception as e:
            self.logger.error(f"Fourier processing failed: {str(e)}")
            raise WaveletProcessingError(f"Fourier processing failed: {str(e)}")

class WaveletProcessor:
    """Advanced wavelet processing for quantum-consciousness fields."""
    
    def __init__(self, config: WaveletConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._init_wavelets()
        self.fourier_processor = FourierProcessor()
    
    def _init_wavelets(self):
        """Initialize wavelet transforms based on configuration."""
        if self.config.wavelet_type == WaveletType.CONSCIOUSNESS:
            self.wavelet = self._create_consciousness_wavelet()
        else:
            self.wavelet = pywt.Wavelet('db4')
    
    def _create_consciousness_wavelet(self) -> pywt.Wavelet:
        """Create custom wavelet for consciousness field processing."""
        return pywt.Wavelet('db4')
    
    async def process_unified_state(self, state: UnifiedState) -> UnifiedState:
        """
        Process quantum and consciousness fields using wavelet analysis.
        
        Args:
            state: UnifiedState instance to process
            
        Returns:
            Processed UnifiedState instance
        """
        try:
            # Handle batched input - process first batch item
            quantum_field = state.quantum_field[0].cpu()  # Get first batch item
            consciousness_field = state.consciousness_field[0].cpu()  # Get first batch item
            
            # Process quantum field
            quantum_coeffs = self._process_quantum_field(quantum_field)
            
            # Process consciousness field
            consciousness_coeffs = self._process_consciousness_field(consciousness_field)
            
            # Analyze cross-field coherence
            coherence_analysis = self._analyze_coherence(quantum_coeffs, consciousness_coeffs)
            
            # Detect patterns
            patterns = self._detect_patterns(quantum_coeffs, consciousness_coeffs)
            
            # Update state with wavelet analysis results
            state.wavelet_coefficients = {
                'quantum': quantum_coeffs,
                'consciousness': consciousness_coeffs,
                'coherence_analysis': coherence_analysis,
                'patterns': patterns
            }
            
            # Keep batch dimension in state fields
            state.quantum_field = state.quantum_field.to(quantum_field.device)
            state.consciousness_field = state.consciousness_field.to(consciousness_field.device)
            
            return state
        except Exception as e:
            self.logger.error(f"Wavelet processing failed: {str(e)}")
            raise WaveletProcessingError(str(e))
    
    def _process_quantum_field(self, field: torch.Tensor) -> Dict[str, Any]:
        """Process quantum field using wavelet transform."""
        field_np = field.detach().cpu().numpy()
        coeffs = pywt.wavedec(field_np, self.wavelet, level=self.config.max_level, mode=self.config.mode)
        analysis = self._analyze_coefficients(coeffs, 'quantum')
        return {
            'coefficients': coeffs,
            'analysis': analysis
        }
    
    def _process_consciousness_field(self, field: torch.Tensor) -> Dict[str, Any]:
        """Process consciousness field using specialized wavelet transform."""
        field_np = field.detach().cpu().numpy()
        coeffs = pywt.wavedec(field_np, self.wavelet, level=self.config.max_level, mode=self.config.mode)
        analysis = self._analyze_coefficients(coeffs, 'consciousness')
        return {
            'coefficients': coeffs,
            'analysis': analysis
        }
    
    def _analyze_coefficients(self, coeffs: List[np.ndarray], field_type: str) -> Dict[str, Any]:
        """Analyze wavelet coefficients."""
        analysis = {}
        
        # Ensure all coefficient arrays are properly sized
        max_size = max(c.size for c in coeffs)
        padded_coeffs = []
        for c in coeffs:
            if c.size < max_size:
                pad_width = max_size - c.size
                padded = np.pad(c, (0, pad_width), mode='constant')
                padded_coeffs.append(padded)
            else:
                padded_coeffs.append(c[:max_size])
        
        # Energy distribution
        total_energy = sum(np.sum(np.abs(c)**2) for c in padded_coeffs)
        energy_distribution = {
            f'level_{i}': float(np.sum(np.abs(c)**2) / total_energy) 
            for i, c in enumerate(padded_coeffs)
        }
        analysis['energy_distribution'] = energy_distribution
        
        # Coefficient statistics
        coefficient_statistics = {}
        for i, c in enumerate(padded_coeffs):
            stats = {
                'mean': float(np.mean(np.abs(c))),
                'std': float(np.std(c)),
                'max': float(np.max(np.abs(c))),
                'entropy': calculate_entropy(torch.tensor(c))
            }
            coefficient_statistics[f'level_{i}'] = stats
        analysis['coefficient_statistics'] = coefficient_statistics
        
        # Field-specific analysis
        if field_type == 'quantum':
            analysis.update(self._analyze_quantum_coefficients(padded_coeffs))
        else:
            analysis.update(self._analyze_consciousness_coefficients(padded_coeffs))
        
        return analysis
    
    def _analyze_quantum_coefficients(self, coeffs: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze quantum-specific properties of coefficients."""
        return {
            'quantum_coherence': self._calculate_quantum_coherence(coeffs),
            'entanglement_measure': self._calculate_entanglement(coeffs),
            'quantum_patterns': self._identify_quantum_patterns(coeffs)
        }
    
    def _analyze_consciousness_coefficients(self, coeffs: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze consciousness-specific properties of coefficients."""
        return {
            'consciousness_coherence': self._calculate_consciousness_coherence(coeffs),
            'field_stability': self._calculate_field_stability(coeffs),
            'consciousness_patterns': self._identify_consciousness_patterns(coeffs)
        }
    
    def _analyze_coherence(self, quantum_coeffs: Dict[str, Any], consciousness_coeffs: Dict[str, Any]) -> Dict[str, float]:
        """Analyze coherence between quantum and consciousness coefficients."""
        coherence_measures = {}
        
        q_coeffs = quantum_coeffs['coefficients']
        c_coeffs = consciousness_coeffs['coefficients']
        
        # Ensure coefficients have same size for correlation
        min_size = min(len(q_coeffs), len(c_coeffs))
        for level in range(min_size):
            q_coeff = q_coeffs[level]
            c_coeff = c_coeffs[level]
            
            # Pad or trim coefficients to match sizes
            max_size = max(q_coeff.size, c_coeff.size)
            if q_coeff.size < max_size:
                q_coeff = np.pad(q_coeff, (0, max_size - q_coeff.size), mode='constant')
            else:
                q_coeff = q_coeff[:max_size]
                
            if c_coeff.size < max_size:
                c_coeff = np.pad(c_coeff, (0, max_size - c_coeff.size), mode='constant')
            else:
                c_coeff = c_coeff[:max_size]
            
            correlation = np.corrcoef(q_coeff.flatten(), c_coeff.flatten())[0,1]
            coherence_measures[f'level_{level}'] = float(correlation)
        
        return coherence_measures
    
    def _detect_patterns(self, quantum_coeffs: Dict[str, Any], consciousness_coeffs: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in wavelet coefficients."""
        patterns = {}
        
        q_coeffs = quantum_coeffs['coefficients']
        c_coeffs = consciousness_coeffs['coefficients']
        
        min_size = min(len(q_coeffs), len(c_coeffs))
        for level in range(min_size):
            q_coeff = q_coeffs[level]
            c_coeff = c_coeffs[level]
            
            # Ensure coefficients have same size
            max_size = max(q_coeff.size, c_coeff.size)
            if q_coeff.size < max_size:
                q_coeff = np.pad(q_coeff, (0, max_size - q_coeff.size), mode='constant')
            else:
                q_coeff = q_coeff[:max_size]
                
            if c_coeff.size < max_size:
                c_coeff = np.pad(c_coeff, (0, max_size - c_coeff.size), mode='constant')
            else:
                c_coeff = c_coeff[:max_size]
            
            # Find dominant frequencies
            q_freq = find_dominant_frequency(torch.tensor(q_coeff))
            c_freq = find_dominant_frequency(torch.tensor(c_coeff))
            
            patterns[f'level_{level}'] = {
                'quantum_dominant_freq': q_freq,
                'consciousness_dominant_freq': c_freq,
                'frequency_ratio': float(q_freq / c_freq) if c_freq != 0 else float('inf')
            }
        
        return patterns
    
    def _calculate_quantum_coherence(self, coeffs: List[np.ndarray]) -> float:
        """Calculate quantum coherence from coefficients."""
        return float(np.mean([np.mean(np.abs(c)) for c in coeffs]))
    
    def _calculate_entanglement(self, coeffs: List[np.ndarray]) -> float:
        """Calculate entanglement measure from coefficients."""
        return float(np.mean([np.var(c) for c in coeffs]))
    
    def _calculate_consciousness_coherence(self, coeffs: List[np.ndarray]) -> float:
        """Calculate consciousness coherence from coefficients."""
        return float(np.mean([np.mean(np.abs(c)) for c in coeffs]))
    
    def _calculate_field_stability(self, coeffs: List[np.ndarray]) -> float:
        """Calculate field stability from coefficients."""
        return float(np.mean([np.std(c) for c in coeffs]))
    
    def _identify_quantum_patterns(self, coeffs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Identify quantum-specific patterns."""
        patterns = []
        for i, c in enumerate(coeffs):
            pattern = {
                'level': i,
                'complexity': calculate_entropy(torch.tensor(c)),
                'stability': float(1.0 / (np.std(c) + 1e-10)),
                'intensity': float(np.mean(np.abs(c)))
            }
            patterns.append(pattern)
        return patterns
    
    def _identify_consciousness_patterns(self, coeffs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Identify consciousness-specific patterns."""
        patterns = []
        for i, c in enumerate(coeffs):
            pattern = {
                'level': i,
                'coherence': float(np.mean(np.abs(c))),
                'variability': float(np.std(c)),
                'energy': float(np.sum(c**2))
            }
            patterns.append(pattern)
        return patterns
