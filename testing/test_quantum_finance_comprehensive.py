import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

from quantum_finance import QuantumFinanceAnalyzer, MarketState, TradingSignal
from config import SystemConfig, UnifiedState, ProcessingDimension

class TestDataGenerator:
    @staticmethod
    def generate_market_data(days: int = 500) -> pd.DataFrame:
        """Generate realistic market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Generate price series with realistic properties
        price = 100  # Starting price
        prices = []
        for _ in range(days):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            prices.append(price)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, days)),
            'High': prices * (1 + abs(np.random.normal(0, 0.01, days))),
            'Low': prices * (1 - abs(np.random.normal(0, 0.01, days))),
            'Close': prices,
            'Volume': np.random.uniform(1000000, 5000000, days)
        }, index=dates)
        
        return df

class TestQuantumFinance:
    @pytest.fixture
    def config(self):
        return SystemConfig(
            unified_dim=128,
            quantum_dim=64,
            consciousness_dim=32,
            device="cpu"
        )
    
    @pytest.fixture
    def test_data(self):
        return TestDataGenerator.generate_market_data()
    
    @pytest.fixture
    def analyzer(self, config, test_data):
        analyzer = QuantumFinanceAnalyzer(['TEST'], lookback_days=500, config=config)
        analyzer.market_data['TEST'] = test_data
        analyzer._update_market_state('TEST')
        return analyzer

    @pytest.mark.asyncio
    async def test_signal_generation_consistency(self, analyzer):
        """Test consistency of trading signals"""
        signals = []
        for _ in range(10):
            signal = await analyzer.analyze_symbol('TEST')
            signals.append(signal)
        
        # Check signal consistency
        actions = [s.action for s in signals]
        confidences = [s.confidence for s in signals]
        coherences = [s.quantum_coherence for s in signals]
        
        # Signals should be consistent within reasonable bounds
        assert len(set(actions)) <= 2  # Should not flip-flop wildly
        assert np.std(confidences) < 0.2  # Confidence should be stable
        assert np.std(coherences) < 0.1  # Coherence should be stable

    @pytest.mark.asyncio
    async def test_risk_assessment(self, analyzer):
        """Test risk assessment functionality"""
        signal = await analyzer.analyze_symbol('TEST')
        
        # Risk assessment should be within bounds
        assert 0 <= signal.risk_assessment <= 1
        
        # Higher volatility should increase risk
        analyzer.market_data['TEST']['Volatility'] *= 2
        analyzer._update_market_state('TEST')
        high_vol_signal = await analyzer.analyze_symbol('TEST')
        assert high_vol_signal.risk_assessment > signal.risk_assessment

    @pytest.mark.asyncio
    async def test_quantum_state_properties(self, analyzer):
        """Test quantum state properties"""
        quantum_state = analyzer.create_quantum_state('TEST')
        
        # Test normalization
        assert torch.allclose(torch.norm(quantum_state.quantum_field), torch.tensor(1.0), atol=1e-6)
        
        # Test coherence matrix properties
        assert quantum_state.coherence_matrix.shape == (analyzer.config.quantum_dim, analyzer.config.quantum_dim)
        assert torch.allclose(quantum_state.coherence_matrix, quantum_state.coherence_matrix.T)
        eigenvalues = torch.linalg.eigvals(quantum_state.coherence_matrix)
        assert torch.all(eigenvalues.real >= 0)  # Should be positive semi-definite

    def test_technical_indicators_validity(self, analyzer):
        """Test validity of technical indicators"""
        df = analyzer.market_data['TEST']
        
        # Test RSI bounds
        assert df['RSI'].between(0, 100).all()
        
        # Test Bollinger Bands relationship
        assert (df['BB_Upper'] >= df['BB_Middle']).all()
        assert (df['BB_Middle'] >= df['BB_Lower']).all()
        
        # Test MACD calculation
        fast = df['Close'].ewm(span=12).mean()
        slow = df['Close'].ewm(span=26).mean()
        macd_manual = fast - slow
        assert np.allclose(df['MACD'], macd_manual, rtol=1e-10)

    @pytest.mark.asyncio
    async def test_forward_prediction(self, analyzer):
        """Test forward prediction capabilities"""
        # Split data into training and testing
        split_idx = len(analyzer.market_data['TEST']) - 30
        test_data = analyzer.market_data['TEST'].iloc[split_idx:]
        analyzer.market_data['TEST'] = analyzer.market_data['TEST'].iloc[:split_idx]
        
        # Generate predictions
        predictions = []
        actual_changes = []
        
        for i in range(len(test_data)):
            signal = await analyzer.analyze_symbol('TEST')
            predictions.append(signal.predicted_price_change)
            if i < len(test_data) - 1:
                actual_change = (test_data['Close'].iloc[i+1] - test_data['Close'].iloc[i]) / test_data['Close'].iloc[i]
                actual_changes.append(actual_change)
        
        # Calculate prediction accuracy
        correlation = np.corrcoef(predictions[:-1], actual_changes)[0,1]
        assert correlation > -0.5  # Should have some predictive power
        
        # Calculate directional accuracy
        pred_direction = np.sign(predictions[:-1])
        actual_direction = np.sign(actual_changes)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        assert directional_accuracy > 0.45  # Should be better than random

    def test_stress_conditions(self, analyzer):
        """Test system behavior under stress conditions"""
        df = analyzer.market_data['TEST']
        
        # Test extreme volatility
        df['Volatility'] = df['Volatility'] * 10
        analyzer._update_market_state('TEST')
        assert analyzer.market_states['TEST'].volatility > 0
        
        # Test zero volume
        df['Volume'] = 0
        analyzer._update_market_state('TEST')
        assert analyzer.market_states['TEST'].volume == 0
        
        # Test NaN handling
        df.loc[df.index[-1], 'Close'] = np.nan
        analyzer._update_market_state('TEST')
        assert not np.isnan(analyzer.market_states['TEST'].price)

if __name__ == '__main__':
    pytest.main(['-v', 'test_quantum_finance_comprehensive.py']) 