import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from src.quantum_finance import QuantumFinanceAnalyzer
from src.config import SystemConfig, UnifiedState

@pytest.fixture
def config():
    """Create test configuration."""
    return SystemConfig(
        unified_dim=128,
        quantum_dim=64,
        consciousness_dim=32
    )

@pytest.fixture
def mock_data():
    """Create mock market data."""
    dates = pd.date_range(end=datetime.now(), periods=30)
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 30),
        'High': np.random.uniform(150, 250, 30),
        'Low': np.random.uniform(90, 180, 30),
        'Close': np.random.uniform(100, 200, 30),
        'Volume': np.random.uniform(1000000, 5000000, 30)
    }, index=dates)
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['RSI'] = 50 + np.random.normal(0, 10, 30)
    
    # Calculate Bollinger Bands
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma + (std * 2)
    df['BB_Middle'] = sma
    df['BB_Lower'] = sma - (std * 2)
    
    # Add quantum-relevant features
    df['Price_Norm'] = df['Close'] / df['Close'].mean()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].mean()
    df['Market_Phase'] = np.angle(
        df['Returns'].fillna(0) + 1j * df['Volatility'].fillna(0)
    )
    
    return df

@pytest.fixture
def analyzer(config, mock_data):
    """Create a test instance of QuantumFinanceAnalyzer."""
    analyzer = QuantumFinanceAnalyzer(['AAPL'], lookback_days=30, config=config)
    analyzer.market_data['AAPL'] = mock_data
    analyzer._update_market_state('AAPL')
    return analyzer

@pytest.mark.asyncio
async def test_market_data_loading(analyzer):
    """Test market data loading functionality."""
    assert len(analyzer.symbols) == 1
    assert 'AAPL' in analyzer.market_data
    assert not analyzer.market_data['AAPL'].empty

@pytest.mark.asyncio
async def test_technical_indicators(analyzer, mock_data):
    """Test technical indicator calculations."""
    # Test RSI
    rsi = analyzer._calculate_rsi(mock_data['Close'])
    assert isinstance(rsi, pd.Series)
    assert (rsi >= 0).all() and (rsi <= 100).all()
    
    # Test MACD
    macd, signal = analyzer._calculate_macd(mock_data['Close'])
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    
    # Test Bollinger Bands
    upper, middle, lower = analyzer._calculate_bollinger_bands(mock_data['Close'])
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert (upper >= middle).all()
    assert (middle >= lower).all()

@pytest.mark.asyncio
async def test_quantum_state_creation(analyzer):
    """Test quantum state creation."""
    analyzer._update_market_state('AAPL')
    quantum_state = analyzer.create_quantum_state('AAPL')
    
    assert isinstance(quantum_state, UnifiedState)
    assert quantum_state.quantum_field.shape == (1, analyzer.config.quantum_dim)
    assert quantum_state.consciousness_field.shape == (1, analyzer.config.consciousness_dim)
    assert isinstance(quantum_state.coherence_matrix, torch.Tensor)
    
    # Test normalization
    field_norm = torch.norm(quantum_state.quantum_field)
    assert torch.allclose(field_norm, torch.tensor(1.0), atol=1e-6)

@pytest.mark.asyncio
async def test_signal_generation(analyzer):
    """Test trading signal generation."""
    signal = await analyzer.analyze_symbol('AAPL')
    
    assert signal.action in ['buy', 'sell', 'hold']
    assert 0 <= signal.confidence <= 1
    assert isinstance(signal.predicted_price_change, float)
    assert isinstance(signal.risk_assessment, float)
    assert 0 <= signal.quantum_coherence <= 1

@pytest.mark.asyncio
async def test_visualization(analyzer):
    """Test visualization functionality."""
    signal = await analyzer.analyze_symbol('AAPL')
    fig = analyzer.plot_analysis('AAPL', signal)
    
    assert fig is not None
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0  # Should have multiple traces

@pytest.mark.asyncio
async def test_quantum_processing(analyzer):
    """Test quantum processing functionality."""
    analyzer._update_market_state('AAPL')
    initial_state = analyzer.create_quantum_state('AAPL')
    processed_state = await analyzer.quantum_processor.process_state(initial_state)
    
    assert isinstance(processed_state, UnifiedState)
    assert processed_state.quantum_field.shape == initial_state.quantum_field.shape
    
    # Test field normalization
    processed_norm = torch.norm(processed_state.quantum_field)
    assert torch.allclose(processed_norm, torch.tensor(1.0), atol=1e-6)
    
    # Test coherence matrix properties
    assert processed_state.coherence_matrix.shape == (1, 1)  # Should be a scalar for market analysis
    assert 0 <= float(processed_state.coherence_matrix) <= 1  # Coherence should be normalized

if __name__ == '__main__':
    pytest.main(['-v', 'test_quantum_finance.py']) 