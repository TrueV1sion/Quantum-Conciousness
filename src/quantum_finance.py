import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
import websockets
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, AsyncGenerator, Literal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert

from src.config import SystemConfig, UnifiedState, ProcessingDimension
from src.processors import HybridStateProcessor

# Set OpenMP variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class AdvancedIndicators:
    def __init__(self):
        self.window_size = 20  # Default window size for calculations
    
    def calculate_fdi_katz(self, prices: np.ndarray) -> float:
        """
        Calculate Fractal Dimension Index using Katz's method
        
        Args:
            prices: Array of price data
        Returns:
            float: FDI value
        """
        n = len(prices)
        L = sum(np.sqrt(1 + np.diff(prices) ** 2))  # Length of the curve
        d = np.max(np.abs(prices - prices[0]))      # Max distance from first point
        
        if d == 0:
            return 1.0
            
        return np.log10(n) / (np.log10(d/L) + np.log10(n))
    
    def calculate_fdi_higuchi(self, prices: np.ndarray, k_max: int = 8) -> float:
        """
        Calculate Fractal Dimension Index using Higuchi's algorithm
        
        Args:
            prices: Array of price data
            k_max: Maximum step size
        Returns:
            float: FDI value
        """
        N = len(prices)
        L = np.zeros(k_max)
        x = np.arange(N)
        
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                idx = np.arange(m, N-k, k)
                Lm = np.sum(np.abs(prices[idx + k] - prices[idx]))
                Lk += Lm * (N - 1) / (((N - m) // k) * k)
            L[k-1] = Lk / k
            
        # Calculate slope through linear regression
        x_reg = np.log(1/np.arange(1, k_max + 1))
        y_reg = np.log(L)
        slope = np.polyfit(x_reg, y_reg, 1)[0]
        
        return -slope
    
    def hilbert_transform_features(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Hilbert Transform components
        
        Args:
            prices: Array of price data
        Returns:
            Tuple containing instantaneous phase and frequency
        """
        analytic_signal = hilbert(prices)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
        
        # Pad frequency array to match input length
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'edge')
        
        return instantaneous_phase, instantaneous_frequency
    
    def dominant_cycle_phase(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate the Hilbert Transform Dominant Cycle Phase
        
        Args:
            prices: Array of price data
        Returns:
            np.ndarray: Dominant cycle phase values
        """
        phase, _ = self.hilbert_transform_features(prices)
        return np.mod(phase, 2 * np.pi)

@dataclass
class MarketState:
    """Market state containing technical indicators and analysis results."""
    price: float
    volume: float
    volatility: float
    momentum: float
    rsi: float
    macd: Tuple[float, float]  # (MACD, Signal)
    bollinger_bands: Tuple[float, float, float]  # (Upper, Middle, Lower)
    indicators: Optional['AdvancedIndicators'] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = AdvancedIndicators()

    def process_market_data(self, prices: np.ndarray) -> dict:
        """Process market data with advanced indicators."""
        return {
            'fdi_katz': self.indicators.calculate_fdi_katz(prices),
            'fdi_higuchi': self.indicators.calculate_fdi_higuchi(prices),
            'dominant_cycle_phase': self.indicators.dominant_cycle_phase(prices),
            'instantaneous_phase': (
                self.indicators.hilbert_transform_features(prices)[0]
            ),
            'instantaneous_frequency': (
                self.indicators.hilbert_transform_features(prices)[1]
            )
        }

@dataclass
class LiveMarketData:
    """Real-time market data update."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: float
    ask: float
    last_trade: float

@dataclass
class TradingSignal:
    """Trading signal with quantum-enhanced analysis results."""
    action: Literal['buy', 'sell', 'hold']
    confidence: float  # 0.0 to 1.0
    predicted_price_change: float  # Percentage
    risk_assessment: float  # 0.0 to 1.0
    quantum_coherence: float  # 0.0 to 1.0
    timestamp: datetime

class QuantumFinanceAnalyzer:
    """Advanced financial analysis using quantum processing."""
    
    def __init__(self, symbols: List[str], lookback_days: int = 365,
                 config: Optional[SystemConfig] = None,
                 websocket_url: str = "wss://ws.finnhub.io"):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.websocket_url = websocket_url
        
        # Initialize configuration
        self.config = config or SystemConfig(
            unified_dim=128,
            quantum_dim=64,
            consciousness_dim=32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(self.config.device)
        
        # Initialize quantum processor
        self.quantum_processor = HybridStateProcessor(self.config.num_qubits)
        
        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.live_data: Dict[str, LiveMarketData] = {}
        
        # Analysis parameters
        self.rsi_period = 14
        self.macd_params = (12, 26, 9)  # Fast, Slow, Signal
        self.bollinger_period = 20
        self.volatility_window = 20
        
        # Live trading parameters
        self.is_live_trading = False
        self.live_update_callbacks = []
        
        # Load initial data
        self._load_market_data()
        self._initialize_quantum_system()

    async def start_live_trading(self, api_key: str):
        """Start real-time market data streaming and live trading."""
        self.is_live_trading = True
        uri = f"{self.websocket_url}?token={api_key}"
        
        async with websockets.connect(uri) as websocket:
            # Subscribe to symbols
            subscribe_message = {
                "type": "subscribe",
                "symbol": self.symbols
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Process real-time data
            while self.is_live_trading:
                try:
                    data = await websocket.recv()
                    await self._process_live_data(json.loads(data))
                except Exception as e:
                    print(f"Error processing live data: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on error

    async def _process_live_data(self, data: Dict):
        """Process incoming live market data."""
        if data['type'] == 'trade':
            symbol = data['symbol']
            
            # Update live data
            self.live_data[symbol] = LiveMarketData(
                symbol=symbol,
                price=data['price'],
                volume=data['volume'],
                timestamp=datetime.fromtimestamp(data['timestamp']/1000),
                bid=data.get('bid', 0),
                ask=data.get('ask', 0),
                last_trade=data['price']
            )
            
            # Update market state
            await self._update_live_market_state(symbol)
            
            # Generate new trading signal
            signal = await self.analyze_symbol(symbol)
            
            # Notify callbacks
            for callback in self.live_update_callbacks:
                await callback(symbol, signal, self.live_data[symbol])

    async def _update_live_market_state(self, symbol: str):
        """Update market state with live data."""
        live = self.live_data[symbol]
        df = self.market_data[symbol]
        
        # Add new data point
        new_data = pd.DataFrame({
            'Close': [live.price],
            'Volume': [live.volume],
            'High': [live.price],
            'Low': [live.price],
            'Open': [live.price]
        }, index=[live.timestamp])
        
        # Update dataframe
        df = pd.concat([df, new_data])
        
        # Recalculate technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(
            window=self.volatility_window
        ).std()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
        bb_result = self._calculate_bollinger_bands(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = bb_result
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Update stored data
        self.market_data[symbol] = df
        self._update_market_state(symbol)

    async def get_live_updates(self) -> AsyncGenerator[Tuple[str, TradingSignal, LiveMarketData], None]:
        """Generator for live market updates."""
        while self.is_live_trading:
            for symbol in self.symbols:
                if symbol in self.live_data:
                    signal = await self.analyze_symbol(symbol)
                    yield symbol, signal, self.live_data[symbol]
            await asyncio.sleep(1)  # Prevent tight loop

    def add_live_update_callback(self, callback):
        """Add callback for live market updates."""
        self.live_update_callbacks.append(callback)

    def remove_live_update_callback(self, callback):
        """Remove callback for live market updates."""
        if callback in self.live_update_callbacks:
            self.live_update_callbacks.remove(callback)

    def stop_live_trading(self):
        """Stop live trading."""
        self.is_live_trading = False
    
    def _initialize_quantum_system(self):
        """Initialize the quantum processing system."""
        # Create initial quantum state
        quantum_field = torch.randn(1, self.config.num_qubits, device=self.device)
        quantum_field = quantum_field / torch.norm(quantum_field)
        
        # Create initial coherence matrix
        coherence_matrix = torch.eye(self.config.num_qubits, device=self.device)
        
        # Create initial quantum state
        quantum_state = UnifiedState(
            quantum_field=quantum_field,
            consciousness_field=torch.zeros(1, 32, device=self.device),
            unified_field=None,
            coherence_matrix=coherence_matrix,
            resonance_patterns={'market': quantum_field},
            dimensional_signatures={
                ProcessingDimension.PHYSICAL: 1.0,
                ProcessingDimension.QUANTUM: 0.8,
                ProcessingDimension.CONSCIOUSNESS: 0.0,
                ProcessingDimension.TEMPORAL: 1.0,
                ProcessingDimension.INFORMATIONAL: 0.9,
                ProcessingDimension.UNIFIED: 0.7,
                ProcessingDimension.TRANSCENDENT: 0.0
            },
            temporal_phase=0.0,
            entanglement_map={'market': 1.0},
            wavelet_coefficients=None,
            metadata={'initialization_time': datetime.now()}
        )
        
        # Initialize quantum processor
        self.quantum_processor.initialize_state(quantum_state)
    
    def _load_market_data(self):
        """Load historical market data for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Calculate technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(
                window=self.volatility_window
            ).std()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'], df['Signal'] = self._calculate_macd(df['Close'])
            bb_result = self._calculate_bollinger_bands(df['Close'])
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = bb_result
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Add quantum-relevant features
            df['Price_Norm'] = df['Close'] / df['Close'].mean()
            df['Volume_Norm'] = df['Volume'] / df['Volume'].mean()
            df['Market_Phase'] = np.angle(
                df['Returns'].fillna(0) + 1j * df['Volatility'].fillna(0)
            )
            
            self.market_data[symbol] = df
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index."""
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_losses = losses.rolling(window=self.rsi_period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.replace([np.inf, -np.inf], 100)
        rsi = rsi.fillna(50)  # Fill NaN with neutral value
        
        return rsi
    
    def _calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        fast = prices.ewm(span=self.macd_params[0]).mean()
        slow = prices.ewm(span=self.macd_params[1]).mean()
        macd = fast - slow
        signal = macd.ewm(span=self.macd_params[2]).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=self.bollinger_period, min_periods=1).mean()
        std = prices.rolling(window=self.bollinger_period, min_periods=1).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        
        # Fill NaN values with first valid observation
        middle = middle.fillna(method='bfill')
        upper = upper.fillna(method='bfill')
        lower = lower.fillna(method='bfill')
        
        return upper, middle, lower
    
    def _update_market_state(self, symbol: str):
        """Update market state for a symbol."""
        df = self.market_data[symbol]
        latest = df.iloc[-1]
        
        self.market_states[symbol] = MarketState(
            price=float(latest['Close']),  # Convert to float
            volume=float(latest['Volume']),  # Convert to float
            volatility=float(latest['Volatility']),  # Convert to float
            momentum=float(latest['Momentum']),  # Convert to float
            rsi=float(latest['RSI']),  # Convert to float
            macd=(float(latest['MACD']), float(latest['Signal'])),  # Convert to float tuple
            bollinger_bands=(
                float(latest['BB_Upper']),
                float(latest['BB_Middle']),
                float(latest['BB_Lower'])
            )  # Convert to float tuple
        )
    
    def create_quantum_state(self, symbol: str) -> UnifiedState:
        """Convert market state to quantum state for processing."""
        state = self.market_states[symbol]
        df = self.market_data[symbol]
        
        # Create normalized feature vector
        features = [
            state.price / float(df['Close'].mean()),
            state.volume / float(df['Volume'].mean()),
            state.volatility / float(df['Volatility'].mean()),
            state.momentum / float(df['Momentum'].abs().mean()),
            state.rsi / 100.0,
            state.macd[0] / float(df['MACD'].abs().mean() or 1.0),  # Handle zero division
            state.macd[1] / float(df['Signal'].abs().mean() or 1.0),  # Handle zero division
            (state.bollinger_bands[0] - state.price) / (state.price or 1.0),  # Handle zero division
            (state.price - state.bollinger_bands[2]) / (state.price or 1.0)  # Handle zero division
        ]
        
        # Handle NaN and inf values
        features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in features]
        
        # Pad features to match quantum_dim
        features = features + [0.0] * (self.config.quantum_dim - len(features))
        
        # Create quantum field
        quantum_field = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        quantum_field = quantum_field / (torch.norm(quantum_field) + 1e-8)
        
        # Create coherence matrix
        coherence_matrix = torch.eye(self.config.quantum_dim, device=self.device)
        
        return UnifiedState(
            quantum_field=quantum_field,
            consciousness_field=torch.zeros(1, 32, device=self.device),
            unified_field=None,
            coherence_matrix=coherence_matrix,
            resonance_patterns={'market': quantum_field},
            dimensional_signatures={
                ProcessingDimension.PHYSICAL: 1.0,
                ProcessingDimension.QUANTUM: 0.8,
                ProcessingDimension.CONSCIOUSNESS: 0.0,
                ProcessingDimension.TEMPORAL: 1.0,
                ProcessingDimension.INFORMATIONAL: 0.9,
                ProcessingDimension.UNIFIED: 0.7,
                ProcessingDimension.TRANSCENDENT: 0.0
            },
            temporal_phase=0.0,
            entanglement_map={'market': 1.0},
            wavelet_coefficients=None,
            metadata={'symbol': symbol}
        )
    
    async def analyze_symbol(self, symbol: str) -> TradingSignal:
        """Perform quantum-enhanced market analysis for a symbol."""
        # Update market state
        self._update_market_state(symbol)
        
        # Create and process quantum state
        quantum_state = self.create_quantum_state(symbol)
        processed_state = await self.quantum_processor.process_state(quantum_state)
        
        # Extract quantum influences
        coherence = float(processed_state.coherence_matrix.mean())
        quantum_field = processed_state.quantum_field[0].detach().cpu().numpy()
        
        # Calculate trading signals
        state = self.market_states[symbol]
        df = self.market_data[symbol]
        
        # Combine classical and quantum signals
        rsi_signal = -1 if state.rsi > 70 else 1 if state.rsi < 30 else 0
        macd_signal = 1 if state.macd[0] > state.macd[1] else -1
        bb_position = (state.price - state.bollinger_bands[1]) / (
            state.bollinger_bands[0] - state.bollinger_bands[1]
        )
        bb_signal = -1 if bb_position > 0.8 else 1 if bb_position < -0.8 else 0
        
        # Combine signals with quantum influence
        quantum_signal = np.mean(quantum_field[:10])  # Use first 10 quantum features
        combined_signal = (
            0.3 * rsi_signal +
            0.3 * macd_signal +
            0.2 * bb_signal +
            0.2 * quantum_signal
        ) * coherence
        
        # Generate trading signal
        if combined_signal > 0.3:
            action = 'buy'
        elif combined_signal < -0.3:
            action = 'sell'
        else:
            action = 'hold'
        
        # Calculate confidence and risk
        confidence = abs(combined_signal)
        risk = state.volatility * (1 - coherence)
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            predicted_price_change=combined_signal * state.volatility,
            risk_assessment=risk,
            quantum_coherence=coherence,
            timestamp=datetime.now()
        )
    
    def _get_previous_signal(self, symbol: str) -> str:
        """Get previous trading signal for hysteresis."""
        if not hasattr(self, '_signal_history'):
            self._signal_history = {}
        return self._signal_history.get(symbol, 'hold')
    
    def _store_signal(self, symbol: str, action: str):
        """Store trading signal for hysteresis."""
        if not hasattr(self, '_signal_history'):
            self._signal_history = {}
        self._signal_history[symbol] = action
    
    def plot_analysis(self, symbol: str, signal: TradingSignal):
        """Create interactive plot of the analysis."""
        try:
            df = self.market_data[symbol]
            
            # Create subplots
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price', 'RSI', 'MACD', 'Volume')
            )
            
            # Price with Bollinger Bands
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    line=dict(color='gray', dash='dash'),
                    name='BB Upper'
                ),
                row=1,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    line=dict(color='gray', dash='dash'),
                    name='BB Lower',
                    fill='tonexty'
                ),
                row=1,
                col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    line=dict(color='purple'),
                    name='RSI'
                ),
                row=2,
                col=1
            )
            
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                row=2,
                col=1
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                row=2,
                col=1
            )
            
            # MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    line=dict(color='blue'),
                    name='MACD'
                ),
                row=3,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Signal'],
                    line=dict(color='orange'),
                    name='Signal'
                ),
                row=3,
                col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume'
                ),
                row=4,
                col=1
            )
            
            # Add signal annotation
            fig.add_annotation(
                text=(
                    f"Signal: {signal.action.upper()}<br>"
                    f"Confidence: {signal.confidence:.2f}<br>"
                    f"Risk: {signal.risk_assessment:.2f}"
                ),
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                showarrow=False,
                font=dict(size=14, color="white"),
                bgcolor={"buy": "green", "sell": "red", "hold": "gray"}[
                    signal.action
                ],
                bordercolor="white",
                borderwidth=2,
                borderpad=4,
                align="right"
            )
            
            # Update layout
            fig.update_layout(
                title=(
                    f"{symbol} Analysis with Quantum Processing "
                    f"(Coherence: {signal.quantum_coherence:.2f})"
                ),
                height=1200,
                showlegend=True,
                xaxis4_title="Date",
                yaxis_title="Price",
                yaxis2_title="RSI",
                yaxis3_title="MACD",
                yaxis4_title="Volume"
            )
            
            return fig
        except Exception as e:
            print(f"Error plotting analysis for {symbol}: {str(e)}")
            raise

    def _calculate_technical_indicators(self, df):
        """Calculate advanced technical indicators."""
        # Existing indicators...
        df['FDI'] = self._calculate_fractal_dimension_index(df['Close'])
        df['Hilbert_Phase'] = self._calculate_hilbert_phase(df['Close'])
        # Update market state with new indicators

    def _calculate_fractal_dimension_index(self, series):
        """Calculate Fractal Dimension Index (FDI)."""
        # Implement the calculation of FDI
        pass

    def _calculate_hilbert_phase(self, series):
        """Calculate Hilbert Transform instantaneous phase."""
        analytic_signal = hilbert(series)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        return instantaneous_phase


async def main():
    try:
        # Initialize analyzer with symbols
        analyzer = QuantumFinanceAnalyzer(['AAPL', 'GOOGL', 'MSFT'])
        
        # Analyze each symbol
        for symbol in analyzer.symbols:
            try:
                signal = await analyzer.analyze_symbol(symbol)
                print(f"\nAnalysis for {symbol}:")
                print(f"Action: {signal.action}")
                print(f"Confidence: {signal.confidence:.2f}")
                print(f"Predicted Price Change: {signal.predicted_price_change:.2f}%")
                print(f"Risk Assessment: {signal.risk_assessment:.2f}")
                print(f"Quantum Coherence: {signal.quantum_coherence:.2f}")
                
                # Create and show plot
                fig = analyzer.plot_analysis(symbol, signal)
                fig.show()
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
