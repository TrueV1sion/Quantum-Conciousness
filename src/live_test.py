import asyncio
import os
from datetime import datetime
from quantum_finance import QuantumFinanceAnalyzer
from config import SystemConfig

# Finnhub API key (get from environment variable)
API_KEY = os.getenv('FINNHUB_API_KEY')

# Test symbols
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

async def live_data_callback(symbol: str, signal, live_data):
    """Callback for live market updates."""
    print(f"\n{datetime.now()} - {symbol} Update:")
    print(f"Price: ${live_data.price:.2f}")
    print(f"Volume: {live_data.volume:,}")
    print(f"Action: {signal.action.upper()}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Predicted Change: {signal.predicted_price_change:.2%}")
    print(f"Risk Level: {signal.risk_assessment:.2%}")
    print(f"Quantum Coherence: {signal.quantum_coherence:.2%}")
    print("-" * 50)

async def main():
    if not API_KEY:
        print("Please set FINNHUB_API_KEY environment variable")
        return
    
    # Initialize analyzer
    config = SystemConfig(
        unified_dim=128,
        quantum_dim=64,
        consciousness_dim=32
    )
    
    analyzer = QuantumFinanceAnalyzer(
        symbols=SYMBOLS,
        lookback_days=30,
        config=config
    )
    
    # Add callback for live updates
    analyzer.add_live_update_callback(live_data_callback)
    
    print(f"Starting live market analysis for: {', '.join(SYMBOLS)}")
    print("Press Ctrl+C to stop")
    
    try:
        # Start live trading
        await analyzer.start_live_trading(API_KEY)
    except KeyboardInterrupt:
        print("\nStopping live market analysis...")
        analyzer.stop_live_trading()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.stop_live_trading()

if __name__ == "__main__":
    asyncio.run(main()) 