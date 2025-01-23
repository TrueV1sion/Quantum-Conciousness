import sys
sys.path.append("..")
from src.quantum_finance import QuantumFinanceAnalyzer
from src.config import SystemConfig
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialAgent:
    def __init__(self, config):
        self.config = config
        self.quantum_finance_analyzer = QuantumFinanceAnalyzer(['AAPL', 'GOOGL', 'MSFT'], config=config)

    async def analyze_markets(self):
        logging.info("Starting market analysis...")
        for symbol in self.quantum_finance_analyzer.symbols:
            try:
                signal = await self.quantum_finance_analyzer.analyze_symbol(symbol)
                logging.info(f"Analysis for {symbol}:")
                logging.info(f"Action: {signal.action}")
                logging.info(f"Confidence: {signal.confidence:.2f}")
                logging.info(f"Predicted Price Change: {signal.predicted_price_change:.2f}%")
                logging.info(f"Risk Assessment: {signal.risk_assessment:.2f}")
                logging.info(f"Quantum Coherence: {signal.quantum_coherence:.2f}")
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
        logging.info("Market analysis finished.")

    async def run(self):
        logging.info("Starting financial agent...")
        await self.analyze_markets()
        logging.info("Financial agent finished.")

if __name__ == "__main__":
    config = SystemConfig()
    agent = FinancialAgent(config)
    asyncio.run(agent.run())
