import asyncio
import logging

from src.trading.engine import TradingEngine

logging.basicConfig(level=logging.INFO)

async def run_test():
    engine = TradingEngine(symbol="BTCUSD", timeframe="15m", paper_trading=True, enable_sentiment_agent=True)
    await engine.initialize()

    # Prepare dummy data
    indicators = {"rsi": 42.5, "macd_hist": -0.1, "ema_9": 67500}
    micro = engine.market_data.get_microstructure_metrics()
    news_data = [
        {"title": "BTC rebound", "summary": "Bitcoin recovers 2%", "source": "coindesk"},
        {"title": "ETH upgrade", "summary": "Ethereum announces upgrade", "source": "cointelegraph"},
    ]
    social_data = {
        "twitter": {
            "APompliano": [{"text": "Big news", "timestamp": "2025-12-07T00:00:00Z"}],
        },
        "reddit": {
            "CryptoCurrency": ["Post 1", "Post 2"]
        }
    }
    fg = "60"

    # Attach a simple event handler to capture agentOutput events
    async def print_event(evt_type, data):
        print(f"EVENT: {evt_type} - {data.get('agent_name')} - extra keys: {list(k for k in data.keys() if k not in ['agent_name', 'data', 'timestamp'])}")

    engine.on_agent_event = print_event

    result = await engine._execute_langgraph_analysis(indicators, micro, news_data, social_data, fg)

    print("--- Sentiment Report ---")
    print(result.get("sentiment_report"))
    print("--- Execution Times ---")
    print(result.get("execution_times"))

if __name__ == '__main__':
    asyncio.run(run_test())
