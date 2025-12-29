#!/usr/bin/env python3
"""
Direct test of Visual Agent with chart image.
This tests the full flow: chart_generator -> base64 -> Ollama vision model.
"""
import asyncio
import pytest
import base64
import logging
import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("VisualTest")

@pytest.mark.asyncio
async def test_visual_agent():
    """Test visual agent with a real chart image."""
    
    # 1. Get market data
    logger.info("📊 Step 1: Getting market data...")
    from src.trading.binance_client import BinanceClient
    
    client = BinanceClient(testnet=False)
    await client.connect()
    
    klines = await client.get_klines("BTCUSD", "15m", limit=100)
    await client.close()
    
    logger.info(f"   Got {len(klines)} candles")
    
    # 2. Generate chart
    logger.info("📈 Step 2: Generating chart...")
    from src.tools.chart_generator import get_chart_generator
    
    chart_gen = get_chart_generator()
    
    kline_data = {
        "open": [float(k["open"]) for k in klines],
        "high": [float(k["high"]) for k in klines],
        "low": [float(k["low"]) for k in klines],
        "close": [float(k["close"]) for k in klines],
        "volume": [float(k["volume"]) for k in klines],
        "timestamp": [k.get("timestamp", i) for i, k in enumerate(klines)]
    }
    
    result = chart_gen.generate_chart(
        kline_data=kline_data,
        symbol="BTCUSD",
        timeframe="15m",
        show_indicators=["ema", "bollinger", "supertrend"],
        show_trendlines=True,
        last_n_candles=50
    )
    
    chart_b64 = result.get("image_b64")
    if not chart_b64:
        logger.error("❌ Chart generation failed - no base64 image")
        return
    
    logger.info(f"   Chart base64 length: {len(chart_b64)} chars")
    logger.info(f"   Chart saved to: {result.get('filepath')}")
    logger.info(f"   Indicators: {result.get('indicators_summary', {}).keys()}")
    
    # 3. Test Ollama vision model directly
    logger.info("🖼️ Step 3: Testing Ollama vision model directly...")
    
    import httpx
    
    # Create prompt for vision model
    visual_prompt = """Analiza esta imagen de un gráfico de velas de BTCUSD.

Observa los siguientes elementos:
1. Patrón de velas (verdes=alcistas, rojas=bajistas)
2. EMAs (líneas de colores)
3. Bollinger Bands (bandas superior e inferior)
4. SuperTrend (línea verde=alcista, roja=bajista)

Responde con un JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.5,
  "visual_analysis": "Descripción de lo que ves en el gráfico",
  "trend_direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "key_observations": ["observación 1", "observación 2"]
}
"""
    
    # Prepare Ollama request with vision
    ollama_url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": "qwen3-vl:8b",
        "messages": [
            {
                "role": "user",
                "content": visual_prompt,
                "images": [chart_b64]  # Base64 image
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1000
        }
    }
    
    logger.info("   Sending to Ollama qwen3-vl:8b...")
    
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        try:
            response = await http_client.post(ollama_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            message_content = data.get("message", {}).get("content", "")
            
            logger.info("=" * 60)
            logger.info("🤖 OLLAMA VISION MODEL RESPONSE:")
            logger.info("=" * 60)
            print(message_content)
            logger.info("=" * 60)
            
            # Try to parse JSON
            import json
            try:
                if "```json" in message_content:
                    json_str = message_content.split("```json")[1].split("```")[0].strip()
                elif "```" in message_content:
                    json_str = message_content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = message_content
                
                parsed = json.loads(json_str)
                logger.info("✅ Successfully parsed JSON response")
                logger.info(f"   Action: {parsed.get('action')}")
                logger.info(f"   Confidence: {parsed.get('confidence')}")
                logger.info(f"   Analysis: {parsed.get('visual_analysis', '')[:100]}...")
            except Exception as parse_err:
                logger.warning(f"⚠️ Could not parse as JSON: {parse_err}")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")

if __name__ == "__main__":
    asyncio.run(test_visual_agent())
