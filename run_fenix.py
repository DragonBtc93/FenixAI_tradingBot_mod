#!/usr/bin/env python3
# run_fenix.py
"""
Script principal de ejecución de Fenix Trading Bot.

Uso:
    python run_fenix.py                    # Paper trading con Ollama
    python run_fenix.py --mode live        # Trading real
    python run_fenix.py --symbol ETHUSDT   # Otro par
    python run_fenix.py --help             # Ver opciones
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Crear directorio de logs si no existe
Path("logs").mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/fenix_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger("Fenix")


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Fenix AI Trading Bot - LangGraph Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_fenix.py                       # Paper trading, BTCUSDT, 15m
  python run_fenix.py --mode live           # Trading real
  python run_fenix.py --symbol ETHUSDT      # Otro par
  python run_fenix.py --timeframe 5m        # Otro timeframe
  python run_fenix.py --no-visual           # Sin agente visual
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Modo de trading (default: paper)",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Requerido para ejecutar en modo live y prevenir operaciones accidentales",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Par a tradear (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Timeframe de análisis (default: 15m)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Modelo Ollama a usar (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Intervalo entre análisis en segundos (default: 60)",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Desactivar agente visual",
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Desactivar agente de sentimiento",
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=2.0,
        help="Máximo riesgo por trade en %% (default: 2.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo simular, no ejecutar órdenes",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Iniciar servidor API (FastAPI + Socket.IO) para el frontend",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host para bind del servidor API (default: 127.0.0.1, no expuesto públicamente)",
    )
    
    return parser.parse_args()


async def main():
    """Función principal."""
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🦅  FENIX AI TRADING BOT                                   ║
    ║   LangGraph Multi-Agent Architecture                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("Iniciando Fenix Trading Bot")
    logger.info(f"  Modo: {args.mode.upper()}")
    logger.info(f"  Símbolo: {args.symbol}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Modelo: {args.model}")
    logger.info(f"  Intervalo: {args.interval}s")
    logger.info(f"  Visual: {'Sí' if not args.no_visual else 'No'}")
    logger.info(f"  Sentiment: {'Sí' if not args.no_sentiment else 'No'}")

    if args.mode == "live" and not args.allow_live:
        logger.error("Modo live solicitado pero --allow-live no fue proporcionado. Abortando por seguridad.")
        return 1
    
    # Verificar Ollama
    logger.info("Verificando conexión a Ollama...")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama no está disponible. Ejecuta: ollama serve")
            return 1
        
        models = [m["name"] for m in response.json().get("models", [])]
        if args.model not in models and not any(args.model.split(":")[0] in m for m in models):
            logger.warning(f"Modelo {args.model} no encontrado. Disponibles: {models[:5]}")
            args.model = models[0] if models else "gemma3:1b"
            logger.info(f"Usando modelo alternativo: {args.model}")
        
        logger.info(f"✅ Ollama OK - Modelo: {args.model}")
        
    except Exception as e:
        logger.error(f"Error conectando a Ollama: {e}")
        return 1
    
    # Verificar Binance
    logger.info("Verificando conexión a Binance...")
    try:
        from src.trading.binance_client import BinanceClient
        
        testnet = args.mode == "paper"
        client = BinanceClient(testnet=testnet)
        connected = await client.connect()
        
        if connected:
            price = await client.get_price(args.symbol)
            if price:
                logger.info(f"✅ Binance OK - {args.symbol}: ${price:,.2f}")
            else:
                logger.warning(f"No se pudo obtener precio de {args.symbol}")
        else:
            logger.warning("No se pudo conectar a Binance, continuando en modo simulado")
        
        await client.close()
        
    except ImportError:
        logger.warning("Cliente Binance no disponible, continuando en modo simulado")
    except Exception as e:
        logger.warning(f"Error conectando a Binance: {e}")
    
    # Iniciar servidor API si se solicita
    if args.api:
        logger.info("🚀 Iniciando servidor API (Frontend Backend)...")
        import uvicorn
        # Importar app_socketio desde el nuevo módulo server
        # Nota: uvicorn necesita el import string "src.api.server:app_socketio"
        uvicorn.run("src.api.server:app_socketio", host=args.host, port=8000, reload=False)
        return 0

    # Iniciar motor de trading estándar (CLI mode)
    logger.info("Iniciando motor de trading (CLI Mode)...")
    
    try:
        from src.trading.engine import TradingEngine
        
        engine = TradingEngine(
            symbol=args.symbol,
            timeframe=args.timeframe,
            use_testnet=args.mode == "paper",
            paper_trading=args.mode == "paper" or args.dry_run,
            enable_visual_agent=not args.no_visual,
            enable_sentiment_agent=not args.no_sentiment,
            allow_live_trading=args.allow_live,
        )
        
        # Manejo de señales de sistema
        stop_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            logger.info("Señal de interrupción recibida, deteniendo...")
            stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Iniciar
        logger.info("✅ Motor de trading listo")
        
        # Ejecutar
        await engine.start()
        
        return 0
        
    except ImportError as e:
        logger.error(f"Error importando motor de trading: {e}")
        logger.info("Ejecutando en modo de prueba simplificado...")
        
        # Modo de prueba simplificado
        return await run_simple_test(args)


async def run_simple_test(args):
    """Ejecuta un test simplificado sin el motor completo."""
    logger.info("=== Modo de Prueba Simplificado ===")
    
    from src.prompts.agent_prompts import format_prompt
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.trading.binance_client import BinanceClient
    
    # Conectar a Binance
    client = BinanceClient(testnet=True)
    await client.connect()
    
    # Obtener datos reales
    price = await client.get_price(args.symbol)
    klines = await client.get_klines(args.symbol, args.timeframe, limit=50)
    
    logger.info(f"Datos recibidos: {args.symbol} @ ${price:,.2f}")
    logger.info(f"Klines: {len(klines)} velas")
    
    # Calcular indicadores simples
    if klines:
        closes = [k["close"] for k in klines]
        
        # RSI simple
        gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
        losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.0001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # EMA simple
        ema_9 = sum(closes[-9:]) / 9 if len(closes) >= 9 else closes[-1]
        ema_21 = sum(closes[-21:]) / 21 if len(closes) >= 21 else closes[-1]
        
        indicators = {
            "rsi": round(rsi, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "price": price,
        }
        
        logger.info(f"Indicadores: RSI={rsi:.1f}, EMA9={ema_9:.0f}, EMA21={ema_21:.0f}")
    else:
        indicators = {"rsi": 50, "price": price}
    
    # Ejecutar análisis con LLM
    logger.info("Ejecutando análisis con LLM...")
    
    messages = format_prompt(
        "technical_analyst",
        symbol=args.symbol,
        timeframe=args.timeframe,
        indicators_json=str(indicators),
        current_price=str(price),
    )
    
    llm = ChatOllama(
        model=args.model,
        temperature=0.1,
        num_predict=500,
    )
    
    response = llm.invoke([
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ])
    
    logger.info("=== Respuesta del Agente Técnico ===")
    print(response.content[:1000])
    
    await client.close()
    return 0


if __name__ == "__main__":
    # Parse args first to handle --api mode which uses uvicorn (blocking, owns loop)
    args = parse_args()

    if args.api:
        print("🚀 Iniciando servidor API (Frontend Backend)...")
        import uvicorn
        host = args.host or "127.0.0.1"
        if host == "0.0.0.0":
            allow_expose = os.getenv("ALLOW_EXPOSE_API", "false").lower() == "true"
            if not allow_expose:
                logger.warning("API host set to 0.0.0.0; to expose the API explicitly set ALLOW_EXPOSE_API=true")
                logger.info("Binding to 127.0.0.1 instead for safety")
                host = "127.0.0.1"
        uvicorn.run("src.api.server:app_socketio", host=host, port=8000, reload=False)
        sys.exit(0)

    try:
        # Pass args to main (we need to modify main signature or use global/re-parse)
        # Easier: Re-parse inside main or refactor main to accept args. 
        # Since main calls parse_args again, it's fine (argparse is idempotent usually if args not passed explicitly)
        # But clearer to pass args.
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario")
        sys.exit(0)
