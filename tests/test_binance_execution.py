
import asyncio
import pytest
import os
import logging
from dotenv import load_dotenv
from src.trading.binance_client import BinanceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("BinanceTest")

load_dotenv()

@pytest.mark.asyncio
async def test_execution_flow():
    # 1. Initialize Client (Testnet)
    print("\n--- 1. Connecting to Binance Testnet ---")
    api_key = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET") or os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Missing API Keys. Please set BINANCE_API_KEY/SECRET in .env")
        return

    client = BinanceClient(api_key=api_key, api_secret=api_secret, testnet=True)
    if not await client.connect():
        logger.error("❌ Failed to connect")
        return

    try:
        # 2. Check Balance
        print("\n--- 2. Checking Balance ---")
        balance = await client.get_balance("USDT")
        logger.info(f"💰 USDT Balance: ${balance:,.2f}")
        
        if balance < 10:
            logger.warning("⚠️ Low balance for testing")

        # 3. Get Price
        print("\n--- 3. Getting BTC Price ---")
        price = await client.get_price("BTCUSDT")
        logger.info(f"BTC Price: ${price:,.2f}")

        # 4. Place limit order (far from price to avoid fill) or Market
        # We will try a small MARKET BUY with SL/TP
        quantity = 0.005 # Min size for BTC is usually 0.001, safe margin
        
        # Calculate SL/TP
        entry_price = price
        stop_loss = entry_price * 0.98 # 2% SL
        take_profit = entry_price * 1.02 # 2% TP
        
        print(f"\n--- 4. Placing MARKET BUY Order (Qty: {quantity}) ---")
        print(f"   Target SL: {stop_loss:.2f}")
        print(f"   Target TP: {take_profit:.2f}")
        
        order = await client.place_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=quantity,
            order_type="MARKET",
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order:
            logger.info(f"✅ Main Order Placed: {order.get('orderId')} status: {order.get('status')}")
            
            # 5. Check Open Orders (SL/TP)
            await asyncio.sleep(2) # Wait for propagation
            open_orders = await client._request("GET", "/fapi/v1/openOrders", {"symbol": "BTCUSDT"}, signed=True)
            print(f"\n--- 5. Verifying SL/TP Orders ({len(open_orders)} open) ---")
            for o in open_orders:
                logger.info(f"   [Order {o['orderId']}] {o['type']} {o['side']} @ {o.get('stopPrice', o.get('price'))}")
            
            if len(open_orders) >= 2:
                logger.info("✅ SL and TP orders detected")
            else:
                logger.warning("⚠️ SL/TP orders might be missing")

            # 6. Cleanup (Cancel All)
            print("\n--- 6. Cleaning Up (Cancel All) ---")
            cancelled = await client.cancel_all_orders("BTCUSDT")
            logger.info("✅ All orders cancelled")
            
            # Close position if filled
            positions = await client.get_positions("BTCUSDT")
            for p in positions:
                amt = float(p['positionAmt'])
                if amt != 0:
                    logger.info(f"Closing remaining position: {amt}")
                    await client.place_order(
                        symbol="BTCUSDT",
                        side="SELL" if amt > 0 else "BUY",
                        quantity=abs(amt),
                        order_type="MARKET"
                    )

    except Exception as e:
        logger.error(f"❌ Test Failed: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_execution_flow())
