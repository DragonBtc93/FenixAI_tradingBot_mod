
import os
import base64
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisualTest")

def test_visual_interaction():
    # Load image
    img_path = "cache/charts/BTCUSDT_15m_20251205_183709.png"
    if not os.path.exists(img_path):
        # Fallback to any png
        files = [f for f in os.listdir("cache/charts") if f.endswith(".png")]
        if files:
            img_path = os.path.join("cache/charts", files[-1])
        else:
            logger.error("No charts found")
            return

    logger.info(f"Using image: {img_path}")
    
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    llm = ChatOllama(
        model="qwen3-vl:4b", # Updated to correct tag
        temperature=0.1,
        # base_url="http://localhost:11434" # Default
    )
    
    logger.info(f"Invoking {llm.model}...")
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a financial analyst. Describe the chart."),
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this chart."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
            ])
        ])
        logger.info(f"Response: {response.content}")
    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    test_visual_interaction()
