"""
agent/chatbot.py
----------------
Terminal chatbot for unified retail demand forecasting (LR, Prophet).
Handles Cluster mapping and model/mode selection via LLM tools.
"""
from __future__ import annotations
import os
import sys

# CRITICAL FIX FOR MAC: Prevent LightGBM C++ Segmentation Faults in Langchain
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings
import pandas as pd
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from inference.predict import predict_retail

# DATASET
_df: pd.DataFrame | None = None

def get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded.")
    return _df

# TOOLS 

@tool
def run_forecast(stock_code: str, model: str = "lgb", horizon_weeks: int = 4) -> str:
    """Forecast weekly retail demand for a specific product using a specified model.
    Args:
        stock_code: Product identifier (e.g., '85123A', '22423')
        model: Model type ('lgb' for LightGBM, 'lr' for Ridge Regression, 'prophet' for Facebook Prophet)
        horizon_weeks: Number of weeks into the future to forecast (default 4)
    """
    stock_code = str(stock_code).strip().upper()
    
    try:
        df = get_df()
        result = predict_retail(stock_code, model, df, horizon_weeks)
        return result.to_summary()
    except Exception as e:
        return f"[Error] Forecast failed for {stock_code}: {str(e)}"



@tool
def get_product_info(stock_code: str) -> str:
    """Retrieve historical summary, seasonal cluster profile, semantic category, and volume metrics for a specific product.
    Args:
        stock_code: Product identifier (e.g., '85123A', '22423')
    """
    stock_code = str(stock_code).strip().upper()
    
    try:
        df = get_df()
        df_p = df[df["StockCode"] == stock_code]
        
        if df_p.empty:
            return f"[Error] No data found for StockCode {stock_code}."
            
        # Extract advanced clustering metadata
        cluster_id = int(df_p["profile_cluster_id"].iloc[0])
        volume_tier = str(df_p["volume_tier"].iloc[0])
        semantic_category = str(df_p["semantic_cluster_name"].iloc[0])
        
        # Calculate historical metrics
        mean_qty = round(df_p["Quantity"].mean(), 1)
        max_qty = round(df_p["Quantity"].max(), 0)
        total_qty = round(df_p["Quantity"].sum(), 0)
        
        # Human-readable cluster names based on our report mapping
        cluster_map = {
            0: "Stable Base Demand",
            1: "Winter/Christmas Peak",
            2: "Summer/Spring Peak",
            3: "Intermittent / Spiky Demand",
            4: "Declining Lifecycle"
        }
        behavior = cluster_map.get(cluster_id, f"Cluster {cluster_id}")
        
        return (
            f"--- RETAIL PRODUCT PROFILE: {stock_code} ---\n"
            f"Semantic Category: {semantic_category}\n"
            f"Volume Tier      : {volume_tier}\n"
            f"Seasonal Profile : {behavior} (Cluster {cluster_id})\n"
            f"Historical Avg   : {mean_qty} units/week\n"
            f"Historical Peak  : {max_qty} units/week\n"
            f"Total Sales Vol  : {total_qty:,.0f} units (Lifetime)\n"
        )
    except Exception as e:
        return f"[Error] Could not retrieve info for {stock_code}: {str(e)}"


# SYSTEM PROMPT 
SYSTEM = """You are an Expert Retail Supply Chain AI. 

Your objective is to provide actionable demand forecasts and historical profiling for a portfolio of retail products (StockCodes) based on weekly sales data.

TOOL USAGE STRATEGY:
1. If a user asks for general information, historical sales, or the profile of a product, ALWAYS invoke the `get_product_info` tool first.
2. If a user requests a forecast, ALWAYS invoke the `run_forecast` tool. Unless specified otherwise, run the tool TWICE to provide a comparative benchmark:
- Run the LightGBM model (model='lgb') as the primary state-of-the-art benchmark.
- Run the Ridge Regression (model='lr') or Prophet (model='prophet') for comparison.

PARAMETERS:
- `stock_code`: The unique identifier of the product (e.g., '85123A').
- `horizon_weeks`: Defaults to 4 weeks (1 month) for short/mid-term planning.

RESPONSE FORMAT:
- Use clean Markdown lists/tables.
- Be analytical. If retrieving product info, explain what their "Seasonal Profile" means practically for inventory management.
- If providing a forecast, compare the models. If Prophet captures a spike that LR misses, mention it might be due to holidays/seasonality.
"""

# Always load environment variables from .env
load_dotenv()

# ── MAIN ─

def main():
    global _df

    # 1. Dynamically select the LLM provider based on environment variables
    # Defaults to 'openai' if not specified in the .env file
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    print(f"Initializing LLM Agent using provider: {provider.upper()}")

    try:
        # Initialize the chosen LLM with temperature 0 for deterministic tool calling
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            # Ollama runs locally, so we default to a standard fast model like llama3
            llm = ChatOllama(model="llama3", temperature=0)
            
        else:
            print(f"[Error] Unsupported LLM_PROVIDER: '{provider}'. Please use openai, gemini, claude, or ollama.")
            return
            
    except ImportError as e:
        print(f"[Error] Missing required library for {provider.upper()}: {e}")
        print(f"Please run: pip install langchain-{provider if provider != 'ollama' else 'community'}")
        return
    except Exception as e:
        print(f"[Critical Error] Failed to initialize {provider.upper()}: {e}")
        print("Please ensure your API keys are correctly set in the .env file.")
        return

    print("Loading retail dataset...")
    # Using relative path from agent/ directory
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_retail_data.parquet")
    
    try:
        _df = pd.read_parquet(parquet_path, engine="pyarrow")
        print(f"Ready — {_df['StockCode'].nunique()} products mapped to clusters.\n")
    except Exception as e:
        print(f"[Critical Error] Could not load dataset at {parquet_path}: {e}")
        return

    # Create the ReAct agent using the dynamically loaded LLM
    tools = [run_forecast, get_product_info]
    agent = create_react_agent(llm, tools)

    chat_history = []
    print("Retail Demand Analyst Bot - (type 'exit' to quit)")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        try:
            messages = [SystemMessage(content=SYSTEM)] + chat_history + [HumanMessage(content=user_input)]
            result   = agent.invoke({"messages": messages})
            
            # Safely extract the text content, handling cases where it might be a list or dict
            raw_content = result["messages"][-1].content
            if isinstance(raw_content, list):
                # If it's a list (like the Gemini response you saw), extract the 'text' key from the first item
                reply = raw_content[0].get('text', str(raw_content))
            else:
                reply = raw_content
                
            print(f"\nAgent:\n{reply}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=reply))

        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()