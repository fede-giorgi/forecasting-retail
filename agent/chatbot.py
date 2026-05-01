"""
agent/chatbot.py
----------------
Terminal chatbot for unified retail demand forecasting (LR, Prophet).
Handles Cluster mapping and model/mode selection via LLM tools.
"""
from __future__ import annotations
import os, warnings
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
def run_forecast(stock_code: str, model: str = "lr", horizon_weeks: int = 4) -> str:
    """Forecast weekly retail demand for a specific product using a specified model.
    Args:
        stock_code: Product identifier (e.g., '85123A', '22423')
        model: Model type ('lr' for Ridge Regression, 'prophet' for Facebook Prophet)
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
    """Retrieve historical summary, seasonal cluster profile, and volume metrics for a specific product.
    Args:
        stock_code: Product identifier (e.g., '85123A', '22423')
    """
    stock_code = str(stock_code).strip().upper()
    
    try:
        df = get_df()
        df_p = df[df["StockCode"] == stock_code]
        
        if df_p.empty:
            return f"[Error] No data found for StockCode {stock_code}."
            
        # Extract metadata
        cluster_id = int(df_p["profile_cluster_id"].iloc[0])
        
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
   - Run the standard Ridge Regression model (model='lr').
   - Run the Prophet model (model='prophet').

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

    # Try to obtain an API key for either OpenAI or Gemini
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        # Fallback to Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[Error] Neither OPENAI_KEY nor GEMINI_API_KEY found in environment.")
            return
        # Initialize Gemini LLM
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, temperature=0)
        except ImportError:
            print("[Error] langchain-google-genai not installed. Install it to use Gemini.")
            return
    else:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

    print("Loading retail dataset...")
    # Using relative path from agent/ directory
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_retail_data.parquet")
    
    try:
        _df = pd.read_parquet(parquet_path, engine="pyarrow")
        print(f"Ready — {_df['StockCode'].nunique()} products mapped to clusters.\n")
    except Exception as e:
        print(f"[Critical Error] Could not load dataset at {parquet_path}: {e}")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
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
            reply    = result["messages"][-1].content
            print(f"\nAgent:\n{reply}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=reply))

        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()