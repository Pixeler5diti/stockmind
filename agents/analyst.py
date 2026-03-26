import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict
import yfinance as yf

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)



class AgentState(TypedDict):
    ticker: str
    predictions: str
    news_sentiment: str
    performance_analysis: str
    final_report: str
    recommendation: str
    confidence: str



def get_news(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        if not news:
            return "No recent news found."
        items = []
        for n in news:
            title = n.get("content", {}).get("title", "No title")
            items.append(f"- {title}")
        return "\n".join(items)
    except Exception as e:
        return f"Could not fetch news: {e}"


# 1st agent: performance analyst
def performance_analyst(state: AgentState) -> AgentState:
    print("  [Agent 1] Performance Analyst running...")

    prompt = f"""You are a quantitative analyst. Analyze these stock predictions for {state['ticker']}:

{state['predictions']}

Provide:
1. Trend direction (bullish/bearish/sideways)
2. Price movement percentage
3. Key observations
4. Risk level (low/medium/high)

Be concise and data-driven. Max 150 words."""

    response = llm.invoke(prompt)
    return {**state, "performance_analysis": response.content}


# 2nd agent :market expert
def market_expert(state: AgentState) -> AgentState:
    print("  [Agent 2] Market Expert running...")

    news = get_news(state["ticker"])

    prompt = f"""You are a market strategist analyzing {state['ticker']}.

Recent news:
{news}

Provide a 3-4 line sentiment summary covering:
- Overall market sentiment (bullish/bearish/neutral)
- Key catalysts or risks from the news
- How this might affect the stock

Max 100 words."""

    response = llm.invoke(prompt)
    return {**state, "news_sentiment": response.content}


# 3rd agent : report generator
def report_generator(state: AgentState) -> AgentState:
    print("  [Agent 3] Report Generator running...")

    prompt = f"""Write a professional Bloomberg-style markdown report for {state['ticker']}.

TECHNICAL ANALYSIS:
{state['performance_analysis']}

MARKET SENTIMENT:
{state['news_sentiment']}

Use this exact structure:

# {state['ticker']} — StckMind Analysis Report

## Executive Summary
[2-3 sentences]

## Technical Outlook
[Price predictions and trend analysis]

## Market Sentiment
[News and sentiment analysis]

## Recommendation
**Market Stance:** BULLISH / BEARISH / NEUTRAL
**Confidence:** High / Medium / Low
**Rationale:** [1-2 sentences]
"""

    response = llm.invoke(prompt)
    text = response.content

    # recommendation and confidence
    upper = text.upper()
    if "BULLISH" in upper:
        stance = "BULLISH"
    elif "BEARISH" in upper:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    if "HIGH" in upper:
        confidence = "High"
    elif "LOW" in upper:
        confidence = "Low"
    else:
        confidence = "Medium"

    return {**state, "final_report": text, "recommendation": stance, "confidence": confidence}


# 4th agent : critc
def critic(state: AgentState) -> AgentState:
    print("  [Agent 4] Critic running...")

    prompt = f"""You are a Senior Financial Editor reviewing this report for {state['ticker']}.

DRAFT REPORT:
{state['final_report']}

Tasks:
1. Verify the recommendation aligns with the technical and sentiment data
2. Fix any inconsistencies
3. Ensure professional tone
4. Return the final polished report only — no commentary, no preamble."""

    response = llm.invoke(prompt)
    return {**state, "final_report": response.content}


# build graph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("performance_analyst", performance_analyst)
    graph.add_node("market_expert", market_expert)
    graph.add_node("report_generator", report_generator)
    graph.add_node("critic", critic)

    graph.set_entry_point("performance_analyst")
    graph.add_edge("performance_analyst", "market_expert")
    graph.add_edge("market_expert", "report_generator")
    graph.add_edge("report_generator", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


# run analysis
def run_analysis(ticker: str, predictions: dict) -> dict:
    print(f"\n[+] Running agent analysis for {ticker}...")


    forecast_str = f"Last known close: ${predictions.get('last_known_close')}\n"
    forecast_str += "5-day forecast:\n"
    for row in predictions.get("forecast", []):
        forecast_str += f"  {row['date']} → ${row['predicted_close']}\n"

    initial_state = AgentState(
        ticker=ticker,
        predictions=forecast_str,
        news_sentiment="",
        performance_analysis="",
        final_report="",
        recommendation="",
        confidence=""
    )

    app = build_graph()
    final_state = app.invoke(initial_state)

    print(f"[✓] Analysis complete — {final_state['recommendation']} ({final_state['confidence']} confidence)\n")

    return {
        "ticker": ticker,
        "recommendation": final_state["recommendation"],
        "confidence": final_state["confidence"],
        "final_report": final_state["final_report"],
        "predictions": predictions
    }


if __name__ == "__main__":
    from models.predict import predict

    ticker = "AAPL"
    predictions = predict(ticker)
    result = run_analysis(ticker, predictions)

    print(result["final_report"])