import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict
import yfinance as yf
import numpy as np

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)


class AgentState(TypedDict):
    ticker: str
    predictions: str
    volatility: str
    news_headlines: str
    final_report: str
    recommendation: str
    confidence: str


def get_news_headlines(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news[:5]
        if not news:
            return "No recent news available."
        return "\n".join(
            f"- {n.get('content', {}).get('title', 'No title')}"
            for n in news
        )
    except Exception as e:
        return f"Could not fetch news: {e}"


# ── Agent 1: Technical Analyst ─────────────────────────
# FIX 6: Only uses model output data — no macro speculation
def technical_analyst(state: AgentState) -> AgentState:
    print("  [Agent 1] Technical Analyst running...")

    prompt = f"""You are a quantitative analyst. Analyze ONLY the data provided below for {state['ticker']}.
Do NOT speculate about macroeconomics, geopolitics, or anything not in this data.

FORECAST DATA:
{state['predictions']}

RECENT VOLATILITY: {state['volatility']}% daily (30-day)

Based strictly on this data, provide:
1. Trend direction: up / down / sideways
2. Average predicted daily move (%)
3. Is the predicted move within normal volatility range? yes/no
4. Risk level: low / medium / high (based on volatility only)

Max 100 words. Be concise and factual."""

    response = llm.invoke(prompt)
    return {**state, "predictions": state["predictions"], "technical_analysis": response.content}


# ── Agent 2: News Summarizer ───────────────────────────
# FIX 6: Only summarizes actual news headlines, no invented context
def news_summarizer(state: AgentState) -> AgentState:
    print("  [Agent 2] News Summarizer running...")

    prompt = f"""Summarize ONLY these recent news headlines for {state['ticker']}.
Do NOT add any analysis or context beyond what is stated.

HEADLINES:
{state['news_headlines']}

Output: 2-3 sentence factual summary of the headlines only. Max 80 words."""

    response = llm.invoke(prompt)
    return {**state, "news_summary": response.content}


# ── Agent 3: Report Generator ──────────────────────────
def report_generator(state: AgentState) -> AgentState:
    print("  [Agent 3] Report Generator running...")

    prompt = f"""Write a concise, data-driven markdown report for {state['ticker']}.
Use ONLY the data provided. Do NOT invent macroeconomic context.

TECHNICAL DATA:
{state.get('technical_analysis', '')}

NEWS SUMMARY:
{state.get('news_summary', '')}

FORECAST:
{state['predictions']}

Use this exact structure:

# {state['ticker']} — StckMind Report

## Technical Outlook
[Based strictly on forecast data and volatility]

## Recent News
[Based strictly on headlines provided]

## Recommendation
**Market Stance:** BULLISH / BEARISH / NEUTRAL
**Confidence:** High / Medium / Low
**Rationale:** [1 sentence, data-driven only]

## Disclaimer
Model predictions are statistical estimates. Not financial advice."""

    response = llm.invoke(prompt)
    text     = response.content

    rec_section = text.split("## Recommendation")[-1].upper() if "## Recommendation" in text else text.upper()
    stance      = "BULLISH" if "BULLISH" in rec_section else "BEARISH" if "BEARISH" in rec_section else "NEUTRAL"
    confidence  = "High" if "HIGH" in rec_section else "Low" if "LOW" in rec_section else "Medium"

    return {**state, "final_report": text, "recommendation": stance, "confidence": confidence}


# ── Agent 4: Critic ────────────────────────────────────
def critic(state: AgentState) -> AgentState:
    print("  [Agent 4] Critic running...")

    prompt = f"""Review this financial report for {state['ticker']}.

REPORT:
{state['final_report']}

Check:
1. Does recommendation match the technical data?
2. Are there any invented claims not backed by data?
3. Remove any duplicate sections

Return ONLY the corrected report. No commentary. No new sections. Same structure."""

    response = llm.invoke(prompt)
    return {**state, "final_report": response.content}


# ── Build Graph ────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("technical_analyst", technical_analyst)
    graph.add_node("news_summarizer",   news_summarizer)
    graph.add_node("report_generator",  report_generator)
    graph.add_node("critic",            critic)

    graph.set_entry_point("technical_analyst")
    graph.add_edge("technical_analyst", "news_summarizer")
    graph.add_edge("news_summarizer",   "report_generator")
    graph.add_edge("report_generator",  "critic")
    graph.add_edge("critic",            END)

    return graph.compile()


# ── Run Analysis ───────────────────────────────────────
def run_analysis(ticker: str, predictions: dict) -> dict:
    print(f"\n[+] Running agent analysis for {ticker}...")

    forecast     = predictions.get("forecast", [])
    last_close   = predictions.get("last_known_close", 0)
    volatility   = predictions.get("daily_volatility", 0)

    # FIX 7: Format as range output, not fake precise prices
    forecast_str = f"Last close: ${last_close}\n"
    forecast_str += f"{'Date':<12} {'Change':>8}  {'Range':>22}\n"
    forecast_str += "-" * 46 + "\n"
    for row in forecast:
        sign = "+" if row["pct_change"] >= 0 else ""
        forecast_str += (
            f"{row['date']:<12} {sign}{row['pct_change']:>7.3f}%  "
            f"[${row['low_estimate']} – ${row['high_estimate']}]\n"
        )

    initial_state = AgentState(
        ticker          = ticker,
        predictions     = forecast_str,
        volatility      = str(volatility),
        news_headlines  = get_news_headlines(ticker),
        final_report    = "",
        recommendation  = "",
        confidence      = ""
    )

    app          = build_graph()
    final_state  = app.invoke(initial_state)

    print(f"[✓] Done — {final_state['recommendation']} ({final_state['confidence']} confidence)\n")

    return {
        "ticker":         ticker,
        "recommendation": final_state["recommendation"],
        "confidence":     final_state["confidence"],
        "final_report":   final_state["final_report"],
        "predictions":    predictions
    }


if __name__ == "__main__":
    from models.predict import predict
    result = run_analysis("AAPL", predict("AAPL"))
    print(result["final_report"])