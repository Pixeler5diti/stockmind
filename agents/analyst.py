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
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


class AgentState(TypedDict):
    ticker:         str
    forecast_table: str
    volatility:     str
    news_headlines: str
    tech_analysis:  str
    news_summary:   str
    final_report:   str
    recommendation: str


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
    except Exception:
        return "News unavailable."


def technical_analyst(state: AgentState) -> AgentState:
    print("  [Agent 1] Technical Analyst...")
    prompt = f"""You are a quantitative analyst. Use ONLY the numbers below. No macroeconomic speculation.

TICKER: {state['ticker']}
SIGNAL DATA:
{state['forecast_table']}
30-DAY DAILY VOLATILITY: {state['volatility']}%

Answer these exactly:
- Regime signal: LONG / SHORT / FLAT
- Position size recommended and why
- Is current volatility high or low relative to normal?
- Key risk: what would invalidate this signal?
- Assessment: BULLISH / BEARISH / NEUTRAL

Max 80 words. Numbers only, no invented context."""
    response = llm.invoke(prompt)
    return {**state, "tech_analysis": response.content}


def news_summarizer(state: AgentState) -> AgentState:
    print("  [Agent 2] News Summarizer...")
    prompt = f"""Summarize these headlines for {state['ticker']} in 2 sentences.
State ONLY what the headlines say. Do not add interpretation.

HEADLINES:
{state['news_headlines']}

Max 60 words."""
    response = llm.invoke(prompt)
    return {**state, "news_summary": response.content}


def report_generator(state: AgentState) -> AgentState:
    print("  [Agent 3] Report Generator...")
    prompt = f"""Write a short data-driven markdown report for {state['ticker']}.
Only use the data provided. No invented analysis.

TECHNICAL ANALYSIS:
{state['tech_analysis']}

NEWS SUMMARY (factual only):
{state['news_summary']}

SIGNAL DATA:
{state['forecast_table']}

Structure EXACTLY like this:

# {state['ticker']} — StckMind Report

## Technical Outlook
[2-3 sentences from technical analysis only]

## Recent Headlines
[1-2 sentences from news summary only]

## Model Output
[Restate the signal data as-is]

## Stance
BULLISH / BEARISH / NEUTRAL — one word only on this line, nothing else after it

## Disclaimer
Statistical model output only. Not financial advice."""

    response = llm.invoke(prompt)
    text     = response.content

    stance = "NEUTRAL"
    if "## Stance" in text:
        stance_section = text.split("## Stance")[-1].split("##")[0].strip().upper()
        if "BULLISH" in stance_section:
            stance = "BULLISH"
        elif "BEARISH" in stance_section:
            stance = "BEARISH"

    return {**state, "final_report": text, "recommendation": stance}


def critic(state: AgentState) -> AgentState:
    print("  [Agent 4] Critic...")
    prompt = f"""Review this report for {state['ticker']}.

{state['final_report']}

Remove any:
- invented macroeconomic claims
- confidence labels (High/Medium/Low)
- duplicate sections
- speculation not backed by the data provided

Return ONLY the cleaned report. Same structure. No new content."""
    response = llm.invoke(prompt)
    return {**state, "final_report": response.content}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("technical_analyst", technical_analyst)
    g.add_node("news_summarizer",   news_summarizer)
    g.add_node("report_generator",  report_generator)
    g.add_node("critic",            critic)
    g.set_entry_point("technical_analyst")
    g.add_edge("technical_analyst", "news_summarizer")
    g.add_edge("news_summarizer",   "report_generator")
    g.add_edge("report_generator",  "critic")
    g.add_edge("critic",            END)
    return g.compile()


def run_analysis(ticker: str, predictions: dict) -> dict:
    print(f"\n[+] Running agent analysis for {ticker}...")

    forecast    = predictions.get("forecast", [])
    last_close  = predictions.get("last_known_close", 0)
    volatility  = predictions.get("daily_volatility", 0)
    stance      = predictions.get("stance", "UNKNOWN")
    regime_prob = predictions.get("regime_prob", 0)
    position    = predictions.get("position_size", 0)

    forecast_str  = f"Last close: ${last_close}\n"
    forecast_str += f"Regime signal: {stance} (prob={regime_prob:.4f}, position={position:.1%})\n"
    forecast_str += f"Daily volatility: {volatility}%\n\n"
    forecast_str += "5-Day 95% price range:\n"
    for row in forecast:
        forecast_str += f"  {row['date']}: ${row['low_95']} - ${row['high_95']}\n"

    initial = AgentState(
        ticker         = ticker,
        forecast_table = forecast_str,
        volatility     = str(volatility),
        news_headlines = get_news_headlines(ticker),
        tech_analysis  = "",
        news_summary   = "",
        final_report   = "",
        recommendation = "",
    )

    app   = build_graph()
    final = app.invoke(initial)

    print(f"[✓] Done — {final['recommendation']}\n")

    return {
        "ticker":         ticker,
        "recommendation": final["recommendation"],
        "confidence":     "Data-driven",
        "final_report":   final["final_report"],
        "predictions":    predictions,
    }


if __name__ == "__main__":
    from models.predict import predict
    result = run_analysis("AAPL", predict("AAPL"))
    print(result["final_report"])