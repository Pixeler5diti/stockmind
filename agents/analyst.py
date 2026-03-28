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


# ── Agent 1: Technical Analyst ─────────────────────────
def technical_analyst(state: AgentState) -> AgentState:
    print("  [Agent 1] Technical Analyst...")

    prompt = f"""You are a quantitative analyst. Use ONLY the numbers below. No macroeconomic speculation.

TICKER: {state['ticker']}
FORECAST:
{state['forecast_table']}
30-DAY DAILY VOLATILITY: {state['volatility']}%

Answer these exactly:
- Overall direction: UP / DOWN / FLAT
- Average predicted daily move: X%
- Is this within normal volatility? YES / NO
- Biggest single-day move in forecast: X%
- Assessment: BULLISH / BEARISH / NEUTRAL

Max 80 words. Numbers only, no invented context."""

    response = llm.invoke(prompt)
    return {**state, "tech_analysis": response.content}


# ── Agent 2: News Summarizer ───────────────────────────
def news_summarizer(state: AgentState) -> AgentState:
    print("  [Agent 2] News Summarizer...")

    prompt = f"""Summarize these headlines for {state['ticker']} in 2 sentences.
State ONLY what the headlines say. Do not add interpretation.

HEADLINES:
{state['news_headlines']}

Max 60 words."""

    response = llm.invoke(prompt)
    return {**state, "news_summary": response.content}


# ── Agent 3: Report Generator ──────────────────────────
def report_generator(state: AgentState) -> AgentState:
    print("  [Agent 3] Report Generator...")

    prompt = f"""Write a short data-driven markdown report for {state['ticker']}.
Only use the data provided. No invented analysis.

TECHNICAL ANALYSIS:
{state['tech_analysis']}

NEWS SUMMARY (factual only):
{state['news_summary']}

FORECAST TABLE:
{state['forecast_table']}

Structure EXACTLY like this:

# {state['ticker']} — StckMind Report

## Technical Outlook
[2-3 sentences from technical analysis only]

## Recent Headlines
[1-2 sentences from news summary only]

## Model Output
[Restate the forecast table as-is]

## Stance
BULLISH / BEARISH / NEUTRAL — one word only on this line, nothing else after it

## Disclaimer
Statistical model output only. Not financial advice."""

    response = llm.invoke(prompt)
    text     = response.content

    # Extract stance from the Stance section only
    stance = "NEUTRAL"
    if "## Stance" in text:
        stance_section = text.split("## Stance")[-1].split("##")[0].strip().upper()
        if "BULLISH" in stance_section:
            stance = "BULLISH"
        elif "BEARISH" in stance_section:
            stance = "BEARISH"

    return {**state, "final_report": text, "recommendation": stance}


# ── Agent 4: Critic ────────────────────────────────────
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


# ── Graph ──────────────────────────────────────────────
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


# ── Run ────────────────────────────────────────────────
def run_analysis(ticker: str, predictions: dict) -> dict:
    print(f"\n[+] Running agent analysis for {ticker}...")

    forecast   = predictions.get("forecast", [])
    last_close = predictions.get("last_known_close", 0)
    volatility = predictions.get("daily_volatility", 0)

    # Build clean forecast table
    table = f"Last close: ${last_close}\n"
    table += f"{'Date':<12} {'Daily Δ%':>9}  {'Range':>26}\n"
    table += "-" * 52 + "\n"
    for row in forecast:
        sign = "+" if row["pct_change"] >= 0 else ""
        table += (
            f"{row['date']:<12} {sign}{row['pct_change']:>8.3f}%  "
            f"[${row['low_estimate']} – ${row['high_estimate']}]\n"
        )

    initial = AgentState(
        ticker         = ticker,
        forecast_table = table,
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