"""EuroPowerRAG — Streamlit Q&A interface.

Run:  streamlit run app.py
"""

from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="EuroPowerRAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ EuroPowerRAG")
    st.caption("European Power Market Intelligence")
    st.divider()

    st.subheader("Filters")

    country_options = {
        "All countries": None,
        "Germany (DE_LU)": ["DE_LU"],
        "France (FR)": ["FR"],
        "Netherlands (NL)": ["NL"],
        "Great Britain (GB)": ["GB"],
        "Spain (ES)": ["ES"],
        "Belgium (BE)": ["BE"],
    }
    selected_country_label = st.selectbox("Country", list(country_options.keys()))
    country_filter = country_options[selected_country_label]

    doc_type_options = {
        "All types": None,
        "Price data": ["price_data"],
        "Generation data": ["generation_data"],
        "News": ["news"],
        "Reports": ["report"],
        "Prices + News": ["price_data", "news"],
    }
    selected_type_label = st.selectbox("Document type", list(doc_type_options.keys()))
    doc_type_filter = doc_type_options[selected_type_label]

    use_date_filter = st.checkbox("Filter by date range")
    date_from = date_to = None
    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", value=datetime.now() - timedelta(days=30))
        with col2:
            date_to = st.date_input("To", value=datetime.now())
        date_from = date_from.strftime("%Y-%m-%d") if date_from else None
        date_to = date_to.strftime("%Y-%m-%d") if date_to else None

    n_results = st.slider("Documents to retrieve", min_value=3, max_value=10, value=5)

    st.divider()
    if st.button("Run ingestion now", use_container_width=True):
        with st.spinner("Ingesting data..."):
            try:
                from ingest import main
                rc = main()
                if rc == 0:
                    st.success("Ingestion complete")
                else:
                    st.error("Ingestion finished with errors — check terminal")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption("Data: ENTSO-E · RSS feeds · PDF reports")
    st.caption("Model: claude-sonnet-4-6")
    
    st.divider()
    page = st.radio("Navigation", ["Q&A", "Dashboard", "Sentiment & Prices", "📊 Project Progress"])

# ── Main ──────────────────────────────────────────────────────────────────────

import pandas as pd
import altair as alt
from pathlib import Path
import sqlite3

if page == "Dashboard":
    st.title("Backtest & Paper Trading Dashboard")
    
    st.header("Backtest Results")
    backtest_dir = Path("data/backtest")
    if backtest_dir.exists():
        csv_files = list(backtest_dir.glob("trades_*.csv"))
        if csv_files:
            selected_csv = st.selectbox("Select Backtest Run", [f.name for f in csv_files])
            df_bt = pd.read_csv(backtest_dir / selected_csv)
            df_bt["date"] = pd.to_datetime(df_bt["date"])
            df_bt = df_bt.sort_values("date")
            
            # Metrics
            st.subheader(f"Run: {selected_csv}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(df_bt))
            col2.metric("Total PnL", f"€{df_bt['pnl'].sum():,.2f}")
            win_rate = (df_bt["pnl"] > 0).mean()
            col3.metric("Win Rate", f"{win_rate:.1%}")
            
            # Equity curve
            df_bt["equity"] = (1.0 + df_bt["return"]).cumprod()
            
            st.write("**Equity Curve**")
            chart_eq = alt.Chart(df_bt).mark_line().encode(
                x='date:T',
                y=alt.Y('equity:Q', scale=alt.Scale(zero=False)),
                tooltip=['date', 'equity', 'country', 'signal']
            ).interactive()
            st.altair_chart(chart_eq, use_container_width=True)
            
            # Drawdown
            st.write("**Drawdown**")
            df_bt["running_max"] = df_bt["equity"].cummax()
            df_bt["drawdown"] = (df_bt["running_max"] - df_bt["equity"]) / df_bt["running_max"]
            chart_dd = alt.Chart(df_bt).mark_area(color='red', opacity=0.3).encode(
                x='date:T',
                y='drawdown:Q',
                tooltip=['date', 'drawdown']
            ).interactive()
            st.altair_chart(chart_dd, use_container_width=True)
            
            st.write("**Trade Log**")
            st.dataframe(df_bt, use_container_width=True)
        else:
            st.info("No backtest runs found. Run `python -m src.backtest.run_6mo`.")
    else:
        st.info("No backtest directory found. Run `python -m src.backtest.run_6mo`.")
        
    st.header("Paper Trading (Recent Signals)")
    try:
        with sqlite3.connect("data/paper_trades.db") as conn:
            df_paper = pd.read_sql("SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT 50", conn)
            if not df_paper.empty:
                st.dataframe(df_paper, use_container_width=True)
            else:
                st.info("No paper trades yet.")
    except Exception as e:
        st.info("No paper trades database found yet. The scheduler will create it.")
    st.stop()

if page == "Sentiment & Prices":
    st.title("Sentiment & Prices Overlay")
    st.write("Compare daily day-ahead prices with news sentiment.")
    
    price_path = Path("data/processed/prices_history.parquet")
    sent_path = Path("data/processed/sentiment_daily.parquet")
    
    if not price_path.exists():
        st.error("Price history not found. Run ingestion.")
        st.stop()
        
    df_p = pd.read_parquet(price_path)
    df_p["date"] = pd.to_datetime(df_p["date"])
    
    if sent_path.exists():
        df_s = pd.read_parquet(sent_path)
        df_s["date"] = pd.to_datetime(df_s["date"])
        df_merged = pd.merge(df_p, df_s, on=["date", "country"], how="left")
        df_merged["mean_weighted_score"] = df_merged["mean_weighted_score"].fillna(0)
    else:
        df_merged = df_p.copy()
        df_merged["mean_weighted_score"] = 0.0
        st.warning("Sentiment data not found. Plotting zero sentiment.")
        
    country_to_plot = st.selectbox("Select Country", df_merged["country"].unique())
    df_plot = df_merged[df_merged["country"] == country_to_plot].sort_values("date").tail(90) # Last 90 days
    
    # Dual axis chart in Altair
    base = alt.Chart(df_plot).encode(x='date:T')
    
    line_price = base.mark_line(color='blue').encode(
        y=alt.Y('price_eur_mwh:Q', title='Price (EUR/MWh)', scale=alt.Scale(zero=False)),
        tooltip=['date', 'price_eur_mwh']
    )
    
    bar_sent = base.mark_bar(opacity=0.5).encode(
        y=alt.Y('mean_weighted_score:Q', title='Sentiment Score'),
        color=alt.condition(
            alt.datum.mean_weighted_score > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=['date', 'mean_weighted_score']
    )
    
    dual_chart = alt.layer(bar_sent, line_price).resolve_scale(y='independent').interactive()
    st.altair_chart(dual_chart, use_container_width=True)
    st.stop()



if page == "📊 Project Progress":
    st.title("📊 EuroPowerRAG v2 — Project Progress")
    st.caption("Live status of the trading signal pipeline. Refresh to update.")

    from pathlib import Path
    import json

    # ── Task registry ─────────────────────────────────────────────────────
    TASKS = [
        # (phase, id, label, status, human_required, note)
        ("Foundation", "F1", "Eval methodology (EVAL.md)", "done", False, ""),
        ("Foundation", "F2", "Price history dataset", "done", False, "Energy-Charts real data + synthetic GB"),
        ("Foundation", "F3", "Cost & latency budgets", "done", False, "src/utils/budget.py"),
        ("Bootstrap",  "T0", "Sentiment gold set (30 labels)", "human", True,  "Run: python -m src.evaluation.corpus_judge after adding ANTHROPIC_API_KEY"),
        ("Bootstrap",  "T2", "Anthropic SDK added", "done", False, "anthropic==0.97.0"),
        ("Bootstrap",  "T3", "Prompt caching wired", "done", False, "src/utils/anthropic_client.py"),
        ("Bootstrap",  "T7", "Eval log initialized", "done", False, "data/eval/eval_log.jsonl"),
        ("Phase 1",    "P1.2", "Sentiment scorer", "done", False, "src/sentiment/scorer.py (Gemini Flash)"),
        ("Phase 1",    "P1.3", "Sentiment κ eval gate", "pending", False, "Blocked by T0 (gold set)"),
        ("Phase 1",    "P1.4", "Sentiment wired into ingest", "done", False, "src/ingestion/rss_scraper.py"),
        ("Phase 1",    "P1.5", "Daily sentiment time series", "done", False, "src/sentiment/aggregator.py"),
        ("Phase 1",    "P1.6", "Streamlit sentiment chart", "done", False, "This app → Sentiment & Prices page"),
        ("Phase 2",    "P2.1", "Naive baselines", "done", False, "src/forecast/baseline.py"),
        ("Phase 2",    "P2.2", "ARIMA forecaster", "done", False, "src/forecast/arima.py (1,1,1)"),
        ("Phase 2",    "P2.3", "XGBoost forecaster", "done", False, "src/forecast/xgb.py"),
        ("Phase 2",    "P2.4", "XGBoost + sentiment", "done", False, "with_sentiment flag in xgb.py"),
        ("Phase 2",    "P2.5", "Forecast API cascade", "done", False, "src/forecast/api.py (XGB→ARIMA→baseline)"),
        ("Phase 3",    "P3.1", "Realized volatility", "done", False, "src/risk/volatility.py"),
        ("Phase 3",    "P3.2", "Kelly position sizer", "done", False, "src/risk/sizing.py (0.25× cap)"),
        ("Phase 3",    "P3.3", "Stop / target generator", "done", False, "1.5σ stop, 2.5σ target"),
        ("Phase 3",    "P3.4", "Risk walk-forward eval", "pending", False, "Needs backtest run"),
        ("Phase 4",    "P4.1", "TradeSignal schema", "done", False, "src/agent/schema.py"),
        ("Phase 4",    "P4.2", "Decision agent (Claude)", "done", False, "src/agent/decide.py + heuristic fallback"),
        ("Phase 4",    "P4.3", "Self-critique loop", "done", False, "Overrides to HOLD if confidence<0.6"),
        ("Phase 4",    "P4.4", "CLI tool", "done", False, "python -m src.agent.decide --country DE_LU --date …"),
        ("Phase 5",    "P5.1", "Backtest runner", "done", False, "src/backtest/runner.py"),
        ("Phase 5",    "P5.2", "Performance metrics", "done", False, "Sharpe, Sortino, max DD, win rate"),
        ("Phase 5",    "P5.3", "6-month walk-forward", "pending", False, "Needs price data + sentiment eval gate"),
        ("Phase 5",    "P5.4", "Paper trading scheduler", "done", False, "scheduler.py → paper_trades SQLite"),
        ("Phase 5",    "P5.5", "Streamlit dashboard", "done", False, "This app → Dashboard page"),
        ("Polish",     "PL1", "README v2 architecture", "done", False, ""),
        ("Polish",     "PL2", "Loom walkthrough", "human", True, "Record after P5.3 hits Sharpe ≥ 0.8"),
        ("Polish",     "PL3", "CI on GitHub Actions", "done", False, ""),
    ]

    STATUS_EMOJI = {"done": "✅", "pending": "⏳", "human": "👤", "blocked": "🔴"}
    STATUS_COLOR = {"done": "green", "pending": "orange", "human": "blue", "blocked": "red"}

    # ── Summary metrics ───────────────────────────────────────────────────
    total  = len(TASKS)
    done   = sum(1 for t in TASKS if t[3] == "done")
    human  = sum(1 for t in TASKS if t[3] == "human")
    pending = sum(1 for t in TASKS if t[3] in ("pending", "blocked"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tasks", total)
    c2.metric("✅ Done", done, delta=f"{done/total:.0%}")
    c3.metric("⏳ Pending", pending)
    c4.metric("👤 Human Required", human)

    st.progress(done / total, text=f"{done}/{total} tasks complete ({done/total:.0%})")
    st.divider()

    # ── Phase-by-phase breakdown ──────────────────────────────────────────
    import pandas as pd
    phases = list(dict.fromkeys(t[0] for t in TASKS))

    for phase in phases:
        phase_tasks = [t for t in TASKS if t[0] == phase]
        phase_done = sum(1 for t in phase_tasks if t[3] == "done")
        phase_total = len(phase_tasks)
        pct = phase_done / phase_total if phase_total else 0

        color = "green" if pct == 1.0 else ("orange" if pct > 0 else "red")
        with st.expander(f"{phase}  —  {phase_done}/{phase_total} done", expanded=(pct < 1.0)):
            for _, tid, label, status, is_human, note in phase_tasks:
                emoji = STATUS_EMOJI.get(status, "⚪")
                badge = " 👤 **Human action**" if is_human else ""
                st.markdown(f"{emoji} **{tid}** — {label}{badge}")
                if note:
                    st.caption(f"   {note}")
        st.progress(pct, text=f"{phase}: {phase_done}/{phase_total}")

    st.divider()

    # ── Price data coverage chart ─────────────────────────────────────────
    st.subheader("📈 Price History Coverage")
    price_path = Path("data/processed/prices_history.parquet")
    if price_path.exists():
        df_p = pd.read_parquet(price_path)
        df_p["date"] = pd.to_datetime(df_p["date"])

        # Source breakdown
        src_counts = df_p.groupby(["country", "source"]).size().reset_index(name="days")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Days per country**")
            country_days = df_p.groupby("country")["date"].count().reset_index(name="days")
            chart_bar = alt.Chart(country_days).mark_bar().encode(
                x=alt.X("days:Q", title="Trading days"),
                y=alt.Y("country:N", sort="-x"),
                color=alt.Color("country:N", legend=None),
                tooltip=["country", "days"]
            ).properties(height=200)
            st.altair_chart(chart_bar, use_container_width=True)
        with col_b:
            st.markdown("**Data source mix**")
            src_totals = df_p.groupby("source").size().reset_index(name="rows")
            chart_src = alt.Chart(src_totals).mark_arc().encode(
                theta="rows:Q",
                color=alt.Color("source:N", scale=alt.Scale(
                    domain=["energy_charts", "synthetic", "entsoe"],
                    range=["#2ecc71", "#f39c12", "#3498db"]
                )),
                tooltip=["source", "rows"]
            ).properties(height=200)
            st.altair_chart(chart_src, use_container_width=True)

        # Recent price chart
        st.markdown("**Last 90 days — DE_LU, FR, NL (real data)**")
        df_real = df_p[df_p["source"] == "energy_charts"].copy()
        df_recent = df_real[df_real["date"] >= df_real["date"].max() - pd.Timedelta(days=90)]
        if not df_recent.empty:
            chart_price = alt.Chart(df_recent).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("price_eur_mwh:Q", title="EUR/MWh", scale=alt.Scale(zero=False)),
                color="country:N",
                tooltip=["date", "country", "price_eur_mwh"]
            ).interactive().properties(height=280)
            st.altair_chart(chart_price, use_container_width=True)
            st.caption("Source: Energy-Charts API (api.energy-charts.info) — free, no token required")
    else:
        st.warning("Price history not yet downloaded. Run: `python -m src.ingestion.price_history`")

    st.divider()

    # ── Human action items ────────────────────────────────────────────────
    st.subheader("👤 Human Action Required")
    human_tasks = [t for t in TASKS if t[3] == "human"]
    for i, (_, tid, label, _, _, note) in enumerate(human_tasks):
        with st.container(border=True):
            st.markdown(f"**{i+1}. {tid} — {label}**")
            if note:
                st.info(note)

    # ── Eval log ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Eval Log")
    eval_log = Path("data/eval/eval_log.jsonl")
    if eval_log.exists():
        rows = [json.loads(l) for l in eval_log.read_text().splitlines() if l.strip()]
        if rows:
            df_log = pd.DataFrame(rows)
            df_log["passed"] = df_log["passed"].map({True: "✅ PASS", False: "❌ FAIL"})
            st.dataframe(df_log[["timestamp", "eval_name", "metric", "threshold", "passed"]],
                         use_container_width=True)
        else:
            st.info("No eval runs logged yet.")
    else:
        st.info("Eval log will appear here once evals are run.")

    st.stop()


st.title("European Power Market Q&A")
st.caption(
    "Ask questions about EU electricity prices, generation mix, and market news. "
    "Answers are grounded in ingested data with source citations."
)

# Example questions
with st.expander("Example questions"):
    examples = [
        "What was the average day-ahead price for Germany last week?",
        "What share of France's power comes from nuclear?",
        "How does wind contribute to European generation?",
        "What factors are driving electricity prices higher?",
        "Compare day-ahead prices between Spain and the Netherlands.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_query"] = ex

# Query input
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Your question",
    value=prefill,
    placeholder="e.g. What drove German baseload prices last week?",
)

# Run query
if query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            from src.pipeline import rag_chain

            result = rag_chain.query(
                question=query,
                country_filter=country_filter,
                doc_type_filter=doc_type_filter,
                date_from=date_from,
                date_to=date_to,
                n_results=n_results,
            )
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    # Answer
    st.divider()
    st.subheader("Answer")
    st.markdown(result.answer)

    # Source citations
    if result.sources:
        st.divider()
        st.subheader(f"Sources ({result.n_docs_retrieved} retrieved)")

        for i, doc in enumerate(result.sources, 1):
            m = doc.metadata
            label = (
                f"**[{i}]** {m.get('source', 'Unknown')} — "
                f"{m.get('doc_type', '?')} — "
                f"{m.get('country_name', m.get('country', '?'))} — "
                f"{m.get('date', 'date unknown')}"
            )
            with st.expander(label):
                st.text(doc.page_content[:600])
                if m.get("url"):
                    st.caption(f"URL: {m['url']}")
    else:
        st.info("No source documents retrieved. Run ingestion to populate the index.")

    # Query history
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    st.session_state["query_history"].insert(0, query)
    st.session_state["query_history"] = st.session_state["query_history"][:10]

# History in sidebar
if st.session_state.get("query_history"):
    with st.sidebar:
        st.divider()
        st.subheader("Recent queries")
        for past_q in st.session_state["query_history"][:5]:
            if st.button(past_q[:50] + ("..." if len(past_q) > 50 else ""), key=f"h_{past_q[:30]}"):
                st.session_state["prefill_query"] = past_q
                st.rerun()
